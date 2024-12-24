import os
import torch
import yaml

import typing as tp
import hydra
import datetime

from munch import DefaultMunch
import logging
from utils.logger import get_logger
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torchaudio
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.tools import load_ckpt,save_ckpt
from itertools import chain
# from torch.optim.lr_scheduler import StepLR
from scipy.stats import multivariate_normal
import random
import shutil
import wandb
import socket
import time
from distortions.audio_effects import (
    compress_with_encodec,
    get_audio_effects,
    select_audio_effects,
)

# from My_model.modules import Encoder, Decoder, Discriminator
from models.old_modules.UNet_modules import Embedder, Extractor, UNet_discriminator
from models.old_modules.modules import Discriminator, Decoder
from models.old_modules.msstft import MultiScaleSTFTDiscriminator

# from model.My_model.privacy_wm import WMEmbedder
# from models.WMBuilder import WMEmbedder, WMExtractor
from models.WMbuilder import WMEmbedder,WMExtractor
from dataset.data import wav_dataset as used_dataset

from losses.loss import Loss
from losses.loudnessloss import TFLoudnessRatio
from losses.loss import WMMbLoss, WMDetectionLoss, hinge_real_loss, hinge_fake_loss, hinge_loss, AdversarialLoss

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    for p in args["path"].values():
         os.makedirs(p, exist_ok=True)
    
    # ---------------- Init logger
    current_time = datetime.datetime.now().strftime("%y-%m-%d-%H:%M")
    logfile=os.path.join(args.path.log_dir, f"{args.experiment_name}#{current_time}.log")
    logger = get_logger(log_file=logfile)
    
    train_audios = used_dataset(config=args.dataset, flag='train')
    val_audios = used_dataset(config=args.dataset,flag='test')
    batch_size = args.optimize.batch_size
    assert batch_size < len(train_audios)
    train_audio_loader = DataLoader(train_audios, batch_size=batch_size, shuffle=True, num_workers=8)
    val_audios_loader = DataLoader(val_audios, batch_size=1, shuffle = False)

    # -------------- build model
    logger.info('building model')
    
    embedder = WMEmbedder(model_config=args.model, klass=args.embedder, nbits=args.watermark.msg_length).to(device)
    detector = WMExtractor(model_config=args.model, klass=args.detector, output_dim=32, nbits=args.watermark.msg_length).to(device)
    # discriminator = Discriminator(args.dataset).to(device)

    if getattr(args, "adv", None):
        discriminator = MultiScaleSTFTDiscriminator().to(device)
        d_op = Adam(
            params=chain(discriminator.parameters()),
            betas=args["optimize"]["betas"],
            eps=args["optimize"]["eps"],
            weight_decay=args["optimize"]["weight_decay"],
            lr = args["optimize"]["lr"]
        )
        adversary = AdversarialLoss(adversary=discriminator, optimizer=d_op)

    # -------------- audio effects
    effects = get_audio_effects(args.augmentation)  # noqa
    aug_weights = {}
    for name in effects.keys():
        aug_weights[name] =args.augmentation.aug_weights.get(name, -1)
    augmentations = {**effects}  # noqa

    # -------------- optimizer
    en_de_optim = Adam(
        params = chain(embedder.parameters(), detector.parameters()),
        betas = args["optimize"]["betas"],
        eps = args["optimize"]["eps"],
        weight_decay=args["optimize"]["weight_decay"],
        lr = args["optimize"]["lr"]
    )

    # -------------- continue from ckpt
    init_epoch = 0
    global_step = -1
    if getattr(args, "continue_from", False):
        module_state_dict=load_ckpt(args["continue_from"])
        embedder.load_state_dict(module_state_dict['embedder'])
        detector.load_state_dict(module_state_dict['detector'])
        discriminator.load_state_dict(module_state_dict['discriminator'])
        en_de_optim.load_state_dict(module_state_dict['en_de_optim'])
        d_op.load_state_dict(module_state_dict['d_optim'])
        # lr_sched.load_state_dict(module_state_dict['lr_sched'])
        # lr_sched_d.load_state_dict(module_state_dict['lr_sched_d'])
        global_step=module_state_dict['global_step']
        init_epoch=module_state_dict['epoch']+1

    # -------------- Loss 
    loss = Loss(args).to(device)

    # ---------------- train
    logger.info("Begin Trainging")
    lambda_wav = args["optimize"]["lambda_wav"]
    lambda_loud = args["optimize"]["lambda_loud"]
    lambda_msmel = args["optimize"]["lambda_msmel"]
    lambda_wm_det = args["optimize"]["lambda_wm_det"]
    lambda_wm_rec = args["optimize"]["lambda_wm_rec"]
    lambda_adv_d = args["optimize"]["lambda_adv_d"] # modify weights of m and a for better convergence
    lambda_adv_g = args["optimize"]["lambda_adv_g"] # modify weights of m and a for better convergence
    lambda_adv_g_map = args["optimize"]["lambda_adv_g_map"] # modify weights of m and a for better convergence

    adv_d_loss=0
    adv_g_loss=0

    for ep in range(init_epoch, args.iter.max_epoch):
        embedder.train()
        detector.train()
        discriminator.train()
        
        logger.info('Epoch {}/{}'.format(ep, args.iter.max_epoch))
        train_audio_loader = DataLoader(train_audios, batch_size=batch_size, shuffle=True)
        for step, sample in enumerate(train_audio_loader):
            if step >=args.iter.steps_per_epoch: break

            global_step += 1
            
            msg = np.random.choice([0, 1], [batch_size, 1, int(args.watermark.msg_length)])
            msg = torch.from_numpy(msg).float().to(device)

            x = sample["matrix"].to(device)

            wm = embedder(x=x, sample_rate=16_000, message=msg.squeeze(1))
            x, wm, mask = crop(x, wm, args.crop)
            x_wm = x+wm

            # decoded_identity = decoder.detect(x_wm)
            # decoded_identity_original = decoder.detect(x)

            decoded_identity_det, decoded_identity_rec = detector(x_wm)
            decoded_identity_original_det, decoded_identity_original_rec = detector(x)

            # decoded_identity_det=decoded_identity[:,:,:1]
            # decoded_identity_rec=decoded_identity[:,:,1:]

            # x_wm = embedder(x=x,  message=msg.squeeze(1)) 
            # decoded_identity_det, decoded_identity_rec = detector.get_logits(x_wm,16000)
            # decoded_identity_original_det, decoded_identity_original_rec = detector.get_logits(x,16000)

            wav_loss = loss.wav_loss(x, x_wm)
            loudness_loss = loss.loud_loss(x, x_wm)
            mel_loss = loss.mel_loss(x, x_wm)

            # wm_rec_identity_loss = loss.wm_rec_loss(msg, decoded_identity)
            # wm_det_identity_loss = loss.wm_det_loss(pos_msg=decoded_identity, neg_msg=decoded_identity_original)

            wm_det_identity_loss = loss.wm_det_loss(decoded_identity_det,decoded_identity_original_det,mask)
            wm_rec_identity_loss = loss.wm_rec_loss(decoded_identity_rec,mask,msg.squeeze(1))

            selected_augs = select_audio_effects(augmentations,
                aug_weights,
                mode=args.augmentation.select_aug_mode,
                max_length=args.augmentation.n_max_aug,
            )
            N_augs = 0
            decoded_det_augs = []
            decoded_rec_augs = []
            wm_det_aug_loss=0
            wm_rec_aug_loss=0
            adv_g_map_loss=0
            t_f=''
            for (
                    augmentation_name,
                    augmentation_method,
                ) in selected_augs.items():

                y_wm = x_wm
                aug_y_wm, mask_aug=augmentation_method(y_wm, mask=mask)    
                
                decoded_aug_det, decoded_aug_rec = detector(aug_y_wm)
                wm_det_aug_loss_ = loss.wm_det_loss(decoded_aug_det,decoded_identity_original_det,mask_aug)
                wm_rec_aug_loss_ = loss.wm_rec_loss(decoded_aug_rec,mask_aug,msg.squeeze(1))

                wm_det_aug_loss += wm_det_aug_loss_
                wm_rec_aug_loss += wm_rec_aug_loss_

                decoded_det_augs.append(decoded_aug_det)
                decoded_rec_augs.append(decoded_aug_rec)

                t_f+=f'{augmentation_name} '
                N_augs+=1
            # logger.info(t_f)
            if N_augs!=0:
                wm_det_aug_loss /= N_augs
                wm_rec_aug_loss /= N_augs
            else: 
                wm_det_aug_loss=wm_det_identity_loss.item()
                wm_rec_aug_loss=wm_rec_identity_loss.item()

            sum_loss =  lambda_wav * wav_loss + \
                        lambda_msmel * mel_loss + \
                        lambda_loud * loudness_loss + \
                        lambda_wm_det * (wm_det_aug_loss + wm_det_identity_loss) + \
                        lambda_wm_rec * (wm_rec_aug_loss + wm_rec_identity_loss)
            
            if global_step> args.adv.adv_starts_from:
                if step%args.adv.update_d_every==0: #控制disc 优化次数
                    # 提升discriminator的能力


                    adv_d_loss = adversary.train_adv(real=x, fake=x_wm.detach(), lambda_loss_d=lambda_adv_d)
                    # lr_sched_d.step()

                    # d_target_label_cover = torch.full((batch_size, 1), 1, device=device).float()
                    # d_on_cover = discriminator(x)
                    # adv_wav_loss = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)
                    # d_target_label_x_wm = torch.full((batch_size, 1), 0, device=device).float()
                    # d_on_x_wm = discriminator(x_wm.detach())

                    # # target label for x_wm images should be 'x_wm', because we want discriminator fight with embedder
                    # adv_wm_loss = F.binary_cross_entropy_with_logits(d_on_x_wm, d_target_label_x_wm)
                    # adv_d_loss=adv_wav_loss+adv_wm_loss
                    # adv_d_loss.backward()
                    # torch.nn.utils.clip_grad_value_(discriminator.parameters(), clip_value=100)
                    # d_op.step()
                    # d_op.zero_grad()
                    # lr_sched_d.step()

                # g_target_label_x_wm = torch.full((batch_size, 1), 1, device=device).float()
                # d_on_x_wm_for_enc = discriminator(x_wm)
                # adv_g_loss = F.binary_cross_entropy_with_logits(d_on_x_wm_for_enc, g_target_label_x_wm)


                # 判断x_wm后的音频是真还是假，我们想要它被判断成真
                adv_g_loss, adv_g_map_loss=adversary(fake=x_wm, real=x)
                
                adv_g_loss *= lambda_adv_g
                adv_g_loss = adv_g_loss*lambda_adv_g
                adv_g_map_loss = adv_g_map_loss*lambda_adv_g_map
                sum_loss += adv_g_loss + adv_g_map_loss


            sum_loss.backward()

            torch.nn.utils.clip_grad_value_(embedder.parameters(), clip_value=100)
            torch.nn.utils.clip_grad_value_(detector.parameters(), clip_value=100)
            
            en_de_optim.step()
            en_de_optim.zero_grad()
            # lr_sched.step()

            wm_identity_det_acc = compute_accuracy(decoded_identity_det, decoded_identity_original_det,mask).item()
            wm_identity_rec_acc = compute_bit_acc(decoded_identity_rec, msg.squeeze(1),mask).item()
            # wm_identity_det_acc = ((decoded_identity_det > 0.5).sum().float() / decoded_identity_det.numel()).item()
            # wm_identity_rec_acc = ((decoded_identity_rec > 0.5).eq(msg > 0.5).sum().float() / msg.numel()).item()


            wm_aug_det_acc=wm_identity_det_acc
            wm_aug_rec_acc=wm_identity_rec_acc
            if N_augs!=0:
                
                decoded_det_augs=torch.stack(decoded_det_augs)
                decoded_rec_augs=torch.stack(decoded_rec_augs)

                # wm_aug_det_acc = ((decoded_aug_det > 0.5).sum().float() / decoded_identity_det.numel()).item()
                # wm_aug_rec_acc = ((decoded_aug_rec > 0.5).eq(msg > 0.5).sum().float() / msg.numel()).item()
                
                wm_aug_det_acc = compute_accuracy(decoded_aug_det, decoded_identity_original_det,mask_aug).item()
                wm_aug_rec_acc = compute_bit_acc(decoded_aug_rec, msg.squeeze(1),mask_aug).item()

            zero_tensor = torch.zeros(x.shape).to(device)

            P_signal = torch.mean(x**2)
            P_noise = torch.mean(wm**2)
            snr = 20 * torch.log10(P_signal / P_noise)
            
            logger.info(  f"{global_step}/{args.iter.steps_per_epoch}({ep})[{N_augs}] | "+
                    f"wm_identity_det_acc:{wm_identity_det_acc:.3f}, " +
                    f"wm_identity_rec_acc:{wm_identity_rec_acc:.3f}, " +
                    f"wm_aug_det_acc:{wm_aug_det_acc:.3f}, " + 
                    f"wm_aug_rec_acc:{wm_aug_rec_acc:.3f}, " +
                    f"snr:{snr:.3f}, " + 
                    f"wav_loss:{wav_loss:.6f}, " + 
                    f"mel_loss:{mel_loss:.5f}, " +
                    f"loudness_loss: {loudness_loss:.3f}, " + 
                    f"wm_det_identity_loss:{wm_det_identity_loss:.3f}, " + 
                    f"wm_det_aug_loss:{wm_det_aug_loss:.3f}, " + 
                    f"wm_rec_identity_loss:{wm_rec_identity_loss:.3f} ," +
                    f"wm_rec_aug_loss:{wm_rec_aug_loss:.3f} ," + 
                    f"adv_d_loss:{adv_d_loss:.5f}, " + 
                    f"adv_g_loss:{adv_g_loss:.5f},"+
                    f"adv_g_map_loss:{adv_g_map_loss:.5f}"
                    )

            if global_step%args.iter.save_sample_by_step==0:
                save_audio_path = os.path.join(args.path.wm_speech, "{}_step{}.wav".format(args["attack_type"], global_step))
                torchaudio.save(save_audio_path, src = x_wm.detach().squeeze()[0].unsqueeze(0).to("cpu"), sample_rate = sample["trans_sr"][0])
                save_wm_path = os.path.join(args.path.wm_speech, "{}_step{}_wm.wav".format(args["attack_type"], global_step))
                torchaudio.save(save_wm_path, src = (wm.squeeze()[0]).detach().unsqueeze(0).to("cpu"), sample_rate = sample["trans_sr"][0])

                wandb.log({
                    "step": global_step,
                    "train/wm_identity_det_acc": wm_identity_det_acc,
                    "train/wm_identity_rec_acc": wm_identity_rec_acc,
                    "train/wm_aug_det_acc": wm_aug_det_acc,
                    "train/wm_aug_rec_acc": wm_aug_rec_acc,
                    "train/wav_loss": wav_loss,
                    "train/snr": snr,
                    "train/mel_loss": mel_loss,
                    "train/loudness_loss": loudness_loss,
                    "train/wm_det_identity_loss": wm_det_identity_loss,
                    "train/wm_det_aug_loss": wm_det_aug_loss,
                    "train/wm_rec_identity_loss": wm_rec_identity_loss,
                    "train/wm_rec_aug_loss": wm_rec_aug_loss,
                    "train/adv_d_loss": adv_d_loss,
                    "train/adv_g_loss": adv_g_loss,
                    "train/adv_g_map_loss": adv_g_map_loss
                }, step=global_step)
                
        if ep % args.iter.save_skpt_by_epoch == 0:

            module_state_dict={
                "embedder": embedder.state_dict(),
                "detector": detector.state_dict(),
                "discriminator": discriminator.state_dict(),
                "en_de_optim": en_de_optim.state_dict(),
                "d_optim":d_op.state_dict(),
                # "lr_sched":lr_sched.state_dict(),
                # "lr_sched_d":lr_sched_d.state_dict(),
                "global_step":global_step,
                "epoch":ep
            }

            save_ckpt(ckpt_dir=args.path.ckpt, base_name=args.experiment_name, module_state_dict=module_state_dict, epoch=ep)
            shutil.copyfile(os.path.realpath(__file__), os.path.join(args.path.ckpt, os.path.basename(os.path.realpath(__file__)))) # save training scripts

        if ep % args.iter.eval_by_epoch == 0:
            eval(args, embedder, detector, val_audios_loader, loss, logger, ep)



def eval(args, embedder, detector, val_audios_loader, loss, logger, ep):
    with torch.no_grad():
        embedder.eval()
        detector.eval()

        count = 0.0
        
        # @TODO save it as track
        eval_batch_size=1
        
        wav_loss=0.0
        loudness_loss=0.0
        mel_loss=0.0
        wm_rec_identity_loss=0.0
        wm_det_identity_loss=0.0

        wm_identity_det_acc=0.0
        wm_identity_rec_acc=0.0
        
        snr=0.0

        for i, sample in enumerate(val_audios_loader):
            count += 1
            # ---------------------- build watermark
            msg = np.random.choice([0, 1], [eval_batch_size, 1, int(args.watermark.msg_length)])
            msg = torch.from_numpy(msg).float()
            x = sample["matrix"].to(device)
            msg = msg.to(device)
            wm = embedder(x=x, message=msg.squeeze(1))
            x, wm, mask = crop(x, wm, args.crop)
            x_wm = x+wm

            decoded_identity_det, decoded_identity_rec  = detector(x_wm)
            decoded_identity_original_det, decoded_identity_original_rec = detector(x.detach())

            wav_loss += loss.wav_loss(x, x_wm).item()
            loudness_loss += loss.loud_loss(x, x_wm).item()
            mel_loss += loss.mel_loss(x, x_wm).item()

            wm_det_identity_loss += loss.wm_det_loss(decoded_identity_det,decoded_identity_original_det,mask).item()
            wm_rec_identity_loss += loss.wm_rec_loss(decoded_identity_rec,mask,msg.squeeze(1)).item()

            wm_identity_det_acc += compute_accuracy(decoded_identity_det, decoded_identity_original_det,mask).item()
            wm_identity_rec_acc += compute_bit_acc(decoded_identity_rec, msg.squeeze(1),mask).item()

            P_signal = torch.mean(x**2)
            P_noise = torch.mean(wm**2)
            snr += 20 * torch.log10(P_signal / P_noise)

        wav_loss /= count
        loudness_loss /= count
        mel_loss /= count
        wm_rec_identity_loss /= count
        wm_det_identity_loss /= count
        wm_identity_det_acc /= count
        wm_identity_rec_acc /= count
        snr /= count

        wandb.log({"step":ep,
                    "eval/wav_loss": wav_loss,
                    "eval/loudness_loss": loudness_loss,
                    "eval/mel_loss": mel_loss,
                    'eval/wm_rec_identity_loss': wm_rec_identity_loss,
                    'eval/wm_det_identity_loss': wm_det_identity_loss,
                    'eval/wm_rec_acc': wm_identity_rec_acc,
                    'eval/wm_det_acc': wm_identity_det_acc,
                    'eval/snr': snr
                    })
        
        logger.info(  f"eval epoch {ep}: " + 
                f"wav_loss:{wav_loss:.6f}, " + 
                f"mel_loss:{mel_loss:.5f}, " +
                f"loudness_loss: {loudness_loss:.3f}, " + 
                f"wm_det_identity_loss:{wm_det_identity_loss:.3f}, " + 
                f"wm_rec_identity_loss:{wm_rec_identity_loss:.3f} ," +
                f"wm_rec_acc:{wm_identity_rec_acc:.3f}, " + 
                f"wm_det_acc:{wm_identity_det_acc:.3f}, " + 
                f"snr:{snr:.3f}")



# 创建水印，选择一个分布来生成水印
def generate_watermark(batch_size, length, micro, sigma):
    sigmoid = torch.nn.Sigmoid()
    eye_matrix = np.eye(length)
    mask_convariance_maxtix = eye_matrix * (sigma ** 2)
    center = np.ones(length) * micro

    w_bit = multivariate_normal.rvs(mean = center, cov = mask_convariance_maxtix, size = [batch_size, 1])
    w_bit = torch.from_numpy(w_bit).float()
    return w_bit

def crop(signal, watermark, cfg ):
        """
        Applies a transformation to modify the watermarked signal to train localization.
        It can be one of the following:
            - zero padding: add zeros at the begining and the end of the signal
            - crop: crop the watermark apply a watermark only on some parts of the signal
            - shuffle: replace some part of the audio with other non watermarked parts
                from the batch
        In every cases the function returns a mask that contains indicates the parts that are or
        not watermarked

        Args:
            watermark (torch.Tensor): The watermark to apply on the signal.
            signal (torch.Tensor): clean signal
        Returns:
            watermark (torch.Tensor): modified watermark
            signal (torch.Tensor): modified signal
            mask (torch.Tensor): mask indicating which portion is still watermarked
        """
        assert (
            cfg.prob + cfg.shuffle_prob + cfg.pad_prob
            <= 1
        ), f"The sum of the probabilities {cfg.prob=} {cfg.shuffle_prob=} \
                {cfg.pad_prob=} should be less than 1"
        mask = torch.ones_like(watermark)
        p = torch.rand(1)
        if p < cfg.pad_prob:  # Pad with some probability
            start = int(torch.rand(1) * 0.33 * watermark.size(-1))
            finish = int((0.66 + torch.rand(1) * 0.33) * watermark.size(-1))
            mask[:, :, :start] = 0
            mask[:, :, finish:] = 0
            if torch.rand(1) > 0.5:
                mask = 1 - mask
            signal *= mask  # pad signal

        elif (
            p < cfg.prob + cfg.pad_prob + cfg.shuffle_prob
        ):
            # Define a mask, then crop or shuffle
            mask_size = round(watermark.shape[-1] * cfg.size)
            n_windows = int(
                torch.randint(1, cfg.max_n_windows + 1, (1,)).item()
            )
            window_size = int(mask_size / n_windows)
            for _ in range(n_windows):  # Create multiple windows in the mask
                mask_start = torch.randint(0, watermark.shape[-1] - window_size, (1,))
                mask[:, :, mask_start: mask_start + window_size] = (
                    0  # Apply window to mask
                )
            # inverse the mask half the time
            if torch.rand(1) > 0.5:
                mask = 1 - mask

            if p < cfg.pad_prob + cfg.shuffle_prob:  # shuffle
                # shuffle
                signal_cloned = signal.clone().detach()  # detach to be sure
                shuffle_idx = torch.randint(0, signal.size(0), (signal.size(0),))
                signal = signal * mask + signal_cloned[shuffle_idx] * (
                    1 - mask
                )  # shuffle signal where not wm

        watermark *= mask  # Apply mask to the watermark
        return signal, watermark, mask


def compute_accuracy(positive, negative, mask=None):

    if mask is not None:
        # cut last dim of positive to keep only where mask is 1
        # new_shape = [*positive.shape[:-1], -1]  # b nbits t -> b nbits -1
        # positive = torch.masked_select(positive, mask == 1).reshape(new_shape)
        acc = ((positive>0) == (mask==1)).float().mean()
        return acc
    else:
        N = (positive[:, 0, :] > 0).float().sum()/mask.sum() + (negative[:, 0, :] < 0).float().mean()
        
        acc = N / 2

        return acc


def compute_bit_acc(decoded, message, mask=None):
    """Compute bit accuracy.
    Args:
        positive: detector outputs [bsz, 2+nbits, time_steps]
        original: original message (0 or 1) [bsz, nbits]
        mask: mask of the watermark [bsz, 1, time_steps]
    """

    if mask is not None:
        # cut last dim of positive to keep only where mask is 1
        new_shape = [*decoded.shape[:-1], -1]  # b nbits t -> b nbits -1
        decoded = torch.masked_select(decoded, mask == 1).reshape(new_shape)
    # average decision over time, then threshold
    # decoded = decoded.mean(dim=-1) > 0  # b nbits
    decoded = decoded > 0  # b nbits
    res = (decoded == message.unsqueeze(-1)).float().mean()
    return res


@hydra.main(config_path="config", config_name="demucs_loc")
def run(args):

    cwd=os.getcwd()
    wandb.init(
            project=args.project_name,
            notes=socket.gethostname(),
            name=args.experiment_name+"_"+str(args.seed),
            group=args.scenario_name,
            job_type="training",
            reinit=True)
    
    
    # seet seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.chdir(cwd)
    main(args)

if __name__ == "__main__":
    run()


