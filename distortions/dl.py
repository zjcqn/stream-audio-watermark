from email.mime import audio
import os
import json
import torch
import random
import julius
import kornia
import pickle
import librosa

import numpy as np
import torch.nn as nn
import torchaudio as ta
import soundfile as sf
import torch.nn.functional as F
from collections import OrderedDict
from librosa.filters import mel as librosa_mel_fn

from scipy import signal
from librosa.core import load
from scipy.signal import get_window
from distutils.command import build
from numpy.random import RandomState
from distortions.mel_transform import STFT
from distortions.frequency import fixed_STFT, TacotronSTFT
from audiomentations import Compose, Mp3Compression
# from TTS.api import TTS
import torchaudio



os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def make_metadata(mel):
    import sys
    from math import ceil
    sys.path.append("autovc")
    from model_bl import D_VECTOR

    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
    c_checkpoint = torch.load('autovc/3000000-BL.ckpt')
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)
    num_uttrs = 10
    len_crop = 128
    
    mel = mel.to(device)
    emb = C(mel).detach().squeeze()
    return emb

SAMPLE_RATE = 22050
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class distortion(nn.Module):
    def __init__(self, process_config):
        super(distortion, self).__init__()
        self.resample_kernel1 = julius.ResampleFrac(SAMPLE_RATE, 16000).to(device)
        self.resample_kernel1_re = julius.ResampleFrac(16000, SAMPLE_RATE).to(device)
        self.resample_kernel2 = julius.ResampleFrac(SAMPLE_RATE, 8000).to(device)
        self.resample_kernel2_re = julius.ResampleFrac(8000, SAMPLE_RATE,).to(device)
        self.augment = Compose([Mp3Compression(p=1.0, min_bitrate=64, max_bitrate=64)])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.band_lowpass = julius.LowPassFilter(2000/SAMPLE_RATE).to(device)
        self.band_highpass = julius.HighPassFilter(500/SAMPLE_RATE).to(device)
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"]).to(self.device)
        self.process_config = process_config

    def none(self, x):
        return x

    def crop(self, x):
        length = x.shape[2]
        if length > 18000:
            start = random.randint(0,1000)
            end = random.randint(1,1000)
            y = x[:,:,start:0-end]
            # print(f"start:{start} and end:{end}")
            # pdb.set_trace()
        else:
            y = x
        return y
    
    def crop2(self, x):
        length = x.shape[2]
        if length > 18000:
            # import pdb
            # pdb.set_trace()
            cut_len = int(length * 0.1) # cut 10% off
            start = random.randint(0,cut_len-1)
            end = cut_len - start
            y = x[:,:,start:0-end]
            # print(f"start:{start} and end:{end}")
            # pdb.set_trace()
        else:
            y = x
        return y

    def resample(self, x):
        return x
    
    def crop_front(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio/100))
        ret = x[:,:,cut_len:]
        # print(f"{x.shape}:{ret.shape}")
        return ret
    
    def crop_middle(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio/100))
        begin = int((x.shape[-1] - cut_len) / 2)
        end = begin + cut_len
        # return torch.cat(x[:,:,:begin], x[:,:,end:],dim=2)
        ret = torch.cat([x[:,:,:begin], x[:,:,end:]],dim=2)
        return ret
    
    def crop_back(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio/100))
        begin = int((x.shape[-1] - cut_len))
        # return x[:,:,:begin]
        ret = x[:,:,:begin]
        # print(f"{x.shape}:{ret.shape}")
        return ret
    
    def resample1(self, y):        
        y = self.resample_kernel1_re(self.resample_kernel1(y))
        return y
    
    def resample2(self, y):        
        y = self.resample_kernel2_re(self.resample_kernel2(y))
        return y
    
    def white_noise(self, y, ratio): # SNR = 10log(ps/pn)
        SNR = ratio
        mean = 0.
        RMS_s = torch.sqrt(torch.mean(y**2, dim=2))
        RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))
        for i in range(y.shape[0]):
            noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
            if i == 0:
                batch_noise = noise
            else:
                batch_noise = torch.cat((batch_noise, noise), dim=0)
        batch_noise = batch_noise.unsqueeze(1).to(self.device)
        signal_edit = y + batch_noise
        return signal_edit
    
    def change_top(self, y, ratio=50):
        y = y*ratio/100
        return y
    
    def mp3(self, y, ratio=64):
        self.augment = Compose([Mp3Compression(p=1.0, min_bitrate=ratio, max_bitrate=ratio)])
        f = []
        a = y.cpu().detach().numpy()
        for i in a:
            f.append(torch.Tensor(self.augment(i,sample_rate=SAMPLE_RATE)))
        f = torch.cat(f,dim=0).unsqueeze(1).to(self.device)
        # y = y + (f - y).detach()
        # return y
        return f
    
    def recount(self, y):
        y2 = torch.tensor(np.array(y.cpu().squeeze(0).data.numpy()*(2**7)).astype(np.int8)) / (2**7)
        y2 = y2.to(self.device)
        y = y + (y2 - y).detach()
        return y
    
    def medfilt(self, y, ratio=3):
        y = kornia.filters.median_blur(y.unsqueeze(1), (1, ratio)).squeeze(1)
        return y
    
    def low_band_pass(self, y):
        y = self.band_lowpass(y)
        return y
    
    def high_band_pass(self, y):
        y = self.band_highpass(y)
        return y
    
    def modify_mel(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        spect = spect*ratio/100
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y
    
    def crop_mel_front(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(ratio/100))
        spect = spect*(torch.cat([torch.zeros(_,cut_len,time_len),torch.ones(_,fre_len-cut_len,time_len)], dim=1).to(self.device))
        return spect
    
    def crop_mel_back(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(ratio/100))
        spect = spect*(torch.cat([torch.ones(_,fre_len-cut_len,time_len),torch.zeros(_,cut_len,time_len)], dim=1).to(self.device))
        return spect
    
    def crop_mel_wave_front(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(ratio/100))
        spect = spect*(torch.cat([torch.zeros(_,cut_len,time_len),torch.ones(_,fre_len-cut_len,time_len)], dim=1).to(self.device))
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y
    
    def crop_mel_wave_back(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(ratio/100))
        spect = spect*(torch.cat([torch.ones(_,fre_len-cut_len,time_len),torch.zeros(_,cut_len,time_len)], dim=1).to(self.device))
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y
    
    def crop_mel_position(self, y, ratio=1):
        assert ratio >= 1 and ratio <= 10, "a must be an integer between 1 and 10"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len*(1/10))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        # spect = spect*(torch.cat([torch.zeros(_,cut_len,time_len),torch.ones(_,fre_len-cut_len,time_len)], dim=1).to(self.device))
        return spect
    
    def crop_mel_wave_position(self, y, ratio=1):
        num_samples = y.shape[2]
        assert ratio >= 1 and ratio <= 10, "a must be an integer between 1 and 10"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len*(1/10))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y
    

    def crop_mel_position_5(self, y, ratio=1):
        assert ratio >= 1 and ratio <= 20, "a must be an integer between 1 and 20"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len*(1/20))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        # spect = spect*(torch.cat([torch.zeros(_,cut_len,time_len),torch.ones(_,fre_len-cut_len,time_len)], dim=1).to(self.device))
        return spect
    
    def crop_mel_wave_position_5(self, y, ratio=1):
        num_samples = y.shape[2]
        assert ratio >= 1 and ratio <= 20, "a must be an integer between 1 and 20"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len*(1/20))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y
    
    def crop_mel_position_20(self, y, ratio=1):
        assert ratio >= 1 and ratio <= 5, "a must be an integer between 1 and 5"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len*(1/5))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        # spect = spect*(torch.cat([torch.zeros(_,cut_len,time_len),torch.ones(_,fre_len-cut_len,time_len)], dim=1).to(self.device))
        return spect
    
    def crop_mel_wave_position_20(self, y, ratio=1):
        num_samples = y.shape[2]
        assert ratio >= 1 and ratio <= 5, "a must be an integer between 1 and 5"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        # cut_len = int(fre_len*(ratio/100))
        cut_len = int(fre_len*(1/5))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def VoiceConversion(self, tgt_audio, src_path):
        import sys
        use_gpu = torch.cuda.is_available()
        mel_basis = librosa_mel_fn(sr = 22050, n_fft = 1024, n_mels = 80, fmin = 0, fmax = 8000)

        sys.path.append('Speech-Backbones/DiffVC/')
        import params
        from VCmodel import DiffVC

        import sys
        sys.path.append('DiffVC/hifi-gan/')
        from env import AttrDict
        from models import Generator as HiFiGAN

        sys.path.append('DiffVC/speaker_encoder/')
        from encoder import inference as spk_encoder
        from pathlib import Path

        def get_mel(wav):
            wav = wav[:(wav.shape[0] // 256)*256]
            wav = np.pad(wav, 384, mode='reflect')
            stft = librosa.core.stft(y = wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
            stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
            mel_spectrogram = np.matmul(mel_basis, stftm)
            log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
            return log_mel_spectrogram

        def get_embed(wav_path):
            wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
            embed = spk_encoder.embed_utterance(wav_preprocessed)
            return embed

        def noise_median_smoothing(x, w=5):
            y = np.copy(x)
            x = np.pad(x, w, "edge")
            for i in range(y.shape[0]):
                med = np.median(x[i:i+2*w+1])
                y[i] = min(x[i+w+1], med)
            return y

        def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
            mel_len = mel_source.shape[-1]
            energy_min = 100000.0
            i_min = 0
            for i in range(mel_len - silence_window):
                energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
                if energy_cur < energy_min:
                    i_min = i
                    energy_min = energy_cur
            estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
            if smoothing_window is not None:
                estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
            mel_denoised = np.copy(mel_synth)
            for i in range(mel_len):
                signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
                estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
                mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
            return mel_denoised

        # loading voice conversion model
        vc_path = 'Speech-Backbones/DiffVC/checkpts/vc/vc_vctk_wodyn.pt' # path to voice conversion model

        generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                        params.layers, params.kernel, params.dropout, params.window_size, 
                        params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                        params.beta_min, params.beta_max)
        if use_gpu:
            generator = generator.cuda()
            generator.load_state_dict(torch.load(vc_path))
        else:
            generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
        generator.eval()

        # loading HiFi-GAN vocoder
        hfg_path = 'Speech-Backbones/DiffVC/checkpts/vocoder/' # HiFi-GAN path

        with open(hfg_path + 'config.json') as f:
            h = AttrDict(json.load(f))

        if use_gpu:
            hifigan_universal = HiFiGAN(h).cuda()
            hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
        else:
            hifigan_universal = HiFiGAN(h)
            hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator',  map_location='cpu')['generator'])

        _ = hifigan_universal.eval()
        hifigan_universal.remove_weight_norm()

        # loading speaker encoder
        enc_model_fpath = Path('Speech-Backbones/DiffVC/checkpts/spk_encoder/pretrained.pt') # speaker encoder path
        if use_gpu:
            spk_encoder.load_model(enc_model_fpath, device="cuda")
        else:
            spk_encoder.load_model(enc_model_fpath, device="cpu")

        # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
        # src_path = '/amax/home/Tiamo/Traceable_watermark/Speech-Backbones/DiffVC/example/6415_111615_000012_000005.wav' # path to source utterance
        # tgt_path = '/amax/home/Tiamo/Traceable_watermark/Speech-Backbones/DiffVC/example/8534_216567_000015_000010.wav' # path to reference utterance

        mel_target = torch.from_numpy(get_mel(tgt_audio.squeeze().detach().cpu().numpy())).float().unsqueeze(0)
        if use_gpu:
            mel_target = mel_target.cuda()
        mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
        if use_gpu:
            mel_target_lengths = mel_target_lengths.cuda()

        src_audio, _ = load(src_path, sr=22050)
        mel_source = torch.from_numpy(get_mel(src_audio)).float().unsqueeze(0)
        if use_gpu:
            mel_source = mel_source.cuda()
        mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
        if use_gpu:
            mel_source_lengths = mel_source_lengths.cuda()

        embed_target = torch.from_numpy(get_embed(tgt_audio.squeeze().detach().cpu().numpy())).float().unsqueeze(0)
        if use_gpu:
            embed_target = embed_target.cuda()

        # performing voice conversion
        mel_encoded, mel_ = generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target, 
                                            n_timesteps=30, mode='ml')
        mel_synth_np = mel_.cpu().detach().squeeze().numpy()
        mel_source_np = mel_.cpu().detach().squeeze().numpy()
        mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
        if use_gpu:
            mel = mel.cuda()
        
        with torch.no_grad():
            source_audio = hifigan_universal.forward(mel_source).cpu().squeeze().clamp(-1, 1)
            source_audio = hifigan_universal.forward(mel_target).cpu().squeeze().clamp(-1, 1)
            conversed_audio = hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)
        
        sf.write('conversed_audio.wav', conversed_audio, samplerate = SAMPLE_RATE)
        return conversed_audio.unsqueeze(0).unsqueeze(0).to(device)


    def AutoVC(self, tgt_audio, src_path):
        import sys
        import torchaudio as ta
        sys.path.append('deepFake/autovc')
        from converision import get_vc_spect
        from make_metadata import get_emb
        from make_spect import get_spect, get_spect_from_wav

        src_spectrum = get_spect(src_path)
        src_embedding = get_emb(src_spectrum)

        tgt_audio = tgt_audio.cpu().squeeze(0).numpy()
        if tgt_audio.shape[0] == 1:
            tgt_audio = tgt_audio.squeeze(0)
        tgt_spectrum = get_spect_from_wav(tgt_audio)
        tgt_embdedding = get_emb(tgt_spectrum)

        vc_spectrum = get_vc_spect(src_spectrum, src_embedding, tgt_embdedding, device)

        sys.path.append('deepFake/autovc/hifi_gan')
        from inference_e2e import inference_from_spec

        audio = inference_from_spec(vc_spectrum.T, device)
        return audio.unsqueeze(0).to(device)

    
    def YourTTS(self, tgt_audio):
        import torch, gc

        gc.collect()
        torch.cuda.empty_cache()

        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
        with torch.no_grad():
            wav = tts.tts(text = "Hello world, my name is john, nice to meet you", speaker_wav = tgt_audio, language = "en").to(device)
        return wav.unsqueeze(0).unsqueeze(0)
    
    # DiffVC: 22050Hz
    def DiffVC(self, tgt_audio, src_path):
        mel_basis = librosa_mel_fn(sr = 22050, n_fft = 1024, n_mels = 80, fmin = 0, fmax = 8000)
        import sys
        sys.path.append('deepFake/DiffVC/speaker_encoder')
        from encoder import inference as spk_encoder
        import argparse
        import IPython.display as ipd
        from scipy.io.wavfile import write

        sys.path.append('deepFake/DiffVC')
        sys.path.append('deepFake/DiffVC/hifi-gan')

        import params
        from DiffVCmodel import DiffVC
        from env import AttrDict
        from models import Generator as HiFiGAN
        from pathlib import Path

        # loading voice conversion model
        vc_path = 'deepFake/DiffVC/checkpts/vc/vc_libritts_wodyn.pt' # path to voice conversion model
        diffvc_generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads, 
                params.layers, params.kernel, params.dropout, params.window_size, 
                params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim, 
                params.beta_min, params.beta_max)
        diffvc_generator = diffvc_generator.to(device)
        diffvc_generator.load_state_dict(torch.load(vc_path))
        diffvc_generator.eval()
        
        # loading HiFi-GAN vocoder
        hfg_path = 'deepFake/DiffVC/checkpts/vocoder/'
        with open(hfg_path + 'config.json') as f:
            h = AttrDict(json.load(f))
        
        diffvc_hifigan_universal = HiFiGAN(h).to(device)
        diffvc_hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])

        _ = diffvc_hifigan_universal.eval()
        diffvc_hifigan_universal.remove_weight_norm()

        # loading speaker encoder
        enc_model_fpath = Path('deepFake/DiffVC/checkpts/spk_encoder/pretrained.pt')
        spk_encoder.load_model(enc_model_fpath, device=device)

        # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
        def get_mel(wav):
            wav = wav[:(wav.shape[0] // 256)*256]
            wav = np.pad(wav, 384, mode='reflect')
            stft = librosa.core.stft(y = wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
            stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
            mel_spectrogram = np.matmul(mel_basis, stftm)
            log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
            return log_mel_spectrogram
        
        def get_embed(wav_path):
            wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
            embed = spk_encoder.embed_utterance(wav_preprocessed)
            return embed

        src_audio, _ = load(src_path, sr=22050)
        mel_source = torch.from_numpy(get_mel(src_audio)).float().unsqueeze(0).to(device)
        mel_source_lengths = torch.LongTensor([mel_source.shape[-1]]).to(device)

        mel_target = torch.from_numpy(get_mel(tgt_audio.squeeze().detach().cpu().numpy())).float().unsqueeze(0).to(device)
        mel_target_lengths = torch.LongTensor([mel_target.shape[-1]]).to(device)

        embed_target = torch.from_numpy(get_embed(tgt_audio.squeeze().detach().cpu().numpy())).float().unsqueeze(0).to(device)

        # performing voice conversion
        def noise_median_smoothing(x, w=5):
            y = np.copy(x)
            x = np.pad(x, w, "edge")
            for i in range(y.shape[0]):
                med = np.median(x[i:i+2*w+1])
                y[i] = min(x[i+w+1], med)
            return y
        
        def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
            mel_len = mel_source.shape[-1]
            energy_min = 100000.0
            i_min = 0
            for i in range(mel_len - silence_window):
                energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
                if energy_cur < energy_min:
                    i_min = i
                    energy_min = energy_cur
            estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
            if smoothing_window is not None:
                estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
            mel_denoised = np.copy(mel_synth)
            for i in range(mel_len):
                signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
                estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
                mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
            return mel_denoised
        
        mel_encoded, mel_ = diffvc_generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target, 
                                            n_timesteps=30, mode='ml')
        mel_synth_np = mel_.cpu().detach().squeeze().numpy()
        mel_source_np = mel_.cpu().detach().squeeze().numpy()
        mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            audio = diffvc_hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)

        ta.save("results/after-vc-diffvc.wav", audio.unsqueeze(0), 22050)

        return audio.unsqueeze(0).unsqueeze(0).to(device)
    
    # VQMIVC: 16000Hz
    def VQMIVC(self, tgt_audio, src_path):
        import sys
        import resampy
        import kaldiio
        import subprocess
        import pyworld as pw
        import soundfile as sf

        sys.path.append('deepFake/VQMIVC')
        from model_encoder import Encoder, Encoder_lf0
        from model_decoder import Decoder_ac
        from model_encoder import SpeakerEncoder as Encoder_spk
        from spectrogram import logmelspectrogram

        out_dir = 'results/vqmivc'
        os.makedirs(out_dir, exist_ok=True)
        out_filename = 'vqmivc'

        encoder = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256)
        encoder_lf0 = Encoder_lf0()
        encoder_spk = Encoder_spk()
        decoder = Decoder_ac(dim_neck=64)
        encoder.to(device)
        encoder_lf0.to(device)
        encoder_spk.to(device)
        decoder.to(device)
        
        checkpoint_path = 'deepFake/VQMIVC/vqmivc_model/checkpoints/useCSMITrue_useCPMITrue_usePSMITrue_useAmpTrue/VQMIVC-model.ckpt-500.pt'
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint['encoder'])
        encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        decoder.load_state_dict(checkpoint["decoder"])

        encoder.eval()
        encoder_spk.eval()
        decoder.eval()

        mel_stats = np.load('./deepFake/VQMIVC/mel_stats/stats.npy')
        mean = mel_stats[0]
        std = mel_stats[1]

        def extract_logmel(wav, mean, std, fs=16000):
            if fs != 16000:
                wav = resampy.resample(wav, fs, 16000, axis=0)
                fs = 16000
            assert fs == 16000
            peak = np.abs(wav).max()
            if peak > 1.0:
                wav /= peak
            mel = logmelspectrogram(
                        x=wav,
                        fs=fs,
                        n_mels=80,
                        n_fft=400,
                        n_shift=160,
                        win_length=400,
                        window='hann',
                        fmin=80,
                        fmax=7600,
                    )
            
            mel = (mel - mean) / (std + 1e-8)
            tlen = mel.shape[0]
            frame_period = 160/fs*1000
            f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
            f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
            f0 = f0[:tlen].reshape(-1).astype('float32')
            nonzeros_indices = np.nonzero(f0)
            lf0 = f0.copy()
            lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
            mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
            lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
            return mel, lf0


        feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir)+'/feats.1'))
        src_wav, sr = sf.read(src_path)
        src_mel, src_lf0 = extract_logmel(src_wav, mean, std, sr)
        ref_mel, _ = extract_logmel(tgt_audio.squeeze().detach().cpu().numpy(), mean, std, 22050)
        src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
        src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)
        ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
        
        with torch.no_grad():
            print('ref_mel:', ref_mel.shape)
            z, _, _, _ = encoder.encode(src_mel)
            lf0_embs = encoder_lf0(src_lf0)
            spk_emb = encoder_spk(ref_mel)
            output = decoder(z, lf0_embs, spk_emb)
            
            feat_writer[out_filename+'_converted'] = output.squeeze(0).cpu().numpy()
            feat_writer[out_filename+'_source'] = src_mel.squeeze(0).cpu().numpy().T
            feat_writer[out_filename+'_reference'] = ref_mel.squeeze(0).cpu().numpy().T

        feat_writer.close()
        print('synthesize waveform...')
        cmd = ['parallel-wavegan-decode', '--checkpoint', \
            './deepFake/VQMIVC/vocoder/checkpoint-3000000steps.pkl', \
            '--scp', f'{str(out_dir)}/feats.1.scp', '--outdir', str(out_dir)]
                #'--feats-scp', f'{str(out_dir)}/feats.1.scp', '--outdir', str(out_dir)]
        subprocess.call(cmd)

        wav, sr = ta.load(f'{str(out_dir)}/{out_filename}_converted_gen.wav')
        resampler = torchaudio.transforms.Resample(sr, 22050)
        wav = resampler(wav)
        ta.save("results/after-vc-vqmivc.wav", wav, 22050)
        return wav.unsqueeze(0).to(device)
    
    def VALLEX(self, tgt_audio, audio_source):
        import sys
        sys.path.append('deepFake/VALL_E_X')
        from vallexutils.prompt_making import make_prompt_by_audio
        from vallexutils.generation import SAMPLE_RATE, generate_audio, preload_models
        from scipy.io.wavfile import write as write_wav
        make_prompt_by_audio(name="vallex", wav_pr=tgt_audio.squeeze(0).cpu(), source_wav=audio_source.squeeze(0).cpu(), sr=22050)
        preload_models()
        text_prompt = """
        Hello world, my name is john, nice to meet you.
        """
        audio = generate_audio(text_prompt, prompt="vallex")
        result = torch.tensor(audio).unsqueeze(0).cpu()
        resampler = torchaudio.transforms.Resample(SAMPLE_RATE, 22050)
        result = resampler(result)
        return result.unsqueeze(0).to(device)
    
    # FreeVC: 16000Hz
    def FreeVC(self, tgt_audio, src_path):
        import sys
        sys.path.append('deepFake/FreeVC')
        import freevcutils
        from models import SynthesizerTrn
        from speaker_encoder.voice_encoder import SpeakerEncoder

        hpfile = "deepFake/FreeVC/configs/freevc.json"
        ptfile = "deepFake/FreeVC/checkpoints/freevc.pth"

        hps = freevcutils.get_hparams_from_file(hpfile)

        net_g = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model).to(device)
        _ = net_g.eval()
        _ = freevcutils.load_checkpoint(ptfile, net_g, None, True)

        cmodel = freevcutils.get_cmodel(0)

        if hps.model.use_spk:
            smodel = SpeakerEncoder('deepFake/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt')

        wav_tgt = tgt_audio.squeeze(0).cpu()
        resampler = ta.transforms.Resample(orig_freq=22050, new_freq=hps.data.sampling_rate)
        wav_tgt = resampler(wav_tgt)
        wav_tgt = wav_tgt.squeeze(0).numpy()
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        g_tgt = smodel.embed_utterance(wav_tgt)
        g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)

        wav_src, _ = librosa.load(src_path, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(device)
        c = freevcutils.get_content(cmodel, wav_src)

        audio = net_g.infer(c, g=g_tgt)
        audio = audio.squeeze(0).cpu()
        resampler = ta.transforms.Resample(hps.data.sampling_rate, 22050)
        audio = resampler(audio)
        return audio.unsqueeze(0).to(device)



    def forward(self, x, x_source, attack_choice=1, ratio=10, src_path=None):
        attack_functions = {
            0: lambda x: self.none(x),
            1: lambda x: self.crop(x),
            2: lambda x: self.crop2(x),
            3: lambda x: self.resample(x),
            4: lambda x: self.crop_front(x, ratio),     # Cropping front
            5: lambda x: self.crop_middle(x, ratio),    # Cropping middle
            6: lambda x: self.crop_back(x, ratio),      # Cropping behind
            7: lambda x: self.resample1(x),             # Resampling 16KHz
            8: lambda x: self.resample2(x),             # Resampling 8KHz
            9: lambda x: self.white_noise(x, ratio),    # Gaussian Noise with SNR ratio/2 dB
            10: lambda x: self.change_top(x, ratio),    # Amplitude Scaling ratio%
            11: lambda x: self.mp3(x, ratio),           # MP3 Compression ratio Kbps
            12: lambda x: self.recount(x),              # Recount 8 bps
            13: lambda x: self.medfilt(x, ratio),       # Median Filtering with ratio samples as window
            14: lambda x: self.low_band_pass(x),        # Low Pass Filtering 2000 Hz
            15: lambda x: self.high_band_pass(x),       # High Pass Filtering 500 Hz 
            16: lambda x: self.modify_mel(x, ratio),    # don't need
            17: lambda x: self.crop_mel_front(x, ratio),# don't need        
            18: lambda x: self.crop_mel_back(x, ratio), # don't need        
            19: lambda x: self.crop_mel_wave_front(x, ratio),   # don't need
            20: lambda x: self.crop_mel_wave_back(x, ratio),    # mask from top with ratio "ratio" and transform back to wav
            21: lambda x: self.crop_mel_position(x, ratio),     # mask 10% at position "ratio"
            22: lambda x: self.crop_mel_wave_position(x, ratio),# mask 10% at position "ratio" and transform back to wav
            
            23: lambda x: self.crop_mel_position_5(x, ratio),       # mask 5% at position "ratio"
            24: lambda x: self.crop_mel_wave_position_5(x, ratio),  # mask 5% at position "ratio" and transform back to wav
            25: lambda x: self.crop_mel_position_20(x, ratio),      # mask 20% at position "ratio"
            26: lambda x: self.crop_mel_wave_position_20(x, ratio), # mask 20% at position "ratio" and transform back to wav
            # 27: lambda x: self.VoiceConversion(x, src_path),
            # 28: lambda x: self.AutoVC(x, src_path),
            # 29: lambda x: self.YourTTS(x),
            # 30: lambda x: self.DiffVC(x, src_path),
            # 31: lambda x: self.VQMIVC(x, src_path),
            # 32: lambda x: self.VALLEX(x, x_source),
            # 33: lambda x: self.FreeVC(x, src_path)
        }

        x = x.clamp(-1, 1)
        y = attack_functions[attack_choice](x)
        y = y.clamp(-1, 1)
        return y

