from base64 import encode
from multiprocessing import process
from re import L
import torch
import torch.nn as nn

from torch.nn import LeakyReLU, Tanh
from .blocks import FCBlock, PositionalEncoding, Mish, Conv1DBlock, Conv2Encoder, WatermarkEmbedder, WatermarkExtracter, ReluBlock, LSTM_Model
from .Myblocks import Msg_Process, Spect_Encoder, Watermark_Embedder, Watermark_Extracter, ReluBlock, Msg_after_Process, Watermark_Extracter_Wav
from distortions.frequency import TacotronSTFT, fixed_STFT, tacotron_mel
from distortions.dl import distortion

import os
import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display

import soundfile


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_spectrum(spect, phase, flag='linear'):
    import numpy as np
    import os
    import librosa
    import librosa.display
    root = "draw_figure"
    import matplotlib.pyplot as plt
    spect = spect/torch.max(torch.abs(spect))
    spec = librosa.amplitude_to_db(spect.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)
    phase = phase/torch.max(torch.abs(phase))
    spec = librosa.amplitude_to_db(phase.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.clim(-40, 40)
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_phase_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)

def save_feature_map(feature_maps):

    feature_maps = feature_maps.cpu().numpy()
    root = "draw_figure"
    output_folder = os.path.join(root,"feature_map_or")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    n_channels = feature_maps.shape[0]
    for channel_idx in range(n_channels):
        fig, ax = plt.subplots()
        ax.imshow(feature_maps[channel_idx, :, :], cmap='gray')
        ax.axis('off')
        output_file = os.path.join(output_folder, f'feature_map_channel_{channel_idx + 1}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

def save_waveform(a_tensor, flag='original'):

    root = "draw_figure"
    y = a_tensor.cpu().numpy()
    soundfile.write(os.path.join(root, flag + "_waveform.wav"), y, samplerate=22050)
    D = librosa.stft(y)
    spectrogram = np.abs(D)
    img=librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram_from_waveform.png'), bbox_inches='tight', pad_inches=0.0)

def freeze_model_and_submodules(model):
    for param in model.parameters():
        param.requires_grad = False

    for module in model.children():
        if isinstance(module, nn.Module):
            freeze_model_and_submodules(module)

class Encoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()
        
        self.name = "conv2"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["LSTM_dim"] + 2
        self.layers_EM = model_config["conv2"]["layers_EM"]
        self.LSTM_dim = model_config["conv2"]["LSTM_dim"]

        self.vocoder_step = model_config["structure"]["vocoder_step"]


        # spect encoder
        self.EN_spect = Spect_Encoder(input_channel=1, latent_dim=self.LSTM_dim, block=self.block, n_layers=self.layers_EM)

        # watermark linear encoder
        self.msg_linear_en = Msg_Process(msg_length, win_dim, activation = LeakyReLU(inplace=True))

        # watermarked_spect decoder
        self.DE_spect = Watermark_Embedder(self.EM_input_dim, self.LSTM_dim, self.layers_EM)

        # stft transform
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

    def forward(self, x, msg, global_step):
        num_samples = x.shape[2]
        # (1, 513, x)
        spect, phase = self.stft.transform(x)

        # (1, hidden_dim, 513, x)
        spect_encoded = self.EN_spect(spect.unsqueeze(1))

        # (1, 1, 513, x)
        watermark_encoded = self.msg_linear_en(msg).transpose(1, 2).unsqueeze(1).repeat(1, 1, 1, spect_encoded.shape[3])

        # (1, hidden_dim + 2, 513, x)
        concatenated_feature = torch.cat((spect_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)

        # (1, 1, 513, x)
        watermarked_spect = self.DE_spect(concatenated_feature)

        self.stft.num_samples = num_samples
        y = self.stft.inverse(watermarked_spect.squeeze(1), phase.squeeze(1))
        return y, watermarked_spect
    

class Decoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.robust = model_config["robust"]
        if self.robust:
            self.dl = distortion(process_config)
        self.mel_transform = TacotronSTFT(filter_length=process_config["mel"]["n_fft"], hop_length=process_config["mel"]["hop_length"], win_length=process_config["mel"]["win_length"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.vocoder = get_vocoder(self.device)
        # self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.EX = Watermark_Extracter(input_channel = 1, latent_dim = model_config["conv2"]["LSTM_dim"], n_layers = nlayers_decoder)
        
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
        
        self.msg_Decoder = Msg_after_Process(win_dim, msg_length)
        self.last_layer = nn.Linear(513, msg_length)
    
    def forward(self, y, source_audio, global_step, attack_type):
        spect, phase = self.stft.transform(y)
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        
        msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_Decoder(msg)
        
        return msg


    def detect(self, y):
        y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(device)
        y= y + y_gap

        spect_identity, phase_identity = self.stft.transform(y)
        extracted_wm_identity = self.EX(spect_identity.unsqueeze(1)).squeeze(1)
        print(y.shape,extracted_wm_identity.shape)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        
        # 没有传输损失的msg
        msg = self.msg_Decoder(msg_identity)
        return msg
    
    def detect_temp(self, y):
        y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(device)
        y= y + y_gap

        spect_identity, phase_identity = self.stft.transform(y)
        extracted_wm_identity = self.EX(spect_identity.unsqueeze(1)).squeeze(1)

        extracted_wm_identity=extracted_wm_identity.transpose(1,2)

        msg_identity = self.last_layer(extracted_wm_identity)
        msg_identity = msg_identity.transpose(1,2)
        # print(msg_identity.shape)

        return msg_identity[:,:1,:], msg_identity[:,1:,:]






class Decoder_Wav(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder_Wav, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.vocoder = get_vocoder(self.device)
        # self.vocoder_step = model_config["structure"]["vocoder_step"]
        

        self.EX = Watermark_Extracter_Wav(input_channel = 1, latent_dim = model_config["conv2"]["LSTM_dim"], out_channel=msg_length, n_layers = nlayers_decoder+2)
        self.last_layer_det = nn.Linear(model_config["conv2"]["LSTM_dim"], 1)
        self.last_layer_rec = nn.Linear(model_config["conv2"]["LSTM_dim"], msg_length-1)
    def forward(self, y, source_audio, global_step, attack_type):
        extracted_wm = self.EX(y.unsqueeze(1)).squeeze(1)
        
        msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1,2)
        
        return msg


    def detect(self, y):
        y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(device)
        y= y + y_gap
        extracted_wm_identity = self.EX(y.unsqueeze(1)).squeeze(1)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        
        # 没有传输损失的msg
        msg = self.msg_Decoder(msg_identity)
        return msg
    
    def detect_temp(self, y):
        y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(device)
        y= y + y_gap

        hidden = self.EX(y)
        hidden = hidden.transpose(1,2)
        det = self.last_layer_det(hidden)
        msg = self.last_layer_rec(hidden)
        det = det.transpose(1,2)
        msg = msg.transpose(1,2)
        return det, msg




def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


