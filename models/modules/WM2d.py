import math
import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F
from torch.nn import LeakyReLU, Tanh
from distortions.frequency import TacotronSTFT, fixed_STFT, tacotron_mel
from .blocks import ReluSpect, ReluWav, Spect_Encoder

class Detector2d(nn.Module):
    def __init__(self, block, latent_dim, msg_length, nlayers_decoder=6, n_fft=1024, hop_length=256, win_length=1024):
        super(Detector2d, self).__init__()

        self.mel_transform = TacotronSTFT(filter_length=n_fft, hop_length=hop_length, win_length=win_length)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.vocoder = get_vocoder(self.device)
        # self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((n_fft / 2) + 1)
        self.block = block
        self.EX = Watermark_Extracter(input_channel = 1, latent_dim = latent_dim, n_layers = nlayers_decoder)
        
        self.stft = fixed_STFT(n_fft, hop_length, win_length)
        
        self.msg_Decoder = Msg_after_Process(win_dim, msg_length)
        self.last_layer = nn.Linear(513, msg_length)
    
    def forward(self, y, source_audio, global_step, attack_type):
        spect, phase = self.stft.transform(y)
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        
        msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_Decoder(msg)
        
        return msg


    def detect(self, y):
        y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(y.device)
        y= y + y_gap

        spect_identity, phase_identity = self.stft.transform(y)
        extracted_wm_identity = self.EX(spect_identity.unsqueeze(1)).squeeze(1)
        print(y.shape,extracted_wm_identity.shape)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        
        # 没有传输损失的msg
        msg = self.msg_Decoder(msg_identity)
        return msg
    
    def detect_temp(self, y):
        y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(y.device)
        y= y + y_gap

        spect_identity, phase_identity = self.stft.transform(y)
        extracted_wm_identity = self.EX(spect_identity.unsqueeze(1)).squeeze(1)

        extracted_wm_identity=extracted_wm_identity.transpose(1,2)

        msg_identity = self.last_layer(extracted_wm_identity)
        msg_identity = msg_identity.transpose(1,2)
        # print(msg_identity.shape)

        return msg_identity[:,:1,:], msg_identity[:,1:,:]

class Watermark_Embedder(nn.Module):
    def __init__(self, input_channel, latent_dim, n_layers):
        super(Watermark_Embedder, self).__init__()
        wm_embedder = [ReluSpect(c_in = input_channel, c_out = latent_dim, kernel_size = 3, stride = 1, padding = 1)]
        for i in range(n_layers - 2):
            wm_embedder.append(ReluSpect(c_in = latent_dim, c_out = latent_dim, padding = i + 1, kernel_size = 2 * i + 3, stride = 1))
        
        wm_embedder.append(ReluSpect(c_in = latent_dim, c_out = 1, kernel_size = 3, stride = 1, padding = 1))

        self.main = nn.Sequential(*wm_embedder)
    
    def forward(self, x):
        return self.main(x)

class Watermark_Extracter(nn.Module):
    def __init__(self, input_channel, latent_dim, n_layers):
        super(Watermark_Extracter, self).__init__()
        wm_extracter = [ReluSpect(c_in=input_channel, c_out=latent_dim, kernel_size=3, stride=1, padding=1)]
        for i in range(n_layers - 2):
            wm_extracter.append(ReluSpect(c_in = latent_dim, c_out = latent_dim, padding = i + 1, kernel_size = 2 * i + 3, stride = 1))
        wm_extracter.append(ReluSpect(c_in = latent_dim, c_out = 1, padding = 1, kernel_size = 3, stride = 1))

        self.main = nn.Sequential(*wm_extracter)
    
    def forward(self, x):
        return self.main(x)



class Msg_Process(nn.Module):
    def __init__(self, input_channel, output_dim, activation=None, dropout=None):
        super(Msg_Process, self).__init__()
        self.linear_layer = nn.Sequential()
        self.linear_layer.add_module(
            "FC_1",
            nn.Linear(input_channel, (output_dim - 1) // 2 )
        )
        self.linear_layer.add_module(
            "FC_2",
            nn.Linear((output_dim - 1) // 2, output_dim)
        )
        nn.init.xavier_uniform_(self.linear_layer[0].weight)
        nn.init.xavier_uniform_(self.linear_layer[1].weight)

        nn.utils.spectral_norm(self.linear_layer[0])
        nn.utils.spectral_norm(self.linear_layer[1])

        if activation is not None:
            self.linear_layer.add_module("activ", activation)
        self.dropout = dropout
        
    def forward(self, msg):
        x = self.linear_layer(msg)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x


class Msg_after_Process(nn.Module):
    def __init__(self, input_channel, output_dim, activation=None, dropout=None):
        super(Msg_after_Process, self).__init__()
        self.linear_layer = nn.Sequential()
        self.linear_layer.add_module(
            "FC_1",
            nn.Linear(input_channel, input_channel // 2 )
        )
        self.linear_layer.add_module(
            "FC_2",
            nn.Linear(input_channel // 2, output_dim)
        )
        nn.init.xavier_uniform_(self.linear_layer[0].weight)
        nn.init.xavier_uniform_(self.linear_layer[1].weight)

        nn.utils.spectral_norm(self.linear_layer[0])
        nn.utils.spectral_norm(self.linear_layer[1])

        if activation is not None:
            self.linear_layer.add_module("activ", activation)
        self.dropout = dropout
        
    def forward(self, msg):
        x = self.linear_layer(msg)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x






        