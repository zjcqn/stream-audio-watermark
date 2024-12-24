
import torch
import torch.nn as nn
from .blocks import ReluWav


class Detector1d(nn.Module):
    def __init__(self, msg_length, nlayers_decoder=6, hidden_size=128):
        super(Detector1d, self).__init__()
        print(msg_length, nlayers_decoder, hidden_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.EX = Watermark_Extracter_Wav(input_channel = 1, latent_dim = hidden_size, out_channel=msg_length, n_layers = nlayers_decoder+2)
        self.last_layer_det = nn.Linear(hidden_size, 1)
        self.last_layer_rec = nn.Linear(hidden_size, msg_length-1)

    def forward(self, y):
        hidden = self.EX(y)
        hidden = hidden.transpose(1,2)
        det = self.last_layer_det(hidden)
        msg = self.last_layer_rec(hidden)
        det = det.transpose(1,2)
        msg = msg.transpose(1,2)
        res = torch.cat((det, msg), dim=1)
        return res

    def detect(self, y):
        y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(y.device)
        y= y + y_gap
        extracted_wm_identity = self.EX(y.unsqueeze(1)).squeeze(1)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_Decoder(msg_identity)
        return msg
    
    def detect_temp(self, y):
        y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(y.device)
        y= y + y_gap

        hidden = self.EX(y)
        hidden = hidden.transpose(1,2)
        det = self.last_layer_det(hidden)
        msg = self.last_layer_rec(hidden)
        det = det.transpose(1,2)
        msg = msg.transpose(1,2)
        return det, msg
    

class Watermark_Extracter_Wav(nn.Module):
    def __init__(self, input_channel, out_channel, latent_dim, n_layers):
        super(Watermark_Extracter_Wav, self).__init__()
        wm_extracter = [ReluWav(c_in=input_channel, c_out=latent_dim, kernel_size=7, stride=1, padding=3)]
        for i in range(n_layers - 2):
            wm_extracter.append(ReluWav(c_in = latent_dim, c_out = latent_dim, padding = 14*i + 1, kernel_size = 14*2 * i + 3, stride = 1))
        wm_extracter.append(ReluWav(c_in = latent_dim, c_out = latent_dim, padding = 3, kernel_size = 7, stride = 1))

        self.main = nn.Sequential(*wm_extracter)
    
    def forward(self, x):
        return self.main(x)

class Watermark_Extracter_Wav_(nn.Module):
    def __init__(self, input_channel, out_channel, latent_dim, n_layers):
        super(Watermark_Extracter_Wav, self).__init__()
        wm_extracter = [ReluWav(c_in=input_channel, c_out=latent_dim, kernel_size=3, stride=1, padding=1)]
        for i in range(n_layers - 2):
            wm_extracter.append(ReluWav(c_in = latent_dim, c_out = latent_dim, padding = i + 1, kernel_size = 2 * i + 3, stride = 1))
        wm_extracter.append(ReluWav(c_in = latent_dim, c_out = latent_dim, padding = 1, kernel_size = 3, stride = 1))

        self.main = nn.Sequential(*wm_extracter)
    
    def forward(self, x):
        return self.main(x)


        