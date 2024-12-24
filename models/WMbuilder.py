import julius
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .modules.seanet import SEANetWMEmbedder, SEANetEncoderKeepDimension
from .modules.demucs import Demucs, DemucsStreamer
from .modules.WM1d import Detector1d
from .modules.WM2d import Detector2d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MsgProcessor(torch.nn.Module):
    """
    Apply the secret message to the encoder output.
    Args:
        nbits: Number of bits used to generate the message. Must be non-zero
        hidden_size: Dimension of the encoder output
    """

    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = torch.nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Build the embedding map: 2 x k -> k x h, then sum on the first dim
        Args:
            hidden: The encoder output, size: batch x hidden x frames
            msg: The secret message, size: batch x k
        """
        # create indices to take from embedding layer
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # b x k -> b x k x h
        msg_aux = msg_aux.sum(dim=-2)  # b x k x h -> b x h
        msg_aux = msg_aux.unsqueeze(-1).repeat(
            1, 1, hidden.shape[2]
        )  # b x h -> b x h x t/f
        hidden = hidden + msg_aux  # -> b x h x t/f
        return hidden

    def forward_without_repeat(self, msg: torch.Tensor) -> torch.Tensor:
        # create indices to take from embedding layer
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # b x k -> b x k x h
        msg_aux = msg_aux.sum(dim=-2)  # b x k x h -> b x h
        msg_aux = msg_aux
        return msg_aux



class WMEmbedder(nn.Module):
    def __init__(self, model_config, klass, *args, nbits: int = 0, **kwargs):
        super(WMEmbedder, self).__init__()
        if klass == 'demucs':
            self.model=Demucs(**model_config['demucs'])
        if klass == 'seanet':
            self.model=SEANetWMEmbedder(**model_config['seanet_embedder'])

        self.msg_processor=MsgProcessor(nbits=nbits, hidden_size=model_config['msg_dimension']['demucs'])
        self._message = None


    @property
    def message(self):
        return self._message
    
    def message(self, message):
        self._message = message
    
    def get_watermark(
        self,
        x,
        sample_rate = 16000,
        message= None,
    ):
        """
        Get the watermark from an audio tensor and a message.
        If the input message is None, a random message of
        n bits {0,1} will be generated.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of the input audio (default 16khz as
                currently supported by the main AudioSeal model)
            message: An optional binary message, size: batch x k
        """
        length = x.size(-1)

        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)
        # hidden = self.encoder(x)

        
        if self.msg_processor is not None:
            if message is None:
                if self.message is None:
                    message = torch.randint(
                        0, 2, (x.shape[0], self.msg_processor.nbits), device=x.device
                    )
                else:
                    message = self.message.to(device=x.device)
            else:
                message = message.to(device=x.device)

            msg_hidden = self.msg_processor.forward_without_repeat(message)

        
        watermark = self.model(x, msg_hidden)

        # x_hidden = self.encoder(x)
        # watermark = self.decoder(x_hidden+msg_hidden)

        return watermark[..., :length]  # trim output cf encodec codebase


    def forward(
        self, x, sample_rate=None, message=None, alpha=1.0):
        """Apply the watermarking to the audio signal x with a tune-down ratio (default 1.0)"""
        wm = self.get_watermark(x, message=message)

        return alpha*wm

class WMExtractor(torch.nn.Module):
    def __init__(self, model_config, klass, *args, nbits: int = 0, **kwargs):
        super().__init__()
        if klass=='seanet_detector':
            self.encoder = SEANetEncoderKeepDimension(**model_config['seanet_detector'])
            self.last_layer = torch.nn.Conv1d(self.encoder.output_dim, 1 + nbits, 1)
            self.detector = torch.nn.Sequential(self.encoder, self.last_layer)

        if klass=='wm1d':
            self.detector=Detector1d(msg_length=nbits+1, **model_config['WM1d'])
        if klass=='wm2d':
            self.detector=Detector2d(msg_length=nbits+1, **model_config['WM2d'])
        self.nbits = nbits

    def detect_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
        message_threshold: float = 0.5,
    ) -> Tuple[float, torch.Tensor]:
        """
        A convenience function that returns a probability of an audio being watermarked,
        together with its message in n-bits (binary) format. If the audio is not watermarked,
        the message will be random.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of the input audio
            message_threshold: threshold used to convert the watermark output (probability
                of each bits being 0 or 1) into the binary n-bit message.
        """
        result, message = self.forward(x, sample_rate=sample_rate)  # b x 2+nbits
        detected = (
            torch.count_nonzero(torch.gt(result[:, 1, :], 0.5)) / result.shape[-1]
        )
        detect_prob = detected.cpu().item()  # type: ignore
        message = torch.gt(message, message_threshold).int()
        return detect_prob, message

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect the watermarks from the audio signal
        Args:
            x: Audio signal, size batch x frames
            sample_rate: The sample rate of the input audio
        """
        result = self.detector(x)
        det = result[:, :1, :]
        rec = result[:, 1:, :]
       
        return det, rec

