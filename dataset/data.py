from concurrent.futures import process
import os
import torch
import julius
import torchaudio
from torch.utils.data import Dataset
import random
import librosa
import time


# pre-load dataset and resample to 22.05KHz
class mel_dataset(Dataset):
    def __init__(self, process_config, train_config):
        self.dataset_name = train_config["dataset"]
        self.dataset_path = train_config["path"]["raw_path"]
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)

        sr = process_config["audio"]["or_sample_rate"]
        self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
        self.sample_list = []
        for idx in range(len(self.wavs)):
            audio_name = self.wavs[idx]
            if not audio_name.endswith('.wav'):
                continue
            # import pdb
            # pdb.set_trace()
            wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
            if wav.shape[1] > self.max_len:
                cuted_len = random.randint(5*sr, self.max_len)
                wav = wav[:, :cuted_len]
            wav = self.resample(wav[0,:].view(1,-1))
            # wav = wav[:,:self.max_len]
            sample = {
                "matrix": wav,
                "sample_rate": sr,
                "patch_num": 0,
                "pad_num": 0,
                "name": audio_name
            }
            self.sample_list.append(sample)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs



class mel_dataset_test(Dataset):
    def __init__(self, process_config, train_config):
        self.dataset_name = train_config["dataset"]
        self.dataset_path = train_config["path"]["raw_path_test"]
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()
        self.resample = julius.ResampleFrac(22050, 16000)
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)
    
    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
        # wav = self.resample(wav)
        # wav = wav[:,:self.max_len]
        # spect, phase = self.stft.transform(wav)
        sample = {
            "matrix": wav,
            "sample_rate": sr,
            "patch_num": 0,
            "pad_num": 0,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        # wavs = os.listdir(self.dataset_path)
        # return wavs
        wav_files = []
        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.wav'):
                wav_files.append(filename)
        return wav_files

class wav_dataset(Dataset):
    def __init__(self, config, flag='train'):
        t0=time.time()
        print(f'loading {flag} data...', end =' ')
        self.dataset_name = config["dataset_name"]
        raw_dataset_path = config["path"]["raw_path"]
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.sample_rate = config["audio"]["sample_rate"]
        self.max_wav_value = config["audio"]["max_wav_value"]
        self.win_len = config["audio"]["win_len"]
        self.max_len = config["audio"]["max_len"]
        # self.wavs = self.process_meta()[:10]
        self.wavs = self.process_meta()
        # n_fft = process_config["mel"]["n_fft"]
        # hop_length = process_config["mel"]["hop_length"]
        # self.stft = STFT(n_fft, hop_length)

        sr = config["audio"]["or_sample_rate"]
        self.resample = torchaudio.transforms.Resample(sr,self.sample_rate)
        self.sample_list = []

        # speed up validation
        if flag != 'test':
            self.length = len(self.wavs)
        else:
            self.length = min(100, len(self.wavs))
            

        for idx in range(self.length):
            audio_path = self.wavs[idx]
            wav, sr = torchaudio.load(os.path.join(audio_path))
            # if wav.shape[1] > self.max_len:
            #     cuted_len = random.randint(5*sr, self.max_len)
            #     wav = wav[:, :cuted_len]
            # wav = self.resample(wav[0,:].view(1,-1))

            wav = self.resample(wav)
            
            if flag != 'test':
                if wav.shape[-1]<=self.max_len:
                    repeat_num = (self.max_len // wav.shape[1]) + 1
                    wav = wav.repeat(1, repeat_num)
                tmp=wav.shape[-1]-self.max_len
                start=random.randint(0,tmp)
                wav = wav[...,start:start+self.max_len]

            sample = {
                "matrix": wav,
                "sample_rate": sr,
                "patch_num": 0,
                "pad_num": 0,
                "name": audio_path.split('/')[-1].split('.')[0],
                "trans_sr": config["audio"]["sample_rate"]
            }
            self.sample_list.append(sample)
        t1=time.time()
        print(f'finished! spend time: {t1-t0:.2f}')
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.sample_list[idx]

    def process_meta(self):
        # wavs = os.listdir(self.dataset_path)
        print(self.dataset_path)
        wavs = []
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.wav') or file.endswith('.flac'):
                    wavs.append(os.path.join(root, file))
        print(len(wavs))
        return wavs
