# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from struct import unpack, pack
import scipy.io.wavfile
import copy
import math

import librosa

#LIBROSA_MEL = False
USE_LIBROSA_MEL = True #False 
PREEMP = True
#PREEMP = False
USE_HAMMING = True
#USE_HAMMING = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        if USE_HAMMING:
            window = torch.hamming_window(win_length).float()
        else:
            window = torch.hann_window(win_length).float()

        if USE_LIBROSA_MEL:
            mel_basis = librosa.filters.mel(
                #sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
                sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax, htk=True, norm=None
                #sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax, htk=True
            )
            mel_basis = torch.from_numpy(mel_basis).float()
        else:
            mel_basis = lmfb_filter(sample_rate=sampling_rate, nch=n_mel_channels, fftsize=n_fft)
            mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        #p = (self.n_fft - self.hop_length) // 2
        #audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        audio = audio.squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        #log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        #log_mel_spec = torch.log10(mel_output)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=1e-8))
        return log_mel_spec

def lmfb_filter(sample_rate=16000, nch=40, fftsize=400):
    nbin = fftsize//2 + 1
    low_freq_mel = 0
    #high_freq_mel = 2595 * np.log10(1+ (sample_rate/2) /700)
    high_freq_mel = 1127 * np.log(1+ (sample_rate/2) /700)
    melcenter = np.linspace(low_freq_mel, high_freq_mel, nch + 2)
    fcenter = 700 * (10**(melcenter / 2595) -1)
    fres = sample_rate / fftsize
    fbank = np.zeros((nch, nbin))
    for c in range(nch):
        v1 = 1.0 / (fcenter[c + 1] - fcenter[c      ]) * fres * np.arange(nbin) - fcenter[c      ] / (fcenter[c + 1] - fcenter[c      ])
        v2 = 1.0 / (fcenter[c + 1] - fcenter[c + 2]) * fres * np.arange(nbin) - fcenter[c + 2] / (fcenter[c + 1] - fcenter[c + 2])
        fbank[c] = np.maximum(np.minimum(v1, v2), 0)
    return fbank

def load_wav_to_torch(full_path):
    #data, sampling_rate = librosa.core.load(full_path, sr=16000)
    sampling_rate, data = scipy.io.wavfile.read(full_path)  # File assumed to be in the same directory
    data = data.astype(np.float32)
    #data = 0.95 * librosa.util.normalize(data)
    if PREEMP:
        data = librosa.effects.preemphasis(data, coef=0.97)

    return torch.from_numpy(data).float(), sampling_rate
#
lmfb = Audio2Mel(n_fft=400, hop_length=160, win_length=400, sampling_rate=16000, n_mel_channels=80, mel_fmin=0.0, mel_fmax=None).to(DEVICE)

script = [line for line in open(sys.argv[1])]

for s in script:
    in_file, out_file = s.strip().split()
    audio, sampleing_rate = load_wav_to_torch(in_file)
    #print(s.strip())
    mel = lmfb(audio.unsqueeze(0).unsqueeze(1).to(DEVICE)) # input to lmfb is (batch_size, 1, timesteps)
    np.save(out_file, mel.squeeze(0).data.cpu().numpy().T)
