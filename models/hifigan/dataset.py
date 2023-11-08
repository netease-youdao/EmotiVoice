# Copyright 2023, YOUDAO
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import jsonlines
from transformers import AutoTokenizer
import os, sys
import numpy as np
from scipy.io.wavfile import read
from torch.nn.utils.rnn import pad_sequence
import copy
from models.prompt_tts_modified.tacotron_stft import  TacotronSTFT


def get_mel(filename, stft, sampling_rate, trim=False):

    sr, wav = read(filename)
    if sr != sampling_rate:
        raise ValueError("{} SR doesn't match target {} SR".format(sr, sampling_rate))

    wav = wav / 32768.0

    wav = torch.FloatTensor(wav.astype(np.float32))
    ### trimming ###
    if trim:
        frac = 0.005
        start = torch.where(
            torch.abs(wav)>(torch.abs(wav).max()*frac)
            )[0][0]
        end = torch.where(torch.abs(wav)>(torch.abs(wav).max()*frac))[0][-1]
        ### 50ms silence padding ###
        wav = torch.nn.functional.pad(wav[start:end], (sampling_rate//20, sampling_rate//20))
    melspec = stft.mel_spectrogram(wav.unsqueeze(0)) 

    return melspec.squeeze(0), wav 

def pad_mel(data, downsample_ratio, max_len ):
    batch_size = len(data)
    num_mels = data[0].size(0)
    padded = torch.zeros((batch_size, num_mels, max_len))
    for i in range(batch_size):
        lens = data[i].size(1)
        if lens % downsample_ratio!=0:
            data[i] = data[i][:,:-(lens % downsample_ratio)]
        padded[i, :, :data[i].size(1)] = data[i]
    
    return padded

class DatasetTTS(torch.utils.data.Dataset):
    def __init__(self, data_path, config):
        self.sampling_rate=config.sampling_rate
        self.datalist = self.load_files(data_path)
        self.stft = TacotronSTFT(
            filter_length=config.filter_length, 
            hop_length=config.hop_length, 
            win_length=config.win_length,
            n_mel_channels=config.n_mel_channels, 
            sampling_rate=config.sampling_rate, 
            mel_fmin=config.mel_fmin,
            mel_fmax=config.mel_fmax
        )
        self.trim = config.trim
        self.config=config


    def load_files(self, data_path):
        with jsonlines.open(data_path) as f:
            data = list(f)
        return data


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        uttid = self.datalist[index]["key"]


        mel, wav = get_mel(self.datalist[index]["wav_path"], self.stft, self.sampling_rate, trim=self.trim)
        
        return {
            "mel": mel,
            "uttid": uttid,
            "wav": wav,
        }
    
    
    def TextMelCollate(self, data):

        # Right zero-pad melspectrogram
        mel = [x['mel'] for x in data]
        max_target_len = max([x.shape[1] for x in mel])

        # wav
        wav = [x["wav"] for x in data]

        padded_wav = pad_sequence(wav,
                                  batch_first=True,
                                  padding_value=0.0)
        padded_mel = pad_mel(mel, self.config.downsample_ratio, max_target_len)
        
        mel_lens = torch.LongTensor([x.shape[1] for x in mel])  
        
        res = {
            "mel"               :   padded_mel,
            "mel_lens"          :   mel_lens,
            "wav"               :   padded_wav,
        }
        return res


