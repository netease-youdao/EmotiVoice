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
from models.prompt_tts_modified.simbert import StyleEncoder
from models.prompt_tts_modified.tacotron_stft import  TacotronSTFT
import models.prompt_tts_modified.feats as feats


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

class Dataset_PromptTTS(torch.utils.data.Dataset):
    def __init__(self, data_path, config, style_encoder):
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
        self.pitch_extractor = feats.Pitch(sr=config.sampling_rate,
                                hop_length=config.hop_length,
                                pitch_min=config.pitch_min,
                                pitch_max=config.pitch_max)
        self.energy_extractor = feats.Energy(sr=config.sampling_rate,
                                    n_fft=config.filter_length,
                                    hop_length=config.hop_length,
                                    win_length=config.win_length,
                                    window=config.window)
        self.pitch_stats=config.pitch_stats
        self.energy_stats=config.energy_stats

        # Phoneme
        with open(config.token_list_path, encoding='utf-8') as f:
            self.token2id = {t.strip():idx for idx, t in enumerate(f.readlines())}
        # Speaker
        with open(config.speaker2id_path, encoding='utf-8') as f:
            self.speaker2id = {t.strip():idx for idx, t in enumerate(f.readlines())}

        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
        self.style_encoder = style_encoder
        self.style_encoder.eval()
        self.content_dir=f"{config.tmp_dir}/content"
        self.style_dir=f"{config.tmp_dir}/style"
        os.makedirs(self.content_dir, exist_ok=True)
        os.makedirs(self.style_dir, exist_ok=True)


    def get_style_embedding(self, uttid, prompt, dir):
        path = f"{dir}/{uttid}.npy"
        try:
            style_embedding = np.load(path)
        except:
            prompt = self.tokenizer([prompt], return_tensors="pt")
            input_ids = prompt["input_ids"]
            token_type_ids = prompt["token_type_ids"]
            attention_mask = prompt["attention_mask"]
            with torch.no_grad():
                output = self.style_encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            style_embedding = output["pooled_output"].cpu().squeeze().numpy()
            np.save(path, style_embedding)
        return style_embedding

    def load_files(self, data_path):
        with jsonlines.open(data_path) as f:
            data = list(f)
        return data

    def get_pitch(self, wav, pitch_stats):
        if type(wav) == torch.Tensor:
            wav=wav.numpy()
        pitch = self.pitch_extractor.get_pitch(wav, use_token_averaged_pitch=False)
        pitch = (pitch - pitch_stats[0]) / pitch_stats[1]
        return pitch
    
    def get_energy(self, wav, energy_stats):
        if type(wav) == torch.Tensor:
            wav=wav.numpy()
        energy = self.energy_extractor.get_energy(wav, use_token_averaged_energy=False)
        energy = (energy - energy_stats[0]) / energy_stats[1]
        return energy

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        uttid = self.datalist[index]["key"]
        text_int=[self.token2id[t] for t in self.datalist[index]["text"]]

        mel, wav = get_mel(self.datalist[index]["wav_path"], self.stft, self.sampling_rate, trim=self.trim)
        
        # pitch
        pitch = torch.from_numpy(self.get_pitch(wav, self.pitch_stats))
        # energy
        energy = torch.from_numpy(self.get_energy(wav, self.energy_stats))
        # speaker 
        speaker = self.speaker2id[self.datalist[index]["speaker"]]


        style_embedding = self.get_style_embedding(uttid, self.datalist[index]["prompt"], self.style_dir)

        content_embedding = self.get_style_embedding(uttid, self.datalist[index]["original_text"], self.content_dir)

        return {
            "phoneme_id":torch.from_numpy(np.array(text_int)),
            "mel": mel,
            "uttid": uttid,
            "style_embedding": torch.from_numpy(style_embedding),
            "content_embedding": torch.from_numpy(content_embedding),
            "pitch": pitch,
            "energy": energy,
            "speaker": speaker,
            "wav": wav,
        }
    
    
    def TextMelCollate(self, data):

        phoneme_id = [x['phoneme_id'] for x in data]
        phoneme_lens = torch.LongTensor([x.shape[0] for x in phoneme_id])
        phoneme_id = pad_sequence(phoneme_id, batch_first=True)

        # Right zero-pad melspectrogram
        mel = [x['mel'] for x in data]
        max_target_len = max([x.shape[1] for x in mel])

        # style embedding
        style_embedding = [x["style_embedding"] for x in data]

        padded_style_embedding = pad_sequence(style_embedding,
                                              batch_first=True,
                                              padding_value=0.0)

        # content embedding 
        content_embedding = [x["content_embedding"] for x in data]
        
        padded_content_embedding = pad_sequence(content_embedding,
                                              batch_first=True,
                                              padding_value=0.0)
        # pitch 
        pitch = [x["pitch"] for x in data]

        padded_pitch = pad_sequence(pitch,
                                    batch_first=True,
                                    padding_value=0.0)

        # energy 
        energy = [x["energy"] for x in data]

        padded_energy = pad_sequence(energy,
                                     batch_first=True,
                                     padding_value=0.0)
        
        # speaker
        speaker = torch.LongTensor([x['speaker'] for x in data])

        padded_mel = pad_mel(mel, self.config.downsample_ratio, max_target_len)
        
        mel_lens = torch.LongTensor([x.shape[1] for x in mel])  
        
        # wav
        wav = [x["wav"] for x in data]

        padded_wav = pad_sequence(wav,
                                  batch_first=True,
                                  padding_value=0.0)

        res = {
            "phoneme_id"        :   phoneme_id,
            "phoneme_lens"      :   phoneme_lens,
            "mel"               :   padded_mel,
            "mel_lens"          :   mel_lens,
            "style_embedding"   :   padded_style_embedding,
            "content_embedding" :   padded_content_embedding,
            "pitch"             :   padded_pitch,
            "energy"            :   padded_energy,
            "speaker"           :   speaker,
            "wav"               :   padded_wav,
        }
        return res



class Dataset_Prompt_Pretrain(torch.utils.data.Dataset):
    def __init__(self, data_path, config):
        
        self.datalist = self.load_files(data_path)
        self.config=config


        with open(config.emotion2id_path) as f:
            self.emotion2id = {t.strip():i for i, t in enumerate(f)}
        with open(config.pitch2id_path) as f:
            self.pitch2id = {t.strip():i for i, t in enumerate(f)}
        with open(config.energy2id_path) as f:
            self.energy2id = {t.strip():i for i, t in enumerate(f)}
        with open(config.speed2id_path) as f:
            self.speed2id = {t.strip():i for i, t in enumerate(f)}
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_path)


    def load_files(self, data_path):
        with jsonlines.open(data_path) as f:
            data = list(f)
        return data

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        prompt= self.datalist[index]["text"]
        uttid = self.datalist[index]["key"]

        emotion = self.emotion2id[self.datalist[index]["emotion"]]
        pitch = self.pitch2id[self.datalist[index]["pitch"]]
        energy = self.energy2id[self.datalist[index]["energy"]]
        speed = self.speed2id[self.datalist[index]["speed"]]

        return {
            "prompt":prompt,
            "uttid": uttid,
            "emotion": emotion,
            "pitch": pitch,
            "energy": energy,
            "speed": speed,
        }
        
    def TextMelCollate(self, data):

        prompt = self.tokenizer.batch_encode_plus([x["prompt"] for x in data], return_tensors="pt", padding=True) # return a dict that has "input_ids", "token_type_ids", "attention_mask"

        input_ids = prompt["input_ids"]
        token_type_ids = prompt["token_type_ids"]
        attention_mask = prompt["attention_mask"]

        # emotion
        emotion = torch.LongTensor([x['emotion'] for x in data])

        # pitch 
        pitch = torch.LongTensor([x['pitch'] for x in data])

        # energy 
        energy = torch.LongTensor([x['energy'] for x in data])

        # speed 
        speed = torch.LongTensor([x['speed'] for x in data])

        res = {
            "input_ids"       :   input_ids,
            "token_type_ids"  :   token_type_ids,
            "attention_mask"  :   attention_mask,
            "emotion"         :   emotion,
            "pitch"           :   pitch,
            "energy"          :   energy,
            "speed"           :   speed,
        }

        return res