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

import os, json, torch
from models.hifigan.env import AttrDict
from models.hifigan.models import Generator

MAX_WAV_VALUE = 32768.0

def vocoder(hifi_gan_path, hifi_gan_name):
    device = torch.device('cpu')
    config_file = os.path.join(os.path.split(hifi_gan_path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)

    state_dict_g = torch.load(hifi_gan_path+hifi_gan_name, map_location=device)

    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def vocoder2(config,hifi_gan_ckpt_path):
    device = torch.device('cpu')
    global h
    generator = Generator(config.model).to(device)

    state_dict_g = torch.load(hifi_gan_ckpt_path, map_location=device)

    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator


def vocoder_inference(vocoder, melspec, max_db, min_db):
    with torch.no_grad():
        x = melspec*(max_db-min_db)+min_db
        device = torch.device('cpu')
        x = torch.FloatTensor(x).to(device)
        y_g_hat = vocoder(x)
        audio = y_g_hat.squeeze().numpy()
    return audio