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

import torch.nn as nn
import torch
from models.hifigan.models import MultiScaleDiscriminator, MultiPeriodDiscriminator



class Discriminator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()
        if config.pretrained_discriminator:
            state_dict_do = torch.load(config.pretrained_discriminator,map_location="cpu")
        
            self.mpd.load_state_dict(state_dict_do['mpd'])
            self.msd.load_state_dict(state_dict_do['msd'])
            print("pretrained discriminator is loaded")
    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)

        return y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g