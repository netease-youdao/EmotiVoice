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
import os
import shutil
import argparse


def main(args):
    from os.path import join
    data_dir = args.data_dir
    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    info_dir = join(exp_dir, 'info')
    prepare_info(data_dir, info_dir)

    config_dir = join(exp_dir, 'config')
    prepare_config(data_dir, info_dir, exp_dir, config_dir)

    ckpt_dir = join(exp_dir, 'ckpt')
    prepare_ckpt(data_dir, info_dir, ckpt_dir)


ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))
def prepare_info(data_dir, info_dir):
    import jsonlines
    print('prepare_info: %s' %info_dir)
    os.makedirs(info_dir, exist_ok=True)

    for name in ["emotion", "energy", "pitch", "speed", "tokenlist"]:
        shutil.copy(f"{ROOT_DIR}/data/youdao/text/{name}", f"{info_dir}/{name}")

    d_speaker = {} # get all the speakers from datalist
    with jsonlines.open(f"{data_dir}/train/datalist.jsonl") as reader:
        for obj in reader:
            speaker = obj["speaker"]
            if not speaker in d_speaker:
                d_speaker[speaker] = 1
            else:
                d_speaker[speaker] += 1

    with open(f"{ROOT_DIR}/data/youdao/text/speaker2") as f, \
        open(f"{info_dir}/speaker", "w") as fout:

        for line in f:
            speaker = line.strip()
            if speaker in d_speaker:
                print('warning: duplicate of speaker [%s] in [%s]' % (speaker, data_dir))
                continue
            fout.write(line.strip()+"\n")

        for speaker in sorted(d_speaker.keys()):
            fout.write(speaker + "\n")


def prepare_config(data_dir, info_dir, exp_dir, config_dir):
    print('prepare_config: %s' %config_dir)
    os.makedirs(config_dir, exist_ok=True)

    with open(f"{ROOT_DIR}/config/template.py") as f, \
        open(f"{config_dir}/config.py", "w") as fout:

        for line in f:
            fout.write(line.replace('<DATA_DIR>', data_dir).replace('<INFO_DIR>', info_dir).replace('<EXP_DIR>', exp_dir))


def prepare_ckpt(data_dir, info_dir, ckpt_dir):
    print('prepare_ckpt: %s' %ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    with open(f"{info_dir}/speaker") as f:
        speaker_list=[line.strip() for line in f]
    assert len(speaker_list) >= 2014
    
    gen_ckpt_path = f"{ROOT_DIR}/outputs/prompt_tts_open_source_joint/ckpt/g_00140000"
    disc_ckpt_path = f"{ROOT_DIR}/outputs/prompt_tts_open_source_joint/ckpt/do_00140000"

    gen_ckpt = torch.load(gen_ckpt_path, map_location="cpu")

    speaker_embeddings = gen_ckpt["generator"]["am.spk_tokenizer.weight"].clone()
    
    new_embedding = torch.randn((len(speaker_list)-speaker_embeddings.size(0), speaker_embeddings.size(1)))

    gen_ckpt["generator"]["am.spk_tokenizer.weight"] = torch.cat([speaker_embeddings, new_embedding], dim=0)


    torch.save(gen_ckpt, f"{ckpt_dir}/pretrained_generator")
    shutil.copy(disc_ckpt_path, f"{ckpt_dir}/pretrained_discriminator")



if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--exp_dir', type=str, required=True)
    args = p.parse_args()

    main(args)
