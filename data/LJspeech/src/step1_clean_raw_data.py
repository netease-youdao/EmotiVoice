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

import os
import argparse
import soundfile as sf
import librosa
import jsonlines
from tqdm import tqdm

def main(args):

    ROOT_DIR=os.path.abspath(args.data_dir)
    RAW_DIR=f"{ROOT_DIR}/raw"
    WAV_DIR=f"{ROOT_DIR}/wavs"
    TEXT_DIR=f"{ROOT_DIR}/text"

    os.makedirs(WAV_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)

    with open(f"{RAW_DIR}/LJSpeech-1.1/metadata.csv") as f, \
        jsonlines.open(f"{TEXT_DIR}/data.jsonl", "w") as fout1:
        # open(f"{TEXT_DIR}/text_raw", "w") as fout2:
        for line in tqdm(f):
            #### Text ####
            line = line.strip().split("|")
            name = line[0]
            text=line[1]
            
            #### Wav #####
            path = f"{RAW_DIR}/LJSpeech-1.1/wavs/{name}.wav"
            wav_path = f"{WAV_DIR}/{name}.wav"
            y, sr = sf.read(path)
            y_16=librosa.resample(y, orig_sr=sr, target_sr=16_000)
            sf.write(wav_path, y_16, 16_000)

            #### Write ####
            fout1.write({
                "key":name,
                "wav_path":wav_path,
                "speaker":"LJ",
                "original_text":text
            })
            # fout2.write(text+"\n")




    return 


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    args = p.parse_args()

    main(args)
