"""
This code is modified from https://github.com/wenet-e2e/wetts. 
"""

import os
import argparse
import soundfile as sf
import librosa
import jsonlines
from tqdm import tqdm
import re

def main(args):

    ROOT_DIR=os.path.abspath(args.data_dir)
    RAW_DIR=f"{ROOT_DIR}/raw"
    WAV_DIR=f"{ROOT_DIR}/wavs"
    TEXT_DIR=f"{ROOT_DIR}/text"

    os.makedirs(WAV_DIR, exist_ok=True)
    os.makedirs(TEXT_DIR, exist_ok=True)


    with open(f"{RAW_DIR}/BZNSYP/ProsodyLabeling/000001-010000.txt", encoding="utf-8") as f, \
        jsonlines.open(f"{TEXT_DIR}/data.jsonl", "w") as fout1:

        lines = f.readlines()
        for i in tqdm(range(0, len(lines), 2)):
            key = lines[i][:6]

            ### Text
            content_org = lines[i][7:].strip()
            content = re.sub("[。，、“”？：……！（ ）—；]", "", content_org)
            content_org = re.sub("#\d", "", content_org)

            chars = []
            prosody = {}
            j = 0
            while j < len(content):
                if content[j] == "#":
                    prosody[len(chars) - 1] = content[j : j + 2]
                    j += 2
                else:
                    chars.append(content[j])
                    j += 1
            
            if key == "005107":
                lines[i + 1] = lines[i + 1].replace(" ng1", " en1")
            if key == "002365":
                continue
            
            syllable = lines[i + 1].strip().split()
            s_index = 0
            phones = []
            phone = []
            for k, char in enumerate(chars):
                # 儿化音处理
                er_flag = False
                if char == "儿" and (s_index == len(syllable) or syllable[s_index][0:2] != "er"):
                    er_flag = True
                else:
                    phones.append(syllable[s_index])
                    #phones.extend(lexicon[syllable[s_index]])
                    s_index += 1
                

                if k in prosody:
                    if er_flag:
                        phones[-1] = prosody[k]
                    else:
                        phones.append(prosody[k])
                else:
                    phones.append("#0")
            
            ### Wav
            path = f"{RAW_DIR}/BZNSYP/Wave/{key}.wav"
            wav_path = f"{WAV_DIR}/{key}.wav"
            y, sr = sf.read(path)
            y_16=librosa.resample(y, orig_sr=sr, target_sr=16_000)
            sf.write(wav_path, y_16, 16_000)

            fout1.write({
                "key":key,
                "wav_path":wav_path,
                "speaker":"BZNSYP",
                "text":["<sos/eos>"] + phones[:-1] + ["<sos/eos>"],
                "original_text":content_org,
            })
            

    return 


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    args = p.parse_args()

    main(args)
