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

import argparse
import os
import jsonlines
import json
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from functools import partial
import re
import sys
DIR=os.path.dirname(os.path.abspath("__file__"))
sys.path.append(DIR)

from frontend_en import read_lexicon, G2p


def onetime(resource, sample):

    text=sample["original_text"]
    del sample["original_text"]

    phoneme = get_phoneme(text, resource["g2p"], resource["lexicon"]).split()

    sample["text"]=phoneme
    sample["original_text"]=text
    sample["prompt"]=text
    
    return sample

def get_phoneme(text, g2p, lexicon):
    filters = {",", " ", "'"}
    phones = []
    words = list(filter(lambda x: x not in {"", " "}, re.split(r"([,;.\-\?\!\s+])", text)))

    for w in words:
        if w.lower() in lexicon:
            
            for ph in lexicon[w.lower()]:
                if ph not in filters:
                    phones += ["[" + ph + "]"]

            if "sp" not in phones[-1]:
                phones += ["engsp1"]
        else:
            phone=g2p(w)
            if not phone:
                continue

            if phone[0].isalnum():
                
                for ph in phone:
                    if ph not in filters:
                        phones += ["[" + ph + "]"]
                    if ph == " " and "sp" not in phones[-1]:
                        phones += ["engsp1"]
            elif phone == " ":
                continue
            elif phones:
                phones.pop() # pop engsp1
                phones.append("engsp4")
    if phones and "engsp" in phones[-1]:
        phones.pop()
        
    mark = "." if text[-1] != "?" else "?"
    phones = ["<sos/eos>"] + phones + [mark, "<sos/eos>"]
    return " ".join(phones)



def main(args):

    ROOT_DIR=args.data_dir
    TRAIN_DIR=f"{ROOT_DIR}/train"
    VALID_DIR=f"{ROOT_DIR}/valid"
    TEXT_DIR=f"{ROOT_DIR}/text"

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALID_DIR, exist_ok=True)

    lexicon = read_lexicon(f"{DIR}/lexicon/librispeech-lexicon.txt")

    g2p = G2p()

    resource={
        "g2p":g2p,
        "lexicon":lexicon,
    }

    with jsonlines.open(f"{TEXT_DIR}/data.jsonl") as f:
        data = list(f)

    new_data=[]
    with jsonlines.open(f"{TEXT_DIR}/datalist.jsonl", "w") as f:
        for sample in tqdm(data):
            sample = onetime(resource, sample)
            f.write(sample)
            new_data.append(sample)
    
    with jsonlines.open(f"{TRAIN_DIR}/datalist.jsonl", "w") as f:
        for sample in tqdm(new_data[:-3]):
            f.write(sample)
            
    with jsonlines.open(f"{VALID_DIR}/datalist.jsonl", "w") as f:
        for sample in tqdm(data[-3:]):
            f.write(sample)


    return 

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    args = p.parse_args()

    main(args)