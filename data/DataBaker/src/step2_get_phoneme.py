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

from frontend_cn import split_py, tn_chinese
from frontend_en import read_lexicon, G2p
from frontend import  contains_chinese, re_digits, g2p_cn

# re_english_word = re.compile('([a-z\-\.\']+|\d+[\d\.]*)', re.I)
re_english_word = re.compile('([^\u4e00-\u9fa5]+|[ \u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09\u4e00-\u9fa5]+)', re.I)

def g2p_cn_en(text, g2p, lexicon):
    # Our policy dictates that if the text contains Chinese, digits are to be converted into Chinese.
    text=tn_chinese(text)
    parts = re_english_word.split(text)
    parts=list(filter(None, parts))
    tts_text = ["<sos/eos>"]
    chartype = ''
    text_contains_chinese = contains_chinese(text)
    for part in parts:
        if part == ' ' or part == '': continue
        if re_digits.match(part) and (text_contains_chinese or chartype == '') or contains_chinese(part):
            if chartype == 'en':
                tts_text.append('eng_cn_sp')
            phoneme = g2p_cn(part).split()[1:-1]
            chartype = 'cn'
        elif re_english_word.match(part):
            if chartype == 'cn':
                if "sp" in tts_text[-1]:
                    ""
                else:
                    tts_text.append('cn_eng_sp')
            phoneme = get_eng_phoneme(part, g2p, lexicon).split()
            if not phoneme :
                # tts_text.pop()
                continue
            else:
                chartype = 'en'
        else:
            continue
        tts_text.extend( phoneme )

    tts_text=" ".join(tts_text).split()
    if "sp" in tts_text[-1]:
        tts_text.pop()
    tts_text.append("<sos/eos>")

    return " ".join(tts_text)

def get_eng_phoneme(text, g2p, lexicon):
    """
    english g2p
    """
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

        
    return " ".join(phones)


def onetime(resource, sample):

    text=sample["text"]
    # del sample["original_text"]

    phoneme = get_phoneme(text, resource["g2p"]).split()

    sample["text"]=phoneme
    # sample["original_text"]=text
    sample["prompt"]=sample["original_text"]
    
    return sample

def onetime2(resource, sample):

    text=sample["original_text"]
    del sample["original_text"]
    try:
        phoneme = g2p_cn_en(text, resource["g2p_en"], resource["lexicon"]).split()#g2p_cn_eng_mix(text, resource["g2p_en"], resource["lexicon"]).split()
    except:
        print("Warning!!! phoneme get error! " + \
        "Please check text")
        print("Text is: ", text)
        return ""
    
    if not phoneme:
        return ""

    sample["text"]=phoneme
    sample["original_text"]=text
    sample["prompt"]=sample["original_text"]
    
    return sample

def get_phoneme(text, g2p):
    special_tokens = {"#0":"sp0", "#1":"sp1", "#2":"sp2", "#3":"sp3", "#4":"sp4", "<sos/eos>":"<sos/eos>"}
    phones = []

    for ph in text:
        if ph not in special_tokens:
            phs = g2p(ph)
            phones.extend([ph for ph in phs if ph])
        else:
            phones.append(special_tokens[ph])

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
        "g2p":split_py,
        "g2p_en":g2p,
        "lexicon":lexicon,
    }

    with jsonlines.open(f"{TEXT_DIR}/data.jsonl") as f:
        data = list(f)

    new_data=[]
    with jsonlines.open(f"{TEXT_DIR}/datalist.jsonl", "w") as f:
        for sample in tqdm(data):
            if not args.generate_phoneme:
                sample = onetime(resource, sample)
            else:
                sample = onetime2(resource, sample)
            if not sample:
                continue
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
    p.add_argument('--generate_phoneme', type=bool, default=False)
    args = p.parse_args()

    main(args)