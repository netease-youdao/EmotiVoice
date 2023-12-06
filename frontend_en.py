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

import re
import argparse
from string import punctuation
import numpy as np

from g2p_en import G2p

import os


ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

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

    # mark = "." if text[-1] != "?" else "?"
    # phones = ["<sos/eos>"] + phones + [mark, "<sos/eos>"]
    return " ".join(phones)
    

if __name__ == "__main__":
    lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
    g2p = G2p()
    phonemes= get_eng_phoneme("Happy New Year", g2p, lexicon)
    import sys
    from os.path import isfile
    if len(sys.argv) < 2:
        print("Usage: python %s <text>" % sys.argv[0])
        exit()
    text_file = sys.argv[1]
    if isfile(text_file):
        fp = open(text_file, 'r')
        for line in fp:
            phoneme=get_eng_phoneme(line.rstrip(), g2p, lexicon)
            print(phoneme)
        fp.close()