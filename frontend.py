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
from frontend_cn import g2p_cn, re_digits
from frontend_en import preprocess_english

# Thanks to GuGCoCo and PatroxGaurab for identifying the issue: 
# the results differ between frontend.py and frontend_en.py. Here's a quick fix.
re_english_word = re.compile('([a-z\-\.\'\s,;\:\!\?]+|\d+[\d\.]*)', re.I)
def g2p_cn_en(text):
    # Our policy dictates that if the text contains Chinese, digits are to be converted into Chinese.
    parts = re_english_word.split(text)
    tts_text = ["<sos/eos>"]
    chartype = ''
    text_contains_chinese = contains_chinese(text)
    for part in parts:
        if part == ' ' or part == '': continue
        if re_digits.match(part) and (text_contains_chinese or chartype == '') or contains_chinese(part):
            if chartype == 'en':
                tts_text.append('eng_cn_sp')
            phoneme = g2p_cn(part)
            chartype = 'cn'
        elif re_english_word.match(part):
            if chartype == 'cn':
                tts_text.append('cn_eng_sp')
            phoneme = preprocess_english(part).replace(".", "")
            chartype = 'en'
        else:
            continue
        tts_text.append( phoneme.replace("[ ]", "").replace("<sos/eos>", "") )
    tts_text.append("<sos/eos>")
    return " ".join(tts_text)

def contains_chinese(text):
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    match = re.search(pattern, text)
    return match is not None


if __name__ == "__main__":
    import sys
    from os.path import isfile
    if len(sys.argv) < 2:
        print("Usage: python %s <text>" % sys.argv[0])
        exit()
    text_file = sys.argv[1]
    if isfile(text_file):
        fp = open(text_file, 'r')
        for line in fp:
            phoneme = g2p_cn_en(line.rstrip())
            print(phoneme)
        fp.close()
