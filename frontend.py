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
from pypinyin import pinyin, lazy_pinyin, Style
import jieba
import string

re_special_pinyin = re.compile(r'^(n|ng|m)$')
def split_py(py):
    tone = py[-1]
    py = py[:-1]
    sm = ""
    ym = ""
    suf_r = ""
    if re_special_pinyin.match(py):
        py = 'e' + py
    if py[-1] == 'r':
        suf_r = 'r'
        py = py[:-1]
    if py == 'zi' or py == 'ci' or py == 'si' or py == 'ri':
        sm = py[:1]
        ym = "ii"
    elif py == 'zhi' or py == 'chi' or py == 'shi':
        sm = py[:2]
        ym = "iii"
    elif py == 'ya' or py == 'yan' or py == 'yang' or py == 'yao' or py == 'ye' or py == 'yong' or py == 'you':
        sm = ""
        ym = 'i' + py[1:]
    elif py == 'yi' or py == 'yin' or py == 'ying':
        sm = ""
        ym = py[1:]
    elif py == 'yu' or py == 'yv' or py == 'yuan' or py == 'yvan' or py == 'yue ' or py == 'yve' or py == 'yun' or py == 'yvn':
        sm = ""
        ym = 'v' + py[2:]
    elif py == 'wu':
        sm = ""
        ym = "u"
    elif py[0] == 'w':
        sm = ""
        ym = "u" + py[1:]
    elif len(py) >= 2 and (py[0] == 'j' or py[0] == 'q' or py[0] == 'x') and py[1] == 'u':
        sm = py[0]
        ym = 'v' + py[2:]
    else:
        seg_pos = re.search('a|e|i|o|u|v', py)
        sm = py[:seg_pos.start()]
        ym = py[seg_pos.start():]
        if ym == 'ui':
            ym = 'uei'
        elif ym == 'iu':
            ym = 'iou'
        elif ym == 'un':
            ym = 'uen'
        elif ym == 'ue':
            ym = 've'
    ym += suf_r + tone
    return sm, ym


chinese_punctuation_pattern = r'[\u3002\uff0c\uff1f\uff01\uff1b\uff1a\u201c\u201d\u2018\u2019\u300a\u300b\u3008\u3009\u3010\u3011\u300e\u300f\u2014\u2026\u3001\uff08\uff09]'


def has_chinese_punctuation(text):
    match = re.search(chinese_punctuation_pattern, text)
    return match is not None
def has_english_punctuation(text):
    return text in string.punctuation
    
def number_to_chinese(char: str):
    chinese_digits = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    chinese_units = ['', '十', '百', '千', '万', '亿']

    result = ''
    char_str = str(char)
    length = len(char_str)

    if char_str.isdigit():
        if length == 1:
            return chinese_digits[int(char)]
        for i in range(length):
            digit = int(char_str[i])
            unit = length - i - 1

            if digit != 0:
                result += chinese_digits[digit] + chinese_units[unit]
            else:
                if unit == 0 or unit == 4 or unit == 8:
                    result += chinese_units[unit]
                elif result[-1] != '零' and result[-1] not in chinese_units:
                    result += chinese_digits[digit]
        return result
    else:
        return char
        
def g2p(text):
    res_text=["<sos/eos>"]
    seg_list = jieba.cut(text)
    for seg in seg_list:
        _seg = [number_to_chinese(_seg) for _seg in seg]
        py =[''.join(_py[0].split()) for _py in pinyin(_seg, style=Style.TONE3,neutral_tone_with_five=True)]

        if any([has_chinese_punctuation(_py) for _py in py])  or any([has_english_punctuation(_py) for _py in py]):
            res_text.pop()
            res_text.append("sp3")
        else:
            
            py = [" ".join(split_py(_py)) for _py in py]
            
            res_text.append(" sp0 ".join(py))
            res_text.append("sp1")
    res_text.pop()
    res_text.append("<sos/eos>")
    return " ".join(res_text)

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
            phoneme=g2p(line.rstrip())
            print(phoneme)
        fp.close()
