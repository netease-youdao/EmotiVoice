# Copyright (c) 2022 Tsinghua University. (authors: Jie Chen)
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
"""Convert full label pingyin sequences into phoneme sequences according to
lexicon.
"""

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, help='Path to text.txt.')
    parser.add_argument('--special_tokens',
                        type=str,
                        help='Path to special_token.txt')
    parser.add_argument('--output', type=str, help='Path to output file.')
    return parser.parse_args()


def main(args):
    with open(args.special_tokens) as fin:
        special_tokens = set([x.strip() for x in fin.readlines()])
    samples = []
    with open(args.text) as fin:
        for line in fin:
            tokens = []
            word = []
            for ph in line.strip().split():
                if ph in special_tokens:
                    word = "_".join(word)

                    tokens.append(word)
                    tokens.append(ph)
                    word = []
                else:
                    ph = ph #[A] -> A
                    word.append(ph)
                    
            samples.append(' '.join(tokens))
    with open(args.output, 'w') as fout:
        fout.writelines([x + '\n' for x in samples])


if __name__ == '__main__':
    main(get_args())
