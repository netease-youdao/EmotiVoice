# Copyright (c) 2022 Tsinghua University(Jie Chen)
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
import jsonlines
import pathlib

def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', type=str, help='Path to wav.txt.')
    parser.add_argument('--speaker', type=str, help='Path to speaker.txt.')
    parser.add_argument('--text', type=str, help='Path to text.txt.')
    parser.add_argument('--duration', type=str, help='Path to duration.txt.')
    parser.add_argument('--datalist_path',
                        type=str,
                        help='Path to export datalist.jsonl.')
    args = parser.parse_args()
    return args


def main(args):
    wavs = read_lists(args.wav)
    speakers = read_lists(args.speaker)
    texts = read_lists(args.text)
    durations = read_lists(args.duration)
    with jsonlines.open(args.datalist_path, 'w') as fdatalist:
        for wav, speaker, text, duration in zip(wavs, speakers, texts,
                                                durations):
            key = pathlib.Path(wav).stem
            fdatalist.write({
                'key': key,
                'wav_path': wav,
                'speaker': speaker,
                'text': text.split(),
                'duration': [float(x) for x in duration.split()]
            })


if __name__ == '__main__':
    main(get_args())
