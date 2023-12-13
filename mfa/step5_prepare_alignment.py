#!/usr/bin/env python
# Copyright 2022 Binbin Zhang(binbzha@qq.com), Jie Chen(unrea1sama@outlook.com)
"""Generate lab files from data list for alignment
"""

import argparse
import pathlib
import random, os
from tqdm import tqdm
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, help='Path to wav.txt.')
    parser.add_argument("--speaker", type=str, help='Path to speaker.txt.')
    parser.add_argument(
        "--text",
        type=str,
        help=('Path to text.txt. ',
              'It should only contain phonemes and special tokens.'))
    parser.add_argument('--special_tokens',
                        type=str,
                        help='Path to special_token.txt.')
    parser.add_argument(
        '--pronounciation_dict',
        type=str,
        help='Path to export pronounciation dictionary for MFA.')
    parser.add_argument('--output_dir',
                        type=str,
                        help='Path to directory for exporting .lab files.')
    return parser.parse_args()


def main(args):
    output_dir = pathlib.Path(args.output_dir)
    pronounciation_dict = set()
    with open(args.special_tokens) as fin:
        special_tokens = set([x.strip() for x in fin.readlines()])

    num_speaker = 1
    with open(args.wav) as f:
        index = [i for i in range(len(f.readlines()))]
    _mfa_groups = [index[i::num_speaker] for i in range(num_speaker)]
    mfa_groups = []
    for i, group in enumerate(_mfa_groups):
        mfa_groups.extend([i for _ in range(len(group))])
    
    random.shuffle(mfa_groups)
    os.system(f"rm -rf {args.output_dir}/*")
    with open(args.wav) as fwav, open(args.speaker) as fspeaker, open(
            args.text) as ftext:
        for wav_path, speaker, text, i in tqdm(zip(fwav, fspeaker, ftext, mfa_groups)):
            i = speaker.strip()#str(i)
            wav_path, speaker, text = (pathlib.Path(wav_path.strip()),
                                       speaker.strip(), text.strip().split())
            lab_dir = output_dir / i
            lab_dir.mkdir(parents=True, exist_ok=True)

            name=wav_path.stem.strip()

            lab_file = output_dir / i / f'{i}_{name}.lab'
            wav_file = output_dir / i / f'{i}_{name}.wav'
            try:
                os.symlink(wav_path, wav_file)
            except:
                print("ERROR PATH",wav_path)
                continue

            
            with lab_file.open('w') as fout:
                text_no_special_tokens = [ph for ph in text if ph not in special_tokens]
                pronounciation_dict |= set(text_no_special_tokens)
                fout.writelines([' '.join(text_no_special_tokens)])
    with open(args.pronounciation_dict, 'w') as fout:
        fout.writelines([
            '{} {}\n'.format(symbol, " ".join(symbol.split("_"))) for symbol in pronounciation_dict
        ])


if __name__ == '__main__':
    main(get_args())
