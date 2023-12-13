
import argparse
import collections
import pathlib
import os
from typing import Iterable
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir',
                        type=str,
                        help='Path to cath dataset')
    parser.add_argument('--wav', type=str, help='Path to export paths of wavs.')
    parser.add_argument('--speaker', type=str, help='Path to export speakers.')
    parser.add_argument('--text', type=str, help='Path to export text of wavs.')
    return parser.parse_args()


def save_scp_files(wav_scp_path: os.PathLike, speaker_scp_path: os.PathLike,
                   text_scp_path: os.PathLike, content: Iterable[str]):
    wav_scp_path = pathlib.Path(wav_scp_path)
    speaker_scp_path = pathlib.Path(speaker_scp_path)
    text_scp_path = pathlib.Path(text_scp_path)

    wav_scp_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_scp_path.parent.mkdir(parents=True, exist_ok=True)
    text_scp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(wav_scp_path, 'w') as wav_scp_file:
        wav_scp_file.writelines([str(line[0]) + '\n' for line in content])
    with open(speaker_scp_path, 'w') as speaker_scp_file:
        speaker_scp_file.writelines([line[1] + '\n' for line in content])
    with open(text_scp_path, 'w') as text_scp_file:
        text_scp_file.writelines([line[2] + '\n' for line in content])


def main(args):
    dataset_dir = pathlib.Path(args.dataset_dir)

    with open(dataset_dir /
              'text_sp1-sp4') as train_set_label_file:
        train_set_label = [
            x.strip() for x in train_set_label_file.readlines()
        ]
    train_set_path={}
    with open(dataset_dir /
              'wav.scp') as train_set_path_file:
        for line in train_set_path_file:
            line = line.strip().split()
            train_set_path[line[0]] = line[1]

    samples = collections.defaultdict(list)

    for line in tqdm(train_set_label):
        line = line.split()
        # sample_name = "_".join(line[0].split("_")[1:])
        sample_name = line[0].split("|")[1]
        tokens = " ".join(line[1:])
        speaker = line[0].split("|")[0]
        wav_path = train_set_path[sample_name]
        if os.path.exists(wav_path):
            samples[speaker].append((wav_path, speaker, tokens))
        else:
            print(wav_path, "is not existed")

    sample_list = []

    for speaker in sorted(samples):
        sample_list.extend(samples[speaker])

    save_scp_files(args.wav, args.speaker, args.text, sample_list)


if __name__ == "__main__":
    main(get_args())
