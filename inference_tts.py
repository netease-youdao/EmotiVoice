# Copyright 2023, YOUDAO
#           2024, Du Jing(thuduj12@163.com)
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


import torch
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer
import os, sys, torch, argparse
import numpy as np
from models.hifigan.get_vocoder import MAX_WAV_VALUE
import soundfile as sf
from yacs import config as CONFIG
from tqdm import tqdm
from frontend import g2p_cn_en
from frontend_en import ROOT_DIR, read_lexicon, G2p


def get_style_embedding(prompt, tokenizer, style_encoder):
    prompt = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt["input_ids"]
    token_type_ids = prompt["token_type_ids"]
    attention_mask = prompt["attention_mask"]

    with torch.no_grad():
        output = style_encoder(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
    )
    style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    return style_embedding

def main(args, config, gpu_id, start_idx, chunk_num):
    device = torch.device(
        f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    root_path = os.path.join(config.output_directory, args.logdir)
    ckpt_path = os.path.join(root_path,  "ckpt")
    checkpoint_path = os.path.join(ckpt_path, args.checkpoint)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(root_path, 'audio')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)

    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    model_CKPT = torch.load(config.style_encoder_ckpt, map_location=device)
    model_ckpt = {}
    for key, value in model_CKPT['model'].items():
        new_key = key[7:]
        model_ckpt[new_key] = value
    style_encoder.load_state_dict(model_ckpt, strict=False)

    generator = JETSGenerator(conf).to(device)
    model_CKPT = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(model_CKPT['generator'])
    generator.eval()

    with open(config.token_list_path, 'r') as f:
        token2id = {t.strip():idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        id2speaker = {idx:t.strip() for idx, t in enumerate(f.readlines())}

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
    g2p = G2p()
    prompts = ['Happy', 'Excited', 'Sad', 'Angry']   # prompt is not efficient.
    speakers = [i for i in range(conf.n_speaker)]

    text_path = args.text_file
    with open(text_path, "r") as f:
        for i, line in enumerate(tqdm(f)):
            if not i in range(start_idx, start_idx+chunk_num):
                continue

            # iteration on prompts and speakers.
            prompt_idx = i % len(prompts)
            speaker_idx = i % len(speakers)
            prompt = prompts[prompt_idx]
            speaker = speakers[speaker_idx]
            speaker_name = id2speaker[speaker]
            speaker_path = os.path.join(output_dir, speaker_name)
            if not os.path.exists(speaker_path):
                os.makedirs(speaker_path, exist_ok=True)
            utt_name = f"{i+1:06d}"
            if os.path.exists(f"{speaker_path}/{utt_name}.wav"):
                print(f"audio {speaker_path}/{utt_name}.wav exists, continue.")
                continue

            try:
                content = line.strip()
                text = g2p_cn_en(content, g2p, lexicon)
                text = text.split()

                style_embedding = get_style_embedding(
                    prompt, tokenizer, style_encoder)
                content_embedding = get_style_embedding(
                    content, tokenizer, style_encoder)

                text_int = [token2id[ph] for ph in text]

                sequence = torch.from_numpy(
                    np.array(text_int)).to(device).long().unsqueeze(0)
                sequence_len = torch.from_numpy(
                    np.array([len(text_int)])).to(device)
                style_embedding = torch.from_numpy(
                    style_embedding).to(device).unsqueeze(0)
                content_embedding = torch.from_numpy(
                    content_embedding).to(device).unsqueeze(0)
                speaker = torch.from_numpy(
                    np.array([speaker])).to(device)
                with torch.no_grad():
                    infer_output = generator(
                            inputs_ling=sequence,
                            inputs_style_embedding=style_embedding,
                            input_lengths=sequence_len,
                            inputs_content_embedding=content_embedding,
                            inputs_speaker=speaker,
                            alpha=1.0
                        )
                    audio = infer_output[
                                "wav_predictions"].squeeze() * MAX_WAV_VALUE
                    audio = audio.cpu().numpy().astype('int16')

                    sf.write(file=f"{speaker_path}/{utt_name}.wav",
                             data=audio, samplerate=config.sampling_rate)
                    with open(f"{speaker_path}/{utt_name}.txt",
                              'w', encoding='utf-8') as ftext:
                        ftext.write(f"{content}\n")
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--logdir', default="prompt_tts_open_source_joint",
                   type=str, required=False)
    p.add_argument("-c", "--config_folder", default="config/joint",
                   type=str, required=False)
    p.add_argument("--checkpoint", type=str, default='g_00140000',
                   required=False, help='inference specific checkpoint。')
    p.add_argument('-t', '--text_file', type=str, required=True,
                   help='the absolute path of test file。')
    p.add_argument('-o', '--output_dir', type=str, required=False,
                   default=None, help='path to save the generated audios.')
    p.add_argument('-g', '--gpu_ids', type=str, required=False, default='0')
    p.add_argument('-n', '--num_thread', type=str, required=False, default='1')

    args = p.parse_args()
    sys.path.append(os.path.dirname(
        os.path.abspath("__file__")) + "/" + args.config_folder)

    from config import Config
    config = Config()

    from multiprocessing import Process
    gpus = args.gpu_ids
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpu_list = gpus.split(',')
    gpu_num = len(gpu_list)
    # 4GB GPU memory per thread, bottleneck is CPU usage!
    thread_per_gpu = int(args.num_thread)
    thread_num = gpu_num * thread_per_gpu  # threads
    torch.set_num_threads(4)  # faster

    total_len = 0
    with open(args.text_file) as fin:
        for line in fin:
            total_len += 1

    print(f"Total texts: {total_len}, Thread nums: {thread_num}")

    if total_len >= thread_num:
        chunk_size = int(total_len / thread_num)
        remains = total_len - chunk_size * thread_num
    else:
        chunk_size = 1
        remains = 0

    process_list = []
    chunk_begin = 0
    for i in range(thread_num):
        print(f"process part {i}...")
        gpu_id = i % gpu_num
        now_chunk_size = chunk_size
        if remains > 0:
            now_chunk_size = chunk_size + 1
            remains = remains - 1
        # use parallel processing or sequential processing
        p = Process(target=main, args=(
            args, config, gpu_id, chunk_begin, now_chunk_size))
        # main(args, config, gpu_id, chunk_begin, now_chunk_size)
        chunk_begin = chunk_begin + now_chunk_size
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()


