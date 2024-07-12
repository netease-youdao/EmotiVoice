"""
This is a simple desktop created using tkitner.

Don't use "ctrl+c" to stop program.

usage:
```
pip install -r requirements.txt
pip install pygame
python gui.py
```
"""
import glob
import io
import os
import threading
import multiprocessing as mp
import time
import tkinter as tk
import wave
from tkinter import ttk, filedialog
from tkinter.scrolledtext import ScrolledText

import pygame


class MainWindow(ttk.Frame):
    status = 'loading'

    def __init__(self, master=None, **kw) -> None:
        self.tx = kw.pop('tx')
        self.rx = kw.pop('rx')

        super().__init__(master, **kw)
        self.root = master

        self.status_value = tk.StringVar(self, "Loading...")
        self.engine_value = tk.StringVar(self, "")
        self.speaker_value = tk.StringVar(self, "")
        self.prompt_value = tk.StringVar(self, "")
        self.language_value = tk.StringVar(self, "zh_us")

        self.images = [
            tk.PhotoImage(name="play", data="iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAKVJREFUOE9jZKAQMFKonwHFAGUJ0f0MDAwKfxmYHR+8ePGAGMPRDbjPwMDQyMDAEP+f4f+Bey/egNh4AYYBd1+8VlSQkFBgYvgTz8jAmMDI8D/xzos3B3CZgtUAmGKQQcwMf+fjcw1eA0AGEXINQQMIuYY+BihJiNTjClBiAnH/f4b/C3BFKfWjERTv/xkYQbY2kJOQwEmZUOJBTlTUzUyE0j02eQCGTHkRGGbdugAAAABJRU5ErkJggg=="),
            tk.PhotoImage(name="stop", data="iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAGtJREFUOE9jZKAQMML0K0hIKBBr1oMXLx7A1MINUJYQ/c/AwHCASEMU7r54rQhSCzYAZDszw9/5d1+8diTGAGUJ0f1/GZgTQS4ZNWA0DKieDu6TkJQd/jIwK8JdAEvOxCRjkBqsmYlYzejqAImGgxFqxQRsAAAAAElFTkSuQmCC"),
            tk.PhotoImage(name="pause", data="iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAJ1JREFUOE/tk8ENwjAMRb+VFRrJNwJhkHQTmAQxSdmkDEKg3CylK0QBo/ZQxAFRcauP/9v/YL1PmDn0fu+YnWqdSDd6n7TRmwRsuQoF1AI4A3BR0loXPdsyaIFQ6ov06r9mEuDZ7grK6ir90bNtM8xelwxyEyXVG64OBLpHSaclYPnB/zgYUG60Ck90Q5RE2gODfPsKZcVzVpl+afYD0SjfEWEtd3IAAAAASUVORK5CYII="),
            tk.PhotoImage(name="save", data="iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAK9JREFUOE/tkkEKwjAQRX+oRzAwOyPxIL2J7UlKT9J6Ek8SzTIQr1AjEwxUSEuk4MrZDnkz+fMENpbg9yfa1wGiA1ADsCtMlfoBYby5Rx8BmuTdOH/UJJuAcOBGDpL6T+wuFabOON8WAXIDNMnrhOpHAEWkrHN2/sXsBu8wh5UgOeTWOD9mARwcT1u6Am+ReouAUi3+gGjvh0gDa1ka4Fz/pHID4MxX/ALSsw8RsKVeNHClccQk6AYAAAAASUVORK5CYII="),
        ]

        state_bar = ttk.Frame(self)
        state_bar.pack(side="bottom", fill="x")
        ttk.Label(state_bar, textvariable=self.status_value).pack(side="left")
        ttk.Sizegrip(state_bar).pack(side="right", anchor="se")
        ttk.Label(state_bar, textvariable=self.engine_value).pack(side="right", padx=15)

        pwin = ttk.PanedWindow(self, orient="horizontal")
        pwin.pack(fill="both", expand=True)

        left_frame = ttk.Frame(pwin)
        left_label_frame = ttk.LabelFrame(left_frame, text="Text to be synthesized into speech (合成文本)")
        self.text = ScrolledText(left_label_frame)
        self.text.pack(fill="both", expand=True)
        left_label_frame.pack(fill="both", expand=True, padx=5, pady=5)
        pwin.add(left_frame, weight=1)

        right_frame = ttk.Frame(pwin)
        frame = ttk.Frame(right_frame)

        ttk.Label(frame, text="Speaker ID (说话人)").grid(row=0, column=0)
        self.combo = ttk.Combobox(frame, textvariable=self.speaker_value, state="readonly")
        self.combo.grid(row=0, column=1)

        ttk.Label(frame, text="Prompt (开心/悲伤)").grid(row=1, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.prompt_value).grid(row=1, column=1, sticky="w", pady=2)

        ttk.Label(frame, text="Language (语言)").grid(row=2, column=0, sticky="e")
        ttk.Combobox(frame, values=("zh_us", ), textvariable=self.language_value, state="readonly").grid(row=2, column=1, sticky="w", pady=2)

        self.syn_btn = ttk.Button(frame, text="Synthesize (合成)")
        self.syn_btn.grid(row=3, column=1)

        frame.pack(padx=5, pady=5)
        player_frame = ttk.Frame(right_frame)
        self.play_btn = ttk.Button(player_frame, image='play', command=self.on_play)
        self.play_btn.pack(side="left")
        # ttk.Button(player_frame, image="pause", command=self.on_pause).pack(side="left")
        ttk.Button(player_frame, image="stop", command=self.on_stop).pack(side="left")
        self.progress = ttk.Progressbar(player_frame, orient="horizontal", mode="determinate")
        self.progress.pack(side="left")
        ttk.Button(player_frame, image="save", command=self.on_save).pack(side="left")
        player_frame.pack(padx=5, pady=5)
        pwin.add(right_frame, weight=1)

        # init pygame mixer
        pygame.init()
        pygame.mixer.init()

        # 定义变量
        self.audio_file = None
        self.audio_length = 0
        self.paused = False
        self.paused_time = 0
        self.progress_updater = None

        t = threading.Thread(target=self.on_load)
        t.setDaemon(False)
        t.start()

    def on_load(self):
        while self.status != 'close':
            data = self.rx.recv()
            if data['tag'] == 'engine':
                self.engine_value.set(data['msg'])
            elif data['tag'] == 'speakers':
                self.combo.config(values=data['msg'])
                self.combo.current(0)
            elif data['tag'] == 'status':
                self.status_value.set(data['msg'])
            elif data['tag'] == "subbtn":
                self.status = 'ready'
                self.syn_btn.config(command=self.on_click)
            elif data['tag'] == 'audio':
                self.status_value.set("Completed")
                self.audio_file = data['audio']
                t = threading.Thread(target=self.init_player)
                t.start()
            elif data['tag'] == 'close':
                self.status = 'close'
                break

    def on_click(self):
        self.syn_btn.config(command=None)
        self.status = 'working'
        self.audio_file = None
        self.audio_length = 0
        self.paused = False
        self.paused_time = 0
        self.progress_updater = None
        self.status_value.set("Start")

        speaker = self.speaker_value.get()
        prompt = self.prompt_value.get()
        content = self.text.get("1.0", "end")
        lang = self.language_value.get()

        self.tx.send({
            'tag': 'synthesize',
            'speaker': speaker,
            'prompt': prompt,
            'content': content,
            'lang': lang
        })

    def init_player(self):
        # 获取音乐长度（秒）
        with wave.open(self.audio_file, 'rb') as wf:
            self.audio_length = wf.getnframes() / wf.getframerate()
        self.progress.config(maximum=self.audio_length)
        self.audio_file.seek(0)
        pygame.mixer.music.load(self.audio_file)

    def on_play(self):
        if self.audio_file:
            if self.paused:
                pygame.mixer.music.unpause()
                self.paused = False
                # 继续更新进度条
                self.update_progress()
            else:
                pygame.mixer.music.play()
                # 开始更新进度条
                self.update_progress()

            self.play_btn.config(image="pause", command=self.on_pause)

    def on_pause(self):
        if pygame.mixer.music.get_busy() and not self.paused:
            pygame.mixer.music.pause()
            self.paused = True
            # 记录当前暂停的时间
            self.paused_time = pygame.mixer.music.get_pos() / 1000
            self.play_btn.config(image='play', command=self.on_play)

    def on_stop(self):
        pygame.mixer.music.stop()
        self.paused = False
        self.paused_time = 0
        self.progress.stop()
        self.play_btn.config(image='play', command=self.on_play)

    def on_save(self):
        self.status_value.set("Saving")
        if self.audio_file:
            filename = filedialog.asksaveasfilename(defaultextension='.wav')
            if filename:
                file = open(filename, 'wb')
                self.audio_file.seek(0)
                file.write(self.audio_file.read())
                file.close()
                self.status_value.set("Saved")
            else:
                self.status_value.set("Save failed")
        else:
            self.status_value.set("No audio files")

    def update_progress(self):
        current_time = pygame.mixer.music.get_pos() / 1000
        self.progress["value"] = current_time

        if pygame.mixer.music.get_busy() and not self.paused:
            self.progress_updater = self.root.after(100, self.update_progress)
        else:
            # 清除进度条更新
            if self.progress_updater:
                self.root.after_cancel(self.progress_updater)
                self.play_btn.config(image='play', command=self.on_play)


def process_service(tx, rx):
    tx.send({"tag": 'status', 'msg': 'Loading...'})
    # Because the load is slow, some of the import operations are also carried out in the child process
    import numpy as np
    from yacs import config as CONFIG
    import torch

    from frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
    from config.joint.config import Config
    from models.prompt_tts_modified.jets import JETSGenerator
    from models.prompt_tts_modified.simbert import StyleEncoder
    from transformers import AutoTokenizer

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tx.send({"tag": 'engine', 'msg': DEVICE})
    MAX_WAV_VALUE = 32768.0

    config = Config()

    def scan_checkpoint(cp_dir, prefix, c=8):
        pattern = os.path.join(cp_dir, prefix + '?'*c)
        cp_list = glob.glob(pattern)
        if len(cp_list) == 0:
            return None
        return sorted(cp_list)[-1]

    def get_models():
        am_checkpoint_path = scan_checkpoint(f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')
        style_encoder_checkpoint_path = scan_checkpoint(f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)  # f'{config.output_directory}/style_encoder/ckpt/checkpoint_163431'

        with open(config.model_config_path, 'r') as fin:
            conf = CONFIG.load_cfg(fin)

        conf.n_vocab = config.n_symbols
        conf.n_speaker = config.speaker_n_labels

        style_encoder = StyleEncoder(config)
        model_CKPT = torch.load(style_encoder_checkpoint_path, map_location="cpu")
        model_ckpt = {}
        for key, value in model_CKPT['model'].items():
            new_key = key[7:]
            model_ckpt[new_key] = value
        style_encoder.load_state_dict(model_ckpt, strict=False)
        generator = JETSGenerator(conf).to(DEVICE)

        model_CKPT = torch.load(am_checkpoint_path, map_location=DEVICE)
        generator.load_state_dict(model_CKPT['generator'])
        generator.eval()

        tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

        with open(config.token_list_path, 'r') as f:
            token2id = {t.strip(): idx for idx, t, in enumerate(f.readlines())}

        with open(config.speaker2id_path, encoding='utf-8') as f:
            speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

        return (style_encoder, generator, tokenizer, token2id, speaker2id)

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

    def tts(name, text, prompt, content, speaker, models):
        (style_encoder, generator, tokenizer, token2id, speaker2id)=models

        style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
        content_embedding = get_style_embedding(content, tokenizer, style_encoder)
        speaker = speaker2id[speaker]
        text_int = [token2id[ph] for ph in text.split()]

        sequence = torch.from_numpy(np.array(text_int)).to(DEVICE).long().unsqueeze(0)
        sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
        style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
        content_embedding = torch.from_numpy(content_embedding).to(DEVICE).unsqueeze(0)
        speaker = torch.from_numpy(np.array([speaker])).to(DEVICE)

        with torch.no_grad():
            infer_output = generator(
                    inputs_ling=sequence,
                    inputs_style_embedding=style_embedding,
                    input_lengths=sequence_len,
                    inputs_content_embedding=content_embedding,
                    inputs_speaker=speaker,
                    alpha=1.0
                )

        audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')
        return audio

    def tts_thread(data):
        try:
            tx.send({"tag": "status", "msg": "Trans phoneme"})
            text = g2p_cn_en(data['content'], g2p, lexicon)
            tx.send({"tag": "status", "msg": "Synthesizing"})
            wav_data = tts(0, text, data['prompt'], data['content'], data['speaker'], models)
            buffer = io.BytesIO()
            wf = wave.open(buffer, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.sampling_rate)
            wf.writeframes(wav_data.tobytes())
            wf.close()
            buffer.seek(0)
            tx.send({'tag': 'audio', 'audio': buffer})
            tx.send({"tag": 'subbtn'})
        except Exception as e:
            tx.send({'tag': 'status', 'msg': e})
            tx.send({"tag": 'subbtn'})

    tx.send({"tag": 'speakers', 'msg': config.speakers})
    try:
        tx.send({"tag": 'status', 'msg': 'Getting models'})
        models = get_models()
        tx.send({"tag": "status", "msg": "Get models success"})
    except Exception as e:
        tx.send({"tag": "status", "msg": e})
        return
    tx.send({"tag": 'status', 'msg': 'Reading lexicon'})
    lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
    tx.send({"tag": 'status', 'msg': 'Setting G2p'})
    g2p = G2p()
    tx.send({"tag": 'status', 'msg': 'Ready'})
    tx.send({"tag": 'subbtn'})

    while True:
        data = rx.recv()
        if data['tag'] == 'synthesize':
            thread = threading.Thread(target=tts_thread, args=(data, ))
            thread.start()
        elif data['tag'] == 'close':
            tx.send({'tag': 'close'})
            break


def process_window(tx, rx):
    def on_closing():
        tx.send({'tag': 'close'})
        mainframe.status = 'close'
        root.destroy()

    root = tk.Tk()
    root.title("EmotiVoice")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    mainframe = MainWindow(root, tx=tx, rx=rx)
    mainframe.pack(fill="both", expand=True)
    root.mainloop()


def main():
    a_tx, a_rx = mp.Pipe()
    b_tx, b_rx = mp.Pipe()

    p1 = mp.Process(target=process_window, args=(a_tx, b_rx))
    p2 = mp.Process(target=process_service, args=(b_tx, a_rx))

    p1.start()
    p2.start()

    while True:
        if not p1.is_alive():
            p2.terminate()
            p2.join()
            break
        time.sleep(1)


if __name__ == "__main__":
    main()
