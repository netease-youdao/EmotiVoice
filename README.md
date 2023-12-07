<font size=4> README: EN | <a href="./README.zh.md">中文</a>  </font>


<div align="center">
    <h1>EmotiVoice 😊: a Multi-Voice and Prompt-Controlled TTS Engine</h1>
</div>

<div align="center">
    <a href="./README.zh.md"><img src="https://img.shields.io/badge/README-中文版本-red"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://twitter.com/YDopensource"><img src="https://img.shields.io/badge/follow-%40YDOpenSource-1DA1F2?logo=twitter&style={style}"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
</div>
<br>

**EmotiVoice** is a powerful and modern open-source text-to-speech engine. EmotiVoice speaks both English and Chinese, and with over 2000 different voices (refer to the [List of Voices](https://github.com/netease-youdao/EmotiVoice/wiki/😊-voice-wiki-page) for details). The most prominent feature is **emotional synthesis**, allowing you to create speech with a wide range of emotions, including happy, excited, sad, angry and others.

An easy-to-use web interface is provided. There is also a scripting interface for batch generation of results. 

Here are a few samples that EmotiVoice generates:


- [Chinese audio sample](https://github.com/netease-youdao/EmotiVoice/assets/3909232/6426d7c1-d620-4bfc-ba03-cd7fc046a4fb)
  
- [English audio sample](https://github.com/netease-youdao/EmotiVoice/assets/3909232/8f272eba-49db-493b-b479-2d9e5a419e26)
  
- [Fun Chinese English audio sample](https://github.com/netease-youdao/EmotiVoice/assets/3909232/a0709012-c3ef-4182-bb0e-b7a2ba386f1c)

## Demo

A demo is hosted on Replicate, [EmotiVoice](https://replicate.com/bramhooimeijer/emotivoice).

## Features under development

- [x] [The EmotiVoice HTTP API](https://github.com/netease-youdao/EmotiVoice/wiki/HTTP-API) was release on December 6th, 2023. Easier to start, faster to use, and with **over 13,000 free calls**. Additionally, users can explore more captivating voices provided by [Zhiyun](https://ai.youdao.com/).
- [ ] Voice Cloning with your personal data (scheduled for release early this month, with examples)

EmotiVoice prioritizes community input and user requests. We welcome your feedback!

## Quickstart

### EmotiVoice Docker image

The easiest way to try EmotiVoice is by running the docker image. You need a machine with a NVidia GPU. If you have not done so, set up NVidia container toolkit by following the instructions for [Linux](https://www.server-world.info/en/note?os=Ubuntu_22.04&p=nvidia&f=2) or [Windows WSL2](https://github.com/nyp-sit/it3103/blob/main/nvidia-docker-wsl2.md). Then EmotiVoice can be run with,

```sh
docker run -dp 127.0.0.1:8501:8501 syq163/emoti-voice:latest
```
The Docker image was updated on November 29, 2023. If you have an older version, please update it by running the following commands:
```sh
docker pull syq163/emoti-voice:latest
docker run -dp 127.0.0.1:8501:8501 syq163/emoti-voice:latest
```
Now open your browser and navigate to http://localhost:8501 to start using EmotiVoice's powerful TTS capabilities.

### Full installation

```sh
conda create -n EmotiVoice python=3.8 -y
conda activate EmotiVoice
pip install torch torchaudio
pip install numpy numba scipy transformers soundfile yacs g2p_en jieba pypinyin
```

### Prepare model files

We recommend that users refer to the wiki page [How to download the pretrained model files](https://github.com/netease-youdao/EmotiVoice/wiki/Pretrained-models) if they encounter any issues.

```sh
git lfs install
git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
```
or, you can run:
```sh
git clone https://www.modelscope.cn/syq163/WangZeJun.git
```

### Inference

1. You can download the [pretrained models](https://drive.google.com/drive/folders/1y6Xwj_GG9ulsAonca_unSGbJ4lxbNymM?usp=sharing) by simply running the following command:
```sh
git clone https://www.modelscope.cn/syq163/outputs.git
```
2. The inference text format is `<speaker>|<style_prompt/emotion_prompt/content>|<phoneme>|<content>`. 

  - inference text example: `8051|Happy|<sos/eos> [IH0] [M] [AA1] [T] engsp4 [V] [OY1] [S] engsp4 [AH0] engsp1 [M] [AH1] [L] [T] [IY0] engsp4 [V] [OY1] [S] engsp1 [AE1] [N] [D] engsp1 [P] [R] [AA1] [M] [P] [T] engsp4 [K] [AH0] [N] [T] [R] [OW1] [L] [D] engsp1 [T] [IY1] engsp4 [T] [IY1] engsp4 [EH1] [S] engsp1 [EH1] [N] [JH] [AH0] [N] . <sos/eos>|Emoti-Voice - a Multi-Voice and Prompt-Controlled T-T-S Engine`.
4. You can get phonemes by `python frontend_en.py data/my_text.txt > data/my_text_for_tts.txt`.

5. Then run:
```sh
TEXT=data/inference/text
python inference_am_vocoder_joint.py \
--logdir prompt_tts_open_source_joint \
--config_folder config/joint \
--checkpoint g_00140000 \
--test_file $TEXT
```
the synthesized speech is under `outputs/prompt_tts_open_source_joint/test_audio`.

1. Or if you just want to use the interactive TTS demo page, run:
```sh
pip install streamlit
streamlit run demo_page.py
```

### OpenAI-compatible-TTS API

Thanks to @lewangdev for adding an OpenAI compatible API [#60](../../issues/60). To set it up, use the following command:

```sh
pip install fastapi
pip install pydub
pip install uvicorn[standard]
uvicorn openaiapi:app --reload
```

### Wiki page

You may find more information from our [wiki](https://github.com/netease-youdao/EmotiVoice/wiki) page.

## Training

To be released.


## Roadmap & Future work

- Our future plan can be found in the [ROADMAP](./ROADMAP.md) file.
- The current implementation focuses on emotion/style control by prompts. It uses only pitch, speed, energy, and emotion as style factors, and does not use gender. But it is not complicated to change it to style/timbre control.
- Suggestions are welcome. You can file issues or [@ydopensource](https://twitter.com/YDopensource) on twitter.


## WeChat group
Welcome to scan the QR code below and join the WeChat group.

<img src="https://github.com/netease-youdao/EmotiVoice/assets/49354974/cc3f4c8b-8369-4e50-89cc-e40d27a6bdeb" alt="qr" width="150"/>

## Credits

- [PromptTTS](https://speechresearch.github.io/prompttts/). The PromptTTS paper is a key basis of this project.
- [LibriTTS](https://www.openslr.org/60/). The LibriTTS dataset is used in training of EmotiVoice.
- [HiFiTTS](https://www.openslr.org/109/). The HiFi TTS dataset is used in training of EmotiVoice.
- [ESPnet](https://github.com/espnet/espnet). 
- [WeTTS](https://github.com/wenet-e2e/wetts)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Transformers](https://github.com/huggingface/transformers)
- [tacotron](https://github.com/keithito/tacotron)
- [KAN-TTS](https://github.com/alibaba-damo-academy/KAN-TTS)
- [StyleTTS](https://github.com/yl4579/StyleTTS)
- [Simbert](https://github.com/ZhuiyiTechnology/simbert)
- [cn2an](https://github.com/Ailln/cn2an). EmotiVoice incorporates cn2an for number processing.

## License

EmotiVoice is provided under the Apache-2.0 License - see the [LICENSE](./LICENSE) file for details.

The interactive page is provided under the [User Agreement](./EmotiVoice_UserAgreement_易魔声用户协议.pdf) file.
