# EmotiVoice - 多音色提示控制TTS

[English](README.md)

**EmotiVoice**是一个强大的开源TTS引擎，支持中英文双语，包含2000多种不同的音色，以及特色的**情感合成**功能，支持合成包含快乐、兴奋、悲伤、愤怒等广泛情感的语音。

EmotiVoice提供一个易于使用的web界面，还有用于批量生成结果的脚本接口。

以下是EmotiVoice生成的几个示例:

- [Chinese audio sample](https://github.com/netease-youdao/EmotiVoice/assets/3909232/af038c31-632a-4626-ad23-3f9a92b2bec2)

- [English audio sample](https://github.com/netease-youdao/EmotiVoice/assets/3909232/bf94e5b4-87bf-40b1-882e-ad96d4084864)

# 快速入门

## EmotiVoice Docker镜像

尝试EmotiVoice最简单的方法是运行docker镜像。你需要一台带有NVidia GPU的机器。先按照[Linux](https://www.server-world.info/en/note?os=Ubuntu_22.04&p=nvidia&f=2)和[Windows WSL2](https://zhuanlan.zhihu.com/p/653173679)平台的说明安装NVidia容器工具包。然后可以直接运行EmotiVoice镜像：

```sh
docker run -dp 127.0.0.1:8501:8501 syq163/emoti-voice:latest
```

现在打开浏览器，导航到 http://localhost:8501 ，就可以体验EmotiVoice强大的TTS功能。

## 完整安装

```sh
conda create -n EmotiVoice python=3.8 -y
conda activate EmotiVoice
pip install torch torchaudio
pip install numpy numba scipy transformers==4.26.1 soundfile yacs g2p_en jieba pypinyin
```

## 准备模型文件

```sh
git lfs install
git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
```

## 推理

1. 下载[预训练模型](https://drive.google.com/drive/folders/1y6Xwj_GG9ulsAonca_unSGbJ4lxbNymM?usp=sharing), 然后运行:

```sh
mkdir -p outputs/style_encoder/ckpt
mkdir -p outputs/prompt_tts_open_source_joint/ckpt
```

2. 将`g_*`, `do_*`文件放到`outputs/prompt_tts_open_source_joint/ckpt`，将`checkpoint_*`放到`outputs/style_encoder/ckpt`中.
3. 推理输入文本格式是：`<speaker>|<style_prompt/emotion_prompt/content>|<phoneme>|<content>`. 
  - 例如: `Maria_Kasper|非常开心|<sos/eos>  uo3 sp1 l ai2 sp0 d ao4 sp1 b ei3 sp0 j ing1 sp3 q ing1 sp0 h ua2 sp0 d a4 sp0 x ve2 <sos/eos>|我来到北京，清华大学`.
4. 其中的音素（phonemes）可以这样得到：`python frontend.py data/my_text.txt > data/my_text_for_tts.txt`.

5. 然后运行：
```sh
TEXT=data/inference/text
python inference_am_vocoder_joint.py \
--logdir prompt_tts_open_source_joint \
--config_folder config/joint \
--checkpoint g_00140000 \
--test_file $TEXT
```
合成的语音结果在：`outputs/prompt_tts_open_source_joint/test_audio`.

6. 或者你可以直接使用交互的网页界面：
```sh
pip install streamlit
streamlit run demo_page.py
```

# 训练

待推出。

# 未来工作

* 当前的实现侧重于通过提示控制情绪/风格。它只使用音高、速度、能量和情感作为风格因素，而不使用性别。但是将其更改为样式、音色控制并不复杂，类似于PromptTTS的原始闭源实现。

# 致谢

- [PromptTTS](https://speechresearch.github.io/prompttts/). PromptTTS论文是本工作的重要基础。
- [LibriTTS](https://www.openslr.org/60/). 训练使用了LibriTTS开放数据集。
- [HiFiTTS](https://www.openslr.org/109/). 训练使用了HiFi TTS开放数据集。
- [ESPnet](https://github.com/espnet/espnet). 
- [WeTTS](https://github.com/wenet-e2e/wetts)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [Transformers](https://github.com/huggingface/transformers)
- [tacotron](https://github.com/keithito/tacotron)
- [KAN-TTS](https://github.com/alibaba-damo-academy/KAN-TTS)
- [StyleTTS](https://github.com/yl4579/StyleTTS)


# 许可

EmotiVoice是根据Apache-2.0许可证提供的 - 有关详细信息，请参阅[许可证文件](./LICENSE)。

交互的网页是根据[用户协议](./EmotiVoice_UserAgreement_易魔声用户协议.pdf)提供的。
