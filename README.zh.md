<font size=4> README: <a href="./README.md">EN</a> | 中文  </font>


<div align="center">
    <h1>EmotiVoice易魔声 😊: 多音色提示控制TTS</h1>
</div>

<div align="center">
    <a href="./README.md"><img src="https://img.shields.io/badge/README-EN-red"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-yellow"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://twitter.com/YDopensource"><img src="https://img.shields.io/badge/follow-%40YDOpenSource-1DA1F2?logo=twitter&style={style}"></a>
    &nbsp;&nbsp;&nbsp;&nbsp;
</div>
<br>


**EmotiVoice**是一个强大的开源TTS引擎，支持中英文双语，包含2000多种不同的音色，以及特色的**情感合成**功能，支持合成包含快乐、兴奋、悲伤、愤怒等广泛情感的语音。

EmotiVoice提供一个易于使用的web界面，还有用于批量生成结果的脚本接口。

以下是EmotiVoice生成的几个示例:

- [Chinese audio sample](https://github.com/netease-youdao/EmotiVoice/assets/3909232/6426d7c1-d620-4bfc-ba03-cd7fc046a4fb)
  
- [English audio sample](https://github.com/netease-youdao/EmotiVoice/assets/3909232/8f272eba-49db-493b-b479-2d9e5a419e26)
  
- [Fun Chinese English audio sample](https://github.com/netease-youdao/EmotiVoice/assets/3909232/a0709012-c3ef-4182-bb0e-b7a2ba386f1c)

## 开发中的特性

- [x] [易魔声 HTTP API](https://github.com/netease-youdao/EmotiVoice/wiki/HTTP-API) 已于2023年12月6日发布上线。更易上手（无需任何安装配置），更快更稳定，单账户提供**超过 13,000 次免费调用**。此外，用户还可以使用[智云](https://ai.youdao.com/)提供的其它迷人的声音。
- [ ] 用你自己的数据定制音色（2023年12月早些时间推出）

易魔声倾听社区需求并积极响应，期待您的反馈！

## 快速入门

### EmotiVoice Docker镜像

尝试EmotiVoice最简单的方法是运行docker镜像。你需要一台带有NVidia GPU的机器。先按照[Linux](https://www.server-world.info/en/note?os=Ubuntu_22.04&p=nvidia&f=2)和[Windows WSL2](https://zhuanlan.zhihu.com/p/653173679)平台的说明安装NVidia容器工具包。然后可以直接运行EmotiVoice镜像：

```sh
docker run -dp 127.0.0.1:8501:8501 syq163/emoti-voice:latest
```

Docker镜像更新于2023年11月29号。如果你使用了老的版本，推荐运行如下命令进行更新：
```sh
docker pull syq163/emoti-voice:latest
docker run -dp 127.0.0.1:8501:8501 syq163/emoti-voice:latest
```

现在打开浏览器，导航到 http://localhost:8501 ，就可以体验EmotiVoice强大的TTS功能。

### 完整安装

```sh
conda create -n EmotiVoice python=3.8 -y
conda activate EmotiVoice
pip install torch torchaudio
pip install numpy numba scipy transformers soundfile yacs g2p_en jieba pypinyin
```

### 准备模型文件

强烈推荐用户参考[如何下载预训练模型文件](https://github.com/netease-youdao/EmotiVoice/wiki/Pretrained-models)的维基页面，尤其遇到问题时。

```sh
git lfs install
git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
```

或者你可以运行:
```sh
git clone https://www.modelscope.cn/syq163/WangZeJun.git
```

### 推理

1. 通过简单运行如下命令来下载[预训练模型](https://drive.google.com/drive/folders/1y6Xwj_GG9ulsAonca_unSGbJ4lxbNymM?usp=sharing):

```sh
git clone https://www.modelscope.cn/syq163/outputs.git
```

2. 推理输入文本格式是：`<speaker>|<style_prompt/emotion_prompt/content>|<phoneme>|<content>`. 

  - 例如: `8051|非常开心|<sos/eos>  uo3 sp1 l ai2 sp0 d ao4 sp1 b ei3 sp0 j ing1 sp3 q ing1 sp0 h ua2 sp0 d a4 sp0 x ve2 <sos/eos>|我来到北京，清华大学`.
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

### 类OpenAI TTS的API

非常感谢 @lewangdev 的相关该工作 [#60](../../issues/60)。通过运行如下命令来完成配置：

```sh
pip install fastapi
pip install pydub
pip install uvicorn[standard]
uvicorn openaiapi:app --reload
```

### Wiki页面

如果遇到问题，或者想获取更多详情，请参考 [wiki](https://github.com/netease-youdao/EmotiVoice/wiki) 页面。

## 训练

待推出。

## 路线图和未来的工作

- 我们未来的计划可以在 [ROADMAP](./ROADMAP.md) 文件中找到。

- 当前的实现侧重于通过提示控制情绪/风格。它只使用音高、速度、能量和情感作为风格因素，而不使用性别。但是将其更改为样式、音色控制并不复杂，类似于PromptTTS的原始闭源实现。

## 微信群

欢迎扫描下方左侧二维码加入微信群。商业合作扫描右侧个人二维码。

<img src="https://github.com/netease-youdao/EmotiVoice/assets/49354974/cc3f4c8b-8369-4e50-89cc-e40d27a6bdeb" alt="qr" width="150"/>
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://github.com/netease-youdao/EmotiVoice/assets/3909232/94ee0824-0304-4487-8682-664fafd09cdf" alt="qr" width="150"/>

## 致谢

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
- [Simbert](https://github.com/ZhuiyiTechnology/simbert)
- [cn2an](https://github.com/Ailln/cn2an). 易魔声集成了cn2an来处理数字。

## 许可

EmotiVoice是根据Apache-2.0许可证提供的 - 有关详细信息，请参阅[许可证文件](./LICENSE)。

交互的网页是根据[用户协议](./EmotiVoice_UserAgreement_易魔声用户协议.pdf)提供的。
