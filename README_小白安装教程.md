## 小白安装教程

#### 环境条件：设备有GPU、已经安装cuda

说明：这是针对Linux环境安装的教程，其他系统可作为参考。

#### 1、创建并进入conda环境

```
conda create -n EmotiVoice python=3.8
conda init
conda activate EmotiVoice
```

如果你不想使用conda环境，也可以省略该步骤，但要保证python版本为3.8


#### 2、安装git-lfs

如果是Ubuntu则执行

```
sudo apt update
sudo apt install git
sudo apt-get install git-lfs
```

CentOS则执行

```
sudo yum update
sudo yum install git
sudo yum install git-lfs
```



#### 3、克隆仓库

```
git lfs install
git lfs clone https://github.com/netease-youdao/EmotiVoice.git
```



#### 4、安装依赖

```
pip install torch torchaudio
pip install numpy numba scipy transformers soundfile yacs g2p_en jieba pypinyin pypinyin_dict
python -m nltk.downloader "averaged_perceptron_tagger_eng"
```



<a id="step5"></a>

#### 5、下载预训练模型文件

(1)首先进入项目文件夹

```
cd EmotiVoice
```

(2)执行下面命令

```
git lfs clone https://huggingface.co/WangZeJun/simbert-base-chinese WangZeJun/simbert-base-chinese
```

或者

```
git clone https://www.modelscope.cn/syq163/WangZeJun.git
```

上面两种下载方式二选一即可。

(3)第三步下载ckpt模型

```
git clone https://www.modelscope.cn/syq163/outputs.git
```

上面步骤完成后，项目文件夹内会多 `WangZeJun` 和 `outputs` 文件夹，下面是项目文件结构

```
├── Dockerfile
├── EmotiVoice_UserAgreement_易魔声用户协议.pdf
├── demo_page.py
├── frontend.py
├── frontend_cn.py
├── frontend_en.py
├── WangZeJun
│   └── simbert-base-chinese
│       ├── README.md
│       ├── config.json
│       ├── pytorch_model.bin
│       └── vocab.txt
├── outputs
│   ├── README.md
│   ├── configuration.json
│   ├── prompt_tts_open_source_joint
│   │   └── ckpt
│   │       ├── do_00140000
│   │       └── g_00140000
│   └── style_encoder
│       └── ckpt
│           └── checkpoint_163431
```



#### 6、运行UI交互界面

(1)安装streamlit

```
pip install streamlit
```

(2)启动

打开运行后显示的server地址，如何正常显示页面则部署完成。

```
streamlit run demo_page.py --server.port 6006 --logger.level debug
```



#### 7、启动API服务

安装依赖

```
pip install fastapi pydub uvicorn[standard] pyrubberband
```

在6006端口启动服务(端口可根据自己的需求修改)

```
uvicorn openaiapi:app --reload --port 6006
```

接口文档地址：你的服务地址+`/docs`
声音试听地址：`/`

&emsp;

#### 8、遇到错误

**(1) 运行UI界面后，打开页面一直显示 "Please wait..." 或者显示一片空白**

原因：

这个错误可能是由于CORS（跨域资源共享）保护配置错误。

解决方法：

在启动时加上一个 `server.enableCORS=false` 参数，即使用下面命令启动程序

```
streamlit run demo_page.py --server.port 6006 --logger.level debug --server.enableCORS=false
```

如果通过临时禁用 CORS 保护解决了问题，建议重新启用 CORS 保护并设置正确的 URL 和端口。

&emsp;

**(2) 运行报错 raise BadZipFile("File is not a zip file") zipfile.BadZipFile: File is not a zip file**

原因：

这可能是由于缺少 `averaged_perceptron_tagger`  这个`nltk`中用于词性标注的一个包，它包含了一个基于平均感知器算法的词性标注器。如果你在代码中使用了这个标注器，但是没有预先下载对应的数据包，就会遇到错误，提示你缺少`averaged_perceptron_tagger.zip`文件。当然也有可能是缺少 `cmudict` CMU 发音词典数据包文件。

正常来说，初次运行程序NLTK会自动下载使用的相关数据包，debug模式下运行会显示如下信息

```
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /root/nltk_data...
[nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.
[nltk_data] Downloading package cmudict to /root/nltk_data...
[nltk_data]   Unzipping corpora/cmudict.zip.
```

可能由于网络(需科学上网)等原因，没能自动下载成功，因此缺少相关文件导致加载报错。



解决方法：重新下载缺少的数据包文件



1)方法一

创建一个 download.py文件，在其中编写如下代码

```
import nltk
print(nltk.data.path)
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
```

保存并运行

```
python download.py
```

这将显示其文件索引位置，并自动下载 缺少的 `averaged_perceptron_tagger.zip`和 `cmudict.zip` 文件到/root/nltk_data目录下的子目录，下载完成后查看根目录下是否有`nltk_data`文件夹，并将其中的压缩包都解压。

&emsp;

2)方法二

如果通过上面代码还是无法正常下载数据包 ，也可以打开以下地址手动搜索并下载压缩包文件(需科学上网)

```
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml
```

其中下面是`averaged_perceptron_tagger.zip` 和`cmudict.zip` 数据包文件的下载地址

```
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip
https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/cmudict.zip
```

然后将该压缩包文件上传至(1)运行`python download.py`时打印显示的文件索引位置，如 `/root/nltk_data`  或者 `/root/miniconda3/envs/EmotiVoice/nltk_data` 等类似目录下，如果没有则创建一个，然后将zip压缩包解压。

&emsp;

解压后nltk_data目录结构应该是下面这样

```
├── nltk_data
│   ├── corpora
│   │   ├── cmudict
│   │   │   ├── README
│   │   │   └── cmudict
│   │   └── cmudict.zip
│   └── taggers
│       ├── averaged_perceptron_tagger
│       │   └── averaged_perceptron_tagger.pickle
│       └── averaged_perceptron_tagger.zip
```

&emsp;

**(3) 报错 AttributeError: 'NoneType' object has no attribute 'seek'.** 

原因：未找到模型文件

解决方法：大概率是你未下载模型文件或者存放路径不正确，查看自己下载的模型文件是否存在，即outputs文件夹存放路径和里面的模型文件是否正确，正确结构可参考 [第五步](#step5) 中的项目结构。

&emsp;

**(4) 运行API服务出错 ImportError: cannot import name 'Doc' from 'typing_extensions'**

原因：typing_extensions 版本问题

解决方法：

尝试将`typing_extensions`升级至最新版本，如果已经是最新版本，则适当降低版本，以下版本在`fastapi V0.104.1`测试正常。

```
pip install typing_extensions==4.8.0 --force-reinstall
```

&emsp;

**(5) 请求文本转语音接口时报错 500 Internal Server Error ，FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'**

原因：未安装ffmpeg

解决方法：

执行以下命令进行安装，如果是Ubuntu执行

```
sudo apt update
sudo apt install ffmpeg
```

CentOS则执行

```
sudo yum install epel-release
sudo yum install ffmpeg
```

安装完成后，你可以在终端中运行以下命令来验证"ffmpeg"是否成功安装：

```
ffmpeg -version
```

如果安装成功，你将看到"ffmpeg"的版本信息。

&emsp;
