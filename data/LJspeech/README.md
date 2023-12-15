

# 😊 LJSpeech Recipe 

This is the recipe of English single female speaker TTS model with LJSpeech corpus.

## Guide For Finetuning
- [Environments Installation](#environments-installation)
- [Step0 Download Data](#step0-download-data)
- [Step1 Preprocess Data](#step1-preprocess-data)
- [Step2 Run MFA (Optional, but Recommended)](#step2-run-mfa-optional-but-recommended)
- [Step3 Prepare for training](#step3-prepare-for-training)
- [Step4 Start training](#step4-finetune-your-model)
- [Step5 Inference](#step5-inference)

Run EmotiVoice Finetuning on Google Colab Notebook! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dDAyjoYGcDGwYpHI3Oj2_OIV-7DIdx2L?usp=sharing) 

### Environments Installation

create conda enviroment
```bash
conda create -n EmotiVoice python=3.8 -y
conda activate EmotiVoice
```
then run:
```bash
pip install EmotiVoice[train]
# or
git clone https://github.com/netease-youdao/EmotiVoice
pip install -e .[train]
```
Additionally, it is important to prepare the pre-trained models as mentioned in the [pretrained models](https://github.com/netease-youdao/EmotiVoice/wiki/Pretrained-models).

### Step0 Download Data

```bash
mkdir data/LJspeech/raw

# download
wget -P data/LJspeech/raw http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
# extract
tar -xjf data/LJspeech/raw/LJSpeech-1.1.tar.bz2 -C data/LJspeech/raw
```

### Step1 Preprocess Data

```bash
# format data
python data/LJspeech/src/step1_clean_raw_data.py \
--data_dir data/LJspeech

# get phoneme
python data/LJspeech/src/step2_get_phoneme.py \
--data_dir data/LJspeech
```

### Step2 Run MFA (Optional, but Recommended!)

```bash
# MFA environment install
conda install -c conda-forge kaldi sox librosa biopython praatio tqdm requests colorama pyyaml pynini openfst baumwelch ngram postgresql -y
pip install pgvector hdbscan montreal-forced-aligner

# MFA Step1
python  mfa/step1_create_dataset.py \
--data_dir data/LJspeech

# MFA Step2
python mfa/step2_prepare_data.py \
--dataset_dir data/LJspeech/mfa \
--wav data/LJspeech/mfa/wav.txt \
--speaker data/LJspeech/mfa/speaker.txt \
--text data/LJspeech/mfa/text.txt

# MFA Step3
python mfa/step3_prepare_special_tokens.py \
--special_tokens data/LJspeech/mfa/special_token.txt

# MFA Step4
python mfa/step4_convert_text_to_phn.py \
--text data/LJspeech/mfa/text.txt \
--special_tokens data/LJspeech/mfa/special_token.txt \
--output data/LJspeech/mfa/text.txt

# MFA Step5
python mfa/step5_prepare_alignment.py \
--wav  data/LJspeech/mfa/wav.txt \
--speaker  data/LJspeech/mfa/speaker.txt \
--text  data/LJspeech/mfa/text.txt \
--special_tokens  data/LJspeech/mfa/special_token.txt \
--pronounciation_dict  data/LJspeech/mfa/mfa_pronounciation_dict.txt \
--output_dir  data/LJspeech/mfa/lab

# MFA Step6
mfa validate \
--overwrite \
--clean \
--single_speaker \
data/LJspeech/mfa/lab \
data/LJspeech/mfa/mfa_pronounciation_dict.txt

mfa train \
--overwrite \
--clean \
--single_speaker \
data/LJspeech/mfa/lab \
data/LJspeech/mfa/mfa_pronounciation_dict.txt \
data/LJspeech/mfa/mfa/mfa_model.zip \
data/LJspeech/mfa/TextGrid

mfa align \
--single_speaker \
data/LJspeech/mfa/lab \
data/LJspeech/mfa/mfa_pronounciation_dict.txt \
data/LJspeech/mfa/mfa/mfa_model.zip \
data/LJspeech/mfa/TextGrid

# MFA Step7
python mfa/step7_gen_alignment_from_textgrid.py \
--wav data/LJspeech/mfa/wav.txt \
--speaker data/LJspeech/mfa/speaker.txt \
--text data/LJspeech/mfa/text.txt \
--special_tokens data/LJspeech/mfa/special_token.txt \
--text_grid data/LJspeech/mfa/TextGrid \
--aligned_wav data/LJspeech/mfa/aligned_wav.txt \
--aligned_speaker data/LJspeech/mfa/aligned_speaker.txt \
--duration data/LJspeech/mfa/duration.txt \
--aligned_text data/LJspeech/mfa/aligned_text.txt \
--reassign_sp True

# MFA Step8
python mfa/step8_make_data_list.py \
--wav data/LJspeech/mfa/aligned_wav.txt \
--speaker data/LJspeech/mfa/aligned_speaker.txt \
--text data/LJspeech/mfa/aligned_text.txt \
--duration data/LJspeech/mfa/duration.txt \
--datalist_path data/LJspeech/mfa/datalist.jsonl

# MFA Step9
python mfa/step9_datalist_from_mfa.py \
--data_dir data/LJspeech
```

### Step3 Prepare for training

```bash
python prepare_for_training.py --data_dir data/LJspeech --exp_dir exp/LJspeech
```
__Please check and change the training and valid file path in the `exp/LJspeech/config/config.py`, especially:__
- `model_config_path`: corresponing model config file
- `DATA_DIR`: data dir
- `train_data_path` and `valid_data_path`: training file and valid file. Change to `datalist_mfa.jsonl` if you run Step2
- `batch_size`

### Step4 Finetune Your Model

```bash
torchrun \
--nproc_per_node=1 \
--master_port 8008 \
train_am_vocoder_joint.py \
--config_folder exp/LJspeech/config \
--load_pretrained_model True
```

Training tips:

- You can run tensorboad by
```
tensorboard --logdir=exp/LJspeech
```
- The model checkpoints are saved at `exp/LJspeech/ckpt`.
- The bert features are extracted in the first epoch and saved in `exp/LJspeech/tmp/` folder, you can change the path in `exp/LJspeech/config/config.py`.


### Step5 Inference


```bash
TEXT=data/inference/text
python inference_am_vocoder_exp.py \
--config_folder exp/LJspeech/config \
--checkpoint g_00010000 \
--test_file $TEXT
```
__Please change the speaker name in the `data/inference/text`__

the synthesized speech is under `exp/LJspeech/test_audio`.