

# 😊 DataBaker Recipe 

This is the recipe of Chinese single female speaker TTS model with DataBaker corpus.

## Guide For Finetuning
- [Environments Installation](#environments-installation)
- [Step0 Download Data](#step0-download-data)
- [Step1 Preprocess Data](#step1-preprocess-data)
- [Step2 Run MFA (Optional)](#step2-run-mfa-optional-since-we-already-have-labeled-prosody)
- [Step3 Prepare for training](#step3-prepare-for-training)
- [Step4 Start training](#step4-finetune-your-model)
- [Step5 Inference](#step5-inference)

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
mkdir data/DataBaker/raw

# download
# please download the data from https://en.data-baker.com/datasets/freeDatasets/, and place the extracted BZNSYP folder under data/DataBaker/raw
```

### Step1 Preprocess Data

For this recipe, since DataBaker has already provided phoneme labels, we will simply utilize that information.

```bash
# format data
python data/DataBaker/src/step1_clean_raw_data.py \
--data_dir data/DataBaker

# get phoneme
python data/DataBaker/src/step2_get_phoneme.py \
--data_dir data/DataBaker
```

If you have prepared your own data with only text labels, you can obtain phonemes using the Text-to-Speech (TTS) frontend. For example, you can run the following command: `python data/DataBaker/src/step2_get_phoneme.py --data_dir data/DataBaker --generate_phoneme True`. However, please note that in this specific DataBaker's recipe, you should omit this command.



### Step2 Run MFA (Optional, since we already have labeled prosody)

Please be aware that in this particular DataBaker's recipe, **you should skip this step**. Nonetheless, if you have already prepared your own data with only text labels, the following commands might assist you:

```bash
# MFA environment install
conda install -c conda-forge kaldi sox librosa biopython praatio tqdm requests colorama pyyaml pynini openfst baumwelch ngram postgresql -y
pip install pgvector hdbscan montreal-forced-aligner

# MFA Step1
python  mfa/step1_create_dataset.py \
--data_dir data/DataBaker

# MFA Step2
python mfa/step2_prepare_data.py \
--dataset_dir data/DataBaker/mfa \
--wav data/DataBaker/mfa/wav.txt \
--speaker data/DataBaker/mfa/speaker.txt \
--text data/DataBaker/mfa/text.txt

# MFA Step3
python mfa/step3_prepare_special_tokens.py \
--special_tokens data/DataBaker/mfa/special_token.txt

# MFA Step4
python mfa/step4_convert_text_to_phn.py \
--text data/DataBaker/mfa/text.txt \
--special_tokens data/DataBaker/mfa/special_token.txt \
--output data/DataBaker/mfa/text.txt

# MFA Step5
python mfa/step5_prepare_alignment.py \
--wav  data/DataBaker/mfa/wav.txt \
--speaker  data/DataBaker/mfa/speaker.txt \
--text  data/DataBaker/mfa/text.txt \
--special_tokens  data/DataBaker/mfa/special_token.txt \
--pronounciation_dict  data/DataBaker/mfa/mfa_pronounciation_dict.txt \
--output_dir  data/DataBaker/mfa/lab

# MFA Step6
mfa validate \
--overwrite \
--clean \
--single_speaker \
data/DataBaker/mfa/lab \
data/DataBaker/mfa/mfa_pronounciation_dict.txt

mfa train \
--overwrite \
--clean \
--single_speaker \
data/DataBaker/mfa/lab \
data/DataBaker/mfa/mfa_pronounciation_dict.txt \
data/DataBaker/mfa/mfa/mfa_model.zip \
data/DataBaker/mfa/TextGrid

mfa align \
--single_speaker \
data/DataBaker/mfa/lab \
data/DataBaker/mfa/mfa_pronounciation_dict.txt \
data/DataBaker/mfa/mfa/mfa_model.zip \
data/DataBaker/mfa/TextGrid

# MFA Step7
python mfa/step7_gen_alignment_from_textgrid.py \
--wav data/DataBaker/mfa/wav.txt \
--speaker data/DataBaker/mfa/speaker.txt \
--text data/DataBaker/mfa/text.txt \
--special_tokens data/DataBaker/mfa/special_token.txt \
--text_grid data/DataBaker/mfa/TextGrid \
--aligned_wav data/DataBaker/mfa/aligned_wav.txt \
--aligned_speaker data/DataBaker/mfa/aligned_speaker.txt \
--duration data/DataBaker/mfa/duration.txt \
--aligned_text data/DataBaker/mfa/aligned_text.txt \
--reassign_sp True

# MFA Step8
python mfa/step8_make_data_list.py \
--wav data/DataBaker/mfa/aligned_wav.txt \
--speaker data/DataBaker/mfa/aligned_speaker.txt \
--text data/DataBaker/mfa/aligned_text.txt \
--duration data/DataBaker/mfa/duration.txt \
--datalist_path data/DataBaker/mfa/datalist.jsonl

# MFA Step9
python mfa/step9_datalist_from_mfa.py \
--data_dir data/DataBaker
```

### Step3 Prepare for training

```bash
python prepare_for_training.py --data_dir data/DataBaker --exp_dir exp/DataBaker
```
__Please check and change the training and valid file path in the `exp/DataBaker/config/config.py`, especially:__
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
--config_folder exp/DataBaker/config \
--load_pretrained_model True
```

Training tips:

- You can run tensorboad by
```
tensorboard --logdir=exp/DataBaker
```
- The model checkpoints are saved at `exp/DataBaker/ckpt`.
- The bert features are extracted in the first epoch and saved in `exp/DataBaker/tmp/` folder, you can change the path in `exp/DataBaker/config/config.py`.


### Step5 Inference


```bash
TEXT=data/inference/text
python inference_am_vocoder_exp.py \
--config_folder exp/DataBaker/config \
--checkpoint g_00010000 \
--test_file $TEXT
```
__Please change the speaker names in the `data/inference/text`__

the synthesized speech is under `exp/DataBaker/test_audio`.