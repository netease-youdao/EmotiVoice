import argparse, time
import sys
import os
import torch, glob, itertools
from yacs import config as CONFIG

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from plot_image import plot_image_sambert
from mel_process import mel_spectrogram_torch
from models.prompt_tts_modified.jets import JETSGenerator, get_segments
from models.hifigan.pretrained_discriminator import Discriminator
from models.hifigan.models import discriminator_loss,  generator_loss, feature_loss
from models.prompt_tts_modified.loss import TTSLoss
from models.prompt_tts_modified.simbert import StyleEncoder
from models.prompt_tts_modified.prompt_dataset import Dataset_PromptTTS as Dataset_PromptTTS_JETS
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings('ignore')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_writer(output_directory):
    logging_path=f'{output_directory}' + "/log"
    if not os.path.exists(logging_path):
        os.makedirs(logging_path, exist_ok=True)
    writer = SummaryWriter(logging_path)
    return writer

def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict




def validate(args, generator, val_loader, iteration, writer, config, device, loss_fn):

    generator.eval()

    with torch.no_grad():
        dec_mel_loss_list = []
        postnet_mel_loss_list = []
        dur_loss_list = []
        pitch_loss_list = []
        energy_loss_list = []
        forwardsum_loss_list = []
        bin_loss_list = []


        for i, batch in enumerate(val_loader):

            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            phoneme_id = batch["phoneme_id"]
            phoneme_lens = batch["phoneme_lens"]
            mel = batch["mel"]
            mel_lens = batch["mel_lens"]
            speaker = batch["speaker"]
            style_embedding = batch["style_embedding"]
            content_embedding = batch["content_embedding"]
            pitch = batch["pitch"]
            energy = batch["energy"]
            wav = batch["wav"]

            output = generator(
                inputs_ling=phoneme_id,
                inputs_style_embedding=style_embedding,
                inputs_content_embedding=content_embedding,
                input_lengths=phoneme_lens,
                inputs_speaker=speaker,
                output_lengths=mel_lens,
                mel_targets=mel,
                pitch_targets=pitch,
                energy_targets=energy,
                cut_flag=False
            )

            y_hat_mel = mel_spectrogram_torch(
                    output["wav_predictions"][:,:,:wav.size(1)].squeeze(1), 
                    config.filter_length, 
                    config.n_mel_channels, 
                    config.sampling_rate, 
                    config.hop_length, 
                    config.win_length, 
                    config.mel_fmin, 
                    config.mel_fmax
                )


            y_mel = mel_spectrogram_torch(
                wav.squeeze(1), 
                config.filter_length, 
                config.n_mel_channels, 
                config.sampling_rate, 
                config.hop_length, 
                config.win_length, 
                config.mel_fmin, 
                config.mel_fmax
            )
            output["dec_outputs"] = y_hat_mel
            output["mel_targets"] = y_mel.transpose(1,2)

            losses = loss_fn(output)

            dec_mel_loss_list.append(losses["dec_mel_loss"].item())
            dur_loss_list.append(losses["dur_loss"].item())
            pitch_loss_list.append(losses["pitch_loss"].item())
            energy_loss_list.append(losses["energy_loss"].item())
            forwardsum_loss_list.append(losses["forwardsum_loss"].item())
            bin_loss_list.append(losses["bin_loss"].item())

            
        dec_mel_loss = sum(dec_mel_loss_list)/len(dec_mel_loss_list)
        dur_loss = sum(dur_loss_list)/len(dur_loss_list)
        pitch_loss = sum(pitch_loss_list)/len(pitch_loss_list)
        energy_loss = sum(energy_loss_list)/len(energy_loss_list)
        forwardsum_loss = sum(forwardsum_loss_list)/len(forwardsum_loss_list)
        bin_loss = sum(bin_loss_list)/len(bin_loss_list)


        message = f'global_step={iteration}, val_dec_mel_loss={dec_mel_loss:0.4f}, val_dur_loss={dur_loss:0.4f}, val_pitch_loss={pitch_loss:0.4f}, val_energy_loss={energy_loss:0.4f}, val_forwardsum_loss={forwardsum_loss:0.4f}, bin_loss={bin_loss:0.4f}, '
        print(message)
        with open(os.path.join(f'{config.output_directory}' + "/log", "train_log.txt"), "a") as f:
            f.write(message + "\n")

    writer.add_scalar('val_dec_mel_loss', dec_mel_loss, global_step=iteration)
    writer.add_scalar('val_dur_loss', dur_loss, global_step=iteration)
    writer.add_scalar('val_pitch_loss', pitch_loss, global_step=iteration)
    writer.add_scalar('val_energy_loss', energy_loss, global_step=iteration)
    writer.add_scalar('val_forwardsum_loss', forwardsum_loss, global_step=iteration)
    writer.add_scalar('val_bin_loss', bin_loss, global_step=iteration)
    
    mel_plots = plot_image_sambert(mel,
                                output["dec_outputs"],
                                mel_lens,
                                phoneme_lens, 
                                save_dir=f'{config.output_directory}', 
                                global_step=iteration, 
                                name='val')
    writer.add_figure('Validation mel_plots', mel_plots, global_step=iteration)
    
    with torch.no_grad():
        T=phoneme_lens[-1]
        output_infer = generator(
                inputs_ling=phoneme_id[-1,:T].unsqueeze(0),
                inputs_style_embedding=style_embedding[-1].unsqueeze(0),
                input_lengths=phoneme_lens[-1].unsqueeze(0),
                inputs_content_embedding=content_embedding[-1].unsqueeze(0),
                inputs_speaker=speaker[-1].unsqueeze(0),
            )
    
    y_hat_mel = mel_spectrogram_torch(
                    output_infer["wav_predictions"].squeeze(1), 
                    config.filter_length, 
                    config.n_mel_channels, 
                    config.sampling_rate, 
                    config.hop_length, 
                    config.win_length, 
                    config.mel_fmin, 
                    config.mel_fmax
                )
    writer.add_audio('generated_audio', output_infer["wav_predictions"].squeeze(1), iteration, 16_000)

    mel_plots_infer = plot_image_sambert(mel,
                                y_hat_mel,
                                mel_lens,
                                phoneme_lens, 
                                save_dir=f'{config.output_directory}', 
                                global_step=iteration, 
                                name='infer')

    writer.add_figure('Inference mel_plots', mel_plots_infer, global_step=iteration)
    generator.train()
    return 


def train(args, config):
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    rank = int(os.environ["LOCAL_RANK"])

    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=args.n_gpus, rank=rank)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    if rank==0:
        print("run!")
        writer = get_writer(config.output_directory)
    
    print("device: ", rank)

    style_encoder = StyleEncoder(config)
    model_CKPT = torch.load(config.style_encoder_ckpt, map_location="cpu")
    model_ckpt = {}
    for key, value in model_CKPT['model'].items():
        new_key = key[7:]
        model_ckpt[new_key] = value
    style_encoder.load_state_dict(model_ckpt, strict=False)

    train_dataset = Dataset_PromptTTS_JETS(config.train_data_path, config, style_encoder)
    data_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=train_dataset.TextMelCollate,
        sampler = data_sampler,
    )
    if rank == 0:
        valid_dataset = Dataset_PromptTTS_JETS(config.valid_data_path, config, style_encoder)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            num_workers=1,
            batch_size=config.batch_size,
            collate_fn=train_dataset.TextMelCollate,
            pin_memory=True,
        )
    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)
    
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    iteration=0


    generator = JETSGenerator(conf).to(device)
    discriminator = Discriminator(conf).to(device)

    os.makedirs(f'{config.output_directory}' + '/ckpt', exist_ok=True)
    cp_g = scan_checkpoint(f'{config.output_directory}' + '/ckpt', 'g_')
    cp_do = scan_checkpoint(f'{config.output_directory}' + '/ckpt', 'do_')

    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        iteration = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if args.load_pretrained_model:
        ckpt=torch.load(f'{config.output_directory}/ckpt/pretrained_generator')
        generator.load_state_dict(ckpt['generator'])
        ckpt=torch.load(f'{config.output_directory}/ckpt/pretrained_discriminator')
        discriminator.load_state_dict(ckpt['discriminator'])
        state_dict_do = None
        last_epoch = -1
        iteration=0

        print()


    generator = DDP(generator, device_ids=[rank]).to(device)
    discriminator = DDP(discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.Adam(generator.parameters(), conf.optimizer.lr, betas=conf.optimizer.betas)
    optim_d = torch.optim.Adam(discriminator.parameters(),
                                conf.optimizer.lr, betas=conf.optimizer.betas)

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=conf.scheduler.gamma, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=conf.scheduler.gamma, last_epoch=last_epoch)

    loss_fn = TTSLoss()

    if rank == 0:
        print("The number of parameters in the model: %0.2f M"%(count_parameters(generator)/1000000.0))
        print(f"Training Start!!! ({config.output_directory})")


    generator.train()
    discriminator.train()

    for epoch in range(max(0, last_epoch), 5_000_000):

        if rank == 0:
            for param_group in optim_g.param_groups:
                print("Current learning rate: " + str(param_group["lr"]))
            print("Epoch: {}".format(epoch+1))
        
        data_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

            phoneme_id = batch["phoneme_id"]
            phoneme_lens = batch["phoneme_lens"]
            mel = batch["mel"]
            mel_lens = batch["mel_lens"]
            speaker = batch["speaker"]
            style_embedding = batch["style_embedding"]
            content_embedding = batch["content_embedding"]
            pitch = batch["pitch"]
            energy = batch["energy"]
            wav = batch["wav"]
        
            output = generator(
                inputs_ling=phoneme_id,
                inputs_style_embedding=style_embedding,
                inputs_content_embedding=content_embedding,
                input_lengths=phoneme_lens,
                inputs_speaker=speaker,
                output_lengths=mel_lens,
                mel_targets=mel,
                pitch_targets=pitch,
                energy_targets=energy,
            )

            y_hat_mel = mel_spectrogram_torch(
                output["wav_predictions"].squeeze(1), 
                config.filter_length, 
                config.n_mel_channels, 
                config.sampling_rate, 
                config.hop_length, 
                config.win_length, 
                config.mel_fmin, 
                config.mel_fmax
            )

            wav = get_segments(
                x=wav.unsqueeze(1),
                start_idxs=output["z_start_idxs"] * (generator.module.upsample_factor if hasattr(generator, "module") else generator.upsample_factor),
                segment_size=output["segment_size"] * (generator.module.upsample_factor if hasattr(generator, "module") else generator.upsample_factor),
            )

            y_mel = mel_spectrogram_torch(
                wav.squeeze(1), 
                config.filter_length, 
                config.n_mel_channels, 
                config.sampling_rate, 
                config.hop_length, 
                config.win_length, 
                config.mel_fmin, 
                config.mel_fmax
            )
            output["dec_outputs"] = y_hat_mel
            output["mel_targets"] = y_mel.transpose(1,2)

            ########################################## Discriminator ##########################################
            optim_d.zero_grad()

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = discriminator(wav, output["wav_predictions"].detach())

            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()




            ########################################## Generator ##########################################

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = discriminator(wav, output["wav_predictions"])
            optim_g.zero_grad()
            loss = loss_fn(output)
            loss_mel = F.l1_loss(y_mel, y_hat_mel)
            loss["dec_mel_loss"]=loss_mel
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            dec_mel_loss = loss["dec_mel_loss"] * 45
            dur_loss = loss["dur_loss"] * 1
            pitch_loss = loss["pitch_loss"] * 1
            energy_loss = loss["energy_loss"] * 1 
            forwardsum_loss = loss["forwardsum_loss"] * 2
            bin_loss = loss["bin_loss"] * 2
            loss_gen = (loss_gen_f + loss_gen_s) * 1
            loss_fm = (loss_fm_f + loss_fm_s)

            loss_gen_all = loss_gen + loss_fm + \
                dec_mel_loss + dur_loss + \
                pitch_loss + energy_loss + \
                forwardsum_loss + bin_loss

    
            loss_gen_all.backward()
            optim_g.step()
            
            iteration += 1

            if rank==0:
                writer.add_scalar('train_loss_fm', loss_fm, global_step=iteration)
                writer.add_scalar('train_loss_gen', loss_gen, global_step=iteration)
                writer.add_scalar('train_dec_mel_loss', dec_mel_loss, global_step=iteration)
                writer.add_scalar('train_dur_loss', dur_loss, global_step=iteration)
                writer.add_scalar('train_pitch_loss', pitch_loss, global_step=iteration)
                writer.add_scalar('train_energy_loss', energy_loss, global_step=iteration)
                writer.add_scalar('train_forwardsum_loss', forwardsum_loss, global_step=iteration)
                writer.add_scalar('train_bin_loss', bin_loss, global_step=iteration)
                message = f'global_step={iteration}, train_dec_mel_loss={dec_mel_loss:0.4f}, train_dur_loss={dur_loss:0.4f}, train_pitch_loss={pitch_loss:0.4f}, train_energy_loss={energy_loss:0.4f}, train_forwardsum_loss={forwardsum_loss:0.4f}, bin_loss={bin_loss:0.4f}, train_loss_fm={loss_fm:0.4f}, train_loss_gen={loss_gen:0.4f},  s/b={time.time() - start_b:4.3f}'
                if iteration % (config.iters_per_validation) == 0:

                    validate(args, generator, valid_loader, iteration, writer, config, device, loss_fn)
                    print(message)
                    with open(os.path.join(f'{config.output_directory}' + "/log", "train_log.txt"), "a") as f:
                        f.write(message + "\n")
                elif iteration % (config.iters_per_validation//10) == 0:

                    print(message)
                    with open(os.path.join(f'{config.output_directory}' + "/log", "train_log.txt"), "a") as f:
                        f.write(message + "\n")

                if iteration % (config.iters_per_checkpoint) == 0:
                    checkpoint_path = "{}/g_{:08d}".format(f'{config.output_directory}' + '/ckpt', iteration)
                    save_checkpoint(checkpoint_path, {'generator': (generator.module if hasattr(generator, 'module') else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(f'{config.output_directory}' + '/ckpt', iteration)
                    save_checkpoint(checkpoint_path, 
                                    {'discriminator': (discriminator.module if hasattr(discriminator, 'module')
                                                        else discriminator).state_dict(),
                                    'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': iteration,
                                    'epoch': epoch})

                if iteration == (config.train_steps):
                    writer.close()
                    print("TRAINING DONE!")
                    break
        scheduler_g.step()
        scheduler_d.step()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config_folder", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--load_pretrained_model", default=False)
    args = p.parse_args() 

    ##################################################
    sys.path.append(args.config_folder)
    from config import Config
    config = Config()
    ##################################################
    n_gpus = torch.cuda.device_count()
    args.n_gpus=n_gpus


    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    # os.environ[
    #     "TORCH_DISTRIBUTED_DEBUG"
    # ] = "DETAIL"
    train(args, config)


if __name__ == '__main__':
    main()