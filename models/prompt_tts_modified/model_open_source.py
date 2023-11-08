"""
This code is modified from https://github.com/espnet/espnet.
"""

import torch
import torch.nn as nn


from models.prompt_tts_modified.modules.encoder import Encoder
from models.prompt_tts_modified.modules.variance import DurationPredictor, VariancePredictor
from models.prompt_tts_modified.modules.alignment import AlignmentModule, GaussianUpsampling, viterbi_decode, average_by_duration
from models.prompt_tts_modified.modules.initialize import initialize

class PromptTTS(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.encoder = Encoder(
            attention_dim=config.model.encoder_n_hidden,
            attention_heads=config.model.encoder_n_heads,
            linear_units=config.model.encoder_n_hidden * 4,            
            num_blocks=config.model.encoder_n_layers,
            dropout_rate=config.model.encoder_p_dropout,
            positional_dropout_rate=config.model.encoder_p_dropout,
            attention_dropout_rate=config.model.encoder_p_dropout,
            normalize_before=True,
            concat_after=False,
            positionwise_conv_kernel_size=config.model.encoder_kernel_size_conv_mod,
            stochastic_depth_rate=0.0,
        )

        self.decoder = Encoder(
            attention_dim=config.model.decoder_n_hidden,
            attention_heads=config.model.decoder_n_heads,
            linear_units=config.model.decoder_n_hidden * 4,            
            num_blocks=config.model.decoder_n_layers,
            dropout_rate=config.model.decoder_p_dropout,
            positional_dropout_rate=config.model.decoder_p_dropout,
            attention_dropout_rate=config.model.decoder_p_dropout,
            normalize_before=True,
            concat_after=False,
            positionwise_conv_kernel_size=config.model.decoder_kernel_size_conv_mod,
            stochastic_depth_rate=0.0,
        )

        self.duration_predictor = DurationPredictor(
            idim=config.model.encoder_n_hidden,
            n_layers=config.model.duration_n_layers,
            n_chans=config.model.variance_n_hidden,
            kernel_size=config.model.duration_kernel_size,
            dropout_rate=config.model.duration_p_dropout,
        )

        self.pitch_predictor = VariancePredictor(
            idim=config.model.encoder_n_hidden,
            n_layers=config.model.variance_n_layers, #pitch_predictor_layers=5 in paddlespeech fs2
            n_chans=config.model.variance_n_hidden,
            kernel_size=config.model.variance_kernel_size, #pitch_predictor_kernel_size=5 in paddlespeech fs2
            dropout_rate=config.model.variance_p_dropout, #pitch_predictor_dropout=0.5 in paddlespeech fs2
        )
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=config.model.encoder_n_hidden,
                kernel_size=config.model.variance_embed_kernel_size, #pitch_embed_kernel_size=1 in paddlespeech fs2
                padding=(config.model.variance_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(config.model.variance_embde_p_dropout), #pitch_embed_dropout=0.0
        )
        self.energy_predictor = VariancePredictor(
            idim=config.model.encoder_n_hidden,
            n_layers=2,
            n_chans=config.model.variance_n_hidden,
            kernel_size=3,
            dropout_rate=config.model.variance_p_dropout,
        )
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=config.model.encoder_n_hidden,
                kernel_size=config.model.variance_embed_kernel_size,
                padding=(config.model.variance_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(config.model.variance_embde_p_dropout),
        )

        self.length_regulator = GaussianUpsampling()
        self.alignment_module = AlignmentModule(config.model.encoder_n_hidden, config.n_mels)

        self.to_mel = nn.Linear(
            in_features=config.model.decoder_n_hidden, 
            out_features=config.n_mels,
        )


        self.spk_tokenizer = nn.Embedding(config.n_speaker, config.model.encoder_n_hidden)
        self.src_word_emb = nn.Embedding(config.n_vocab, config.model.encoder_n_hidden)
        self.embed_projection1 = nn.Linear(config.model.encoder_n_hidden * 2 + config.model.bert_embedding * 2, config.model.encoder_n_hidden)

        initialize(self, "xavier_uniform")

    def forward(self, inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding , inputs_content_embedding, mel_targets=None, output_lengths=None, pitch_targets=None, energy_targets=None, alpha=1.0):
        
        B = inputs_ling.size(0)
        T = inputs_ling.size(1)
        src_mask = self.get_mask_from_lengths(input_lengths)
        token_embed = self.src_word_emb(inputs_ling)
        x, _ = self.encoder(token_embed, ~src_mask.unsqueeze(-2))
        speaker_embedding = self.spk_tokenizer(inputs_speaker)
        x = torch.concat([x, speaker_embedding.unsqueeze(1).expand(B, T, -1), inputs_style_embedding.unsqueeze(1).expand(B, T, -1), inputs_content_embedding.unsqueeze(1).expand(B, T, -1)], dim=-1)
        x = self.embed_projection1(x)

        if mel_targets is not None:
            log_p_attn = self.alignment_module(text=x, feats=mel_targets.transpose(1,2), text_lengths=input_lengths, feats_lengths=output_lengths, x_masks=src_mask)
            ds, bin_loss = viterbi_decode(log_p_attn, input_lengths, output_lengths)

            ps = average_by_duration(ds, pitch_targets.squeeze(-1), input_lengths, output_lengths)
            es = average_by_duration(ds, energy_targets.squeeze(-1), input_lengths, output_lengths)

        p_outs = self.pitch_predictor(x, src_mask.unsqueeze(-1))
        e_outs = self.energy_predictor(x, src_mask.unsqueeze(-1))

        if mel_targets is not None:
            d_outs = self.duration_predictor(x, src_mask.unsqueeze(-1))
            p_embs = self.pitch_embed(ps.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(es.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)

        else:
            log_p_attn, ds, bin_loss, ps, es =None, None, None, None, None
            d_outs = self.duration_predictor.inference(x, src_mask.unsqueeze(-1))
            p_embs = self.pitch_embed(p_outs.unsqueeze(1)).transpose(1, 2)
            e_embs = self.energy_embed(e_outs.unsqueeze(1)).transpose(1, 2)

        x = x + p_embs + e_embs

        if mel_targets is not None:
            h_masks_upsampling = self.make_non_pad_mask(output_lengths).to(x.device) 
            x = self.length_regulator(x, ds, h_masks_upsampling, ~src_mask, alpha=alpha)
            h_masks = self.make_non_pad_mask(output_lengths).unsqueeze(-2).to(x.device)

        else:
            x = self.length_regulator(x, d_outs, None, ~src_mask)
            mel_lenghs = torch.sum(d_outs, dim=-1).int()
            # h_masks=make_non_pad_mask(mel_lenghs).unsqueeze(-2).to(x.device)
            h_masks=None
        x, _ = self.decoder(x, h_masks)
        x = self.to_mel(x)
        
        return {
            "mel_targets":mel_targets,
            "dec_outputs": x, 
            "postnet_outputs": None, 
            "pitch_predictions": p_outs.squeeze(),
            "pitch_targets": ps,
            "energy_predictions": e_outs.squeeze(),
            "energy_targets": es,
            "log_duration_predictions": d_outs,
            "duration_targets": ds,
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
            "log_p_attn": log_p_attn,
            "bin_loss": bin_loss,
        }
    def get_mask_from_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_len = torch.max(lengths).item()
        ids = (
            torch.arange(0, max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask
    def average_utterance_prosody(
        self, u_prosody_pred: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        lengths = ((~src_mask) * 1.0).sum(1)
        u_prosody_pred = u_prosody_pred.sum(1, keepdim=True) / lengths.view(-1, 1, 1)
        return u_prosody_pred
    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(f"{name} is not loaded")

    def make_pad_mask(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).int()

        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(
            batch_size, -1)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask


    def make_non_pad_mask(self, length, max_len=None):
        return ~self.make_pad_mask(length, max_len)


