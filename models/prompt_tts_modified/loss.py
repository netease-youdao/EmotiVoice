"""
This code is modified from https://github.com/alibaba-damo-academy/KAN-TTS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = (
        torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    )
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

class MelReconLoss(torch.nn.Module):
    def __init__(self, loss_type="mae"):
        super(MelReconLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "mae":
            self.criterion = torch.nn.L1Loss(reduction="none")
        elif loss_type == "mse":
            self.criterion = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError("Unknown loss type: {}".format(loss_type))

    def forward(self, output_lengths, mel_targets, dec_outputs, postnet_outputs=None):
        """
        mel_targets: B, C, T
        """
        output_masks = get_mask_from_lengths(
            output_lengths, max_len=mel_targets.size(1)
        )
        output_masks = ~output_masks
        valid_outputs = output_masks.sum()

        mel_loss_ = torch.sum(
            self.criterion(mel_targets, dec_outputs) * output_masks.unsqueeze(-1)
        ) / (valid_outputs * mel_targets.size(-1))

        if postnet_outputs is not None:
            mel_loss = torch.sum(
                self.criterion(mel_targets, postnet_outputs)
                * output_masks.unsqueeze(-1)
            ) / (valid_outputs * mel_targets.size(-1))
        else:
            mel_loss = 0.0

        return mel_loss_, mel_loss



class ForwardSumLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        log_p_attn: torch.Tensor,
        ilens: torch.Tensor,
        olens: torch.Tensor,
        blank_prob: float = np.e**-1,
    ) -> torch.Tensor:
        B = log_p_attn.size(0)

        # a row must be added to the attention matrix to account for
        #    blank token of CTC loss
        # (B,T_feats,T_text+1)
        log_p_attn_pd = F.pad(log_p_attn, (1, 0, 0, 0, 0, 0), value=np.log(blank_prob))

        loss = 0
        for bidx in range(B):
            # construct target sequnece.
            # Every text token is mapped to a unique sequnece number.
            target_seq = torch.arange(1, ilens[bidx] + 1).unsqueeze(0)
            cur_log_p_attn_pd = log_p_attn_pd[
                bidx, : olens[bidx], : ilens[bidx] + 1
            ].unsqueeze(
                1
            )  # (T_feats,1,T_text+1)
            cur_log_p_attn_pd = F.log_softmax(cur_log_p_attn_pd, dim=-1)
            loss += F.ctc_loss(
                log_probs=cur_log_p_attn_pd,
                targets=target_seq,
                input_lengths=olens[bidx : bidx + 1],
                target_lengths=ilens[bidx : bidx + 1],
                zero_infinity=True,
            )
        loss = loss / B
        return loss

class ProsodyReconLoss(torch.nn.Module):
    def __init__(self, loss_type="mae"):
        super(ProsodyReconLoss, self).__init__()
        self.loss_type = loss_type
        if loss_type == "mae":
            self.criterion = torch.nn.L1Loss(reduction="none")
        elif loss_type == "mse":
            self.criterion = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError("Unknown loss type: {}".format(loss_type))

    def forward(
        self,
        input_lengths,
        duration_targets,
        pitch_targets,
        energy_targets,
        log_duration_predictions,
        pitch_predictions,
        energy_predictions,
    ):
        input_masks = get_mask_from_lengths(
            input_lengths, max_len=duration_targets.size(1)
        )
        input_masks = ~input_masks
        valid_inputs = input_masks.sum()

        dur_loss = (
            torch.sum(
                self.criterion(
                    torch.log(duration_targets.float() + 1), log_duration_predictions
                )
                * input_masks
            )
            / valid_inputs
        )
        pitch_loss = (
            torch.sum(self.criterion(pitch_targets, pitch_predictions) * input_masks)
            / valid_inputs
        )
        energy_loss = (
            torch.sum(self.criterion(energy_targets, energy_predictions) * input_masks)
            / valid_inputs
        )

        return dur_loss, pitch_loss, energy_loss


class TTSLoss(torch.nn.Module):
    def __init__(self, loss_type="mae") -> None:
        super().__init__()
    
        self.Mel_Loss = MelReconLoss()
        self.Prosodu_Loss = ProsodyReconLoss(loss_type)
        self.ForwardSum_Loss = ForwardSumLoss()

    def forward(self, outputs):
        
        dec_outputs = outputs["dec_outputs"]
        postnet_outputs = outputs["postnet_outputs"]
        log_duration_predictions = outputs["log_duration_predictions"]
        pitch_predictions = outputs["pitch_predictions"]
        energy_predictions = outputs["energy_predictions"]
        duration_targets = outputs["duration_targets"]
        pitch_targets = outputs["pitch_targets"]
        energy_targets = outputs["energy_targets"]
        output_lengths = outputs["output_lengths"]
        input_lengths = outputs["input_lengths"]
        mel_targets = outputs["mel_targets"].transpose(1,2)
        log_p_attn = outputs["log_p_attn"]
        bin_loss = outputs["bin_loss"]
        
        dec_mel_loss, postnet_mel_loss = self.Mel_Loss(output_lengths, mel_targets, dec_outputs, postnet_outputs)
        dur_loss, pitch_loss, energy_loss = self.Prosodu_Loss(input_lengths, duration_targets, pitch_targets, energy_targets, log_duration_predictions, pitch_predictions, energy_predictions)
        forwardsum_loss = self.ForwardSum_Loss(log_p_attn, input_lengths, output_lengths)
        
        res = {
            "dec_mel_loss": dec_mel_loss,
            "postnet_mel_loss": postnet_mel_loss,
            "dur_loss": dur_loss,
            "pitch_loss": pitch_loss,
            "energy_loss": energy_loss,
            "forwardsum_loss": forwardsum_loss,
            "bin_loss": bin_loss,
        }
        
        return res