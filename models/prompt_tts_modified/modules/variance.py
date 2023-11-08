"""
This code is modified from https://github.com/espnet/espnet.
"""

import torch 

from models.prompt_tts_modified.modules.encoder import LayerNorm

class DurationPredictor(torch.nn.Module):

    def __init__(
        self, idim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0
    ):

        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        # NOTE: calculate in log domain
        xs = self.linear(xs.transpose(1, -1))  # (B, Tmax)
        if is_inference:
            # NOTE: calculate in linear domain
            xs = torch.clamp(
                torch.round(xs.exp() - self.offset), min=0
            ).long()  # avoid negative value

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs.squeeze(-1)

    def forward(self, xs, x_masks=None):

        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):

        return self._forward(xs, x_masks, True)



class VariancePredictor(torch.nn.Module):


    def __init__(
        self,
        idim: int,
        n_layers: int = 2,
        n_chans: int = 384,
        kernel_size: int = 3,
        bias: bool = True,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs: torch.Tensor, x_masks: torch.Tensor = None) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor): Batch of masks indicating padded part (B, Tmax).

        Returns:
            Tensor: Batch of predicted sequences (B, Tmax, 1).

        """
        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        xs = xs.transpose(1, -1)  # (B, idim, Tmax)
        for f in self.conv:
            xs = f(xs)  # (B, C, Tmax)

        xs = self.linear(xs.transpose(1, 2))  # (B, Tmax, 1)

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs.squeeze(-1)