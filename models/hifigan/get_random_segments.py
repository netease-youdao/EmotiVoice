""" 
from https://github.com/espnet/espnet 
"""

import torch


def get_random_segments( x: torch.Tensor, x_lengths: torch.Tensor, segment_size: int):
    b, d, t = x.size()
    max_start_idx = x_lengths - segment_size
    max_start_idx = torch.clamp(max_start_idx, min=0)
    start_idxs = (torch.rand([b]).to(x.device) * max_start_idx).to(
        dtype=torch.long,
    )
    segments = get_segments(x, start_idxs, segment_size)
    return segments, start_idxs, segment_size


def get_segments( x: torch.Tensor, start_idxs: torch.Tensor, segment_size: int):
    b, c, t = x.size()
    segments = x.new_zeros(b, c, segment_size)
    if t < segment_size:
        x = torch.nn.functional.pad(x, (0, segment_size - t), 'constant')
    for i, start_idx in enumerate(start_idxs):
        segment = x[i, :, start_idx : start_idx + segment_size]
        segments[i,:,:segment.size(1)] = segment
    return segments
