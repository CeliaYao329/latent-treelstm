# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn


class LstmRnn(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.i_dim = input_dim
        self.h_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.h0 = nn.Parameter(torch.empty(size=(1, hidden_dim), dtype=torch.float32))
        self.c0 = nn.Parameter(torch.empty(size=(1, hidden_dim), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.h0, val=0)
        nn.init.constant_(self.c0, val=0)

        nn.init.xavier_uniform_(self.lstm.weight_ih)
        nn.init.orthogonal_(self.lstm.weight_hh)
        nn.init.constant_(self.lstm.bias_ih, val=0)
        nn.init.constant_(self.lstm.bias_hh, val=0)

    def forward(self, x, mask, backward=False):
        # x: [B, L, word_dim]
        # mask: [B, L]
        L = x.shape[1] # length of the sequence
        prev_h = self.h0.expand(x.shape[0], -1)
        prev_c = self.c0.expand(x.shape[0], -1)
        # prev_h [B, hidden_dim]
        # prev_c [B, hidden_dim]
        h = []
        for idx in range(L):
            idx = L - 1 - idx if backward else idx
            mask_idx = mask[:, idx, None]
            # mask_idx: the idx column but with size[B, 1] not [B]
            h_idx, c_idx = self.lstm(x[:, idx], (prev_h, prev_c))
            # TODO not 100% sure of the use of mask and its relation with h/c
            prev_h = h_idx * mask_idx + prev_h * (1. - mask_idx)
            prev_c = c_idx * mask_idx + prev_c * (1. - mask_idx)
            # prev_h [B, hidden_dim]
            h.append(prev_h)
        # return [B, L, hidden_dim]
        return torch.stack(h[::-1] if backward else h, dim=1)
