import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn
import torchtune
import torch.optim as optim

class GCSTransformer(nn.Module):
    def __init__(self, num_time_bins = 512, num_space_bins = 1024, d_model = 128*6, nhead = 8, num_layers = 12, num_map_tokens = 36, num_channels = 6):
        super().__init__()

        self.num_time_bins = num_time_bins
        self.num_space_bins = num_space_bins
        self.num_map_tokens = num_map_tokens
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.embedding_dim = self.d_model // self.num_channels

        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // self.nhead

        
        self.map_embs = nn.Embedding(self.num_map_tokens, self.embedding_dim)
        self.traj_embs = nn.Embedding(num_time_bins + num_space_bins + 3, # Plus 3 for start/end of traj, end of seq, and pad tokens
                                        self.d_model, padding_idx = num_time_bins + num_space_bins + 2)
        
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nhead, 
                                          num_encoder_layers=self.num_layers, num_decoder_layers=self.num_layers, 
                                          dim_feedforward=2048, batch_first=True)
        self.rope = torchtune.modules.RotaryPositionalEmbeddings(self.d_head)

        self.linear = nn.Linear(self.d_model, self.traj_embs.num_embeddings)

    
    def forward(self, map_array, trajs):
        assert map_array.shape[0] == trajs.shape[0]

        # Assume map_array has structure:
        # [B, H * W, Channels]        
        batch_size, map_seq_len, channels = map_array.shape
        map_embs = self.map_embs(map_array).reshape(batch_size, map_seq_len, -1)

        batch_size, traj_seq_len = trajs.shape
        traj_embs = self.traj_embs(trajs)

        # Apply rope after concatenating map and traj embs, then split again.
        all_embs = torch.cat([map_embs, traj_embs], dim=1).reshape(batch_size, map_seq_len + traj_seq_len, self.nhead, self.d_head)
        all_embs = self.rope(all_embs).reshape(batch_size, map_seq_len + traj_seq_len, -1)

        map_embs = all_embs[:, :map_seq_len]
        traj_embs = all_embs[:, map_seq_len:]

        # Pass through transformer
        # [batch_size, traj_seq_len, d_model]
        output = self.transformer(src = map_embs, tgt = traj_embs,
                                  tgt_mask = nn.Transformer.generate_square_subsequent_mask(traj_seq_len, device=map_array.device),
                                  src_is_causal = False,
                                  tgt_is_causal = True)
        
        logits = self.linear(output)

        return logits