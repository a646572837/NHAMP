import torch.nn as nn
import numpy as np
import math
from einops import rearrange
import torch
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        # make classifier free guidance
        self.num_classes = num_classes 
        self.dropout_prob = dropout_prob
        # Make an extra to unconditon
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # Drop samples make it uncond
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class Denoiser(nn.Module):
    def __init__(self, esm_model, input_dim=320, mlp_dims=[320, 1024, 320], dropout_prob=0.1):
        super().__init__()
        self.denoiser_embedding = input_dim
        self.denoiser_mlp = mlp_dims
        
        # Timestep 
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(self.denoiser_embedding * 2),
            nn.Linear(self.denoiser_embedding * 2, self.denoiser_embedding * 4),
            nn.SiLU(),
            nn.Linear(self.denoiser_embedding * 4, self.denoiser_embedding * 4),
        )
        
        #label(0,1,2 to the correspond label,3 to the uncond)
        self.label_emb = LabelEmbedder(num_classes=3, hidden_size=self.denoiser_embedding * 4, dropout_prob=dropout_prob)

        esm_attention_list = []
        for layer in esm_model.encoder.layer:
            esm_attention_list.append(layer.attention)
        self.esm_attention_list = nn.ModuleList(esm_attention_list)

        time_emb_proj_list = []
        for _ in self.esm_attention_list:
            time_emb_proj_list.append(nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.denoiser_embedding * 4, self.denoiser_embedding * 2)
            ))
        self.time_emb_proj_list = nn.ModuleList(time_emb_proj_list)

        self.norm_before = nn.LayerNorm(self.denoiser_embedding, eps=1e-6)
        self.norm_after = nn.LayerNorm(self.denoiser_embedding, eps=1e-6)

        # MLP output
        mlp_list = [
            nn.Linear(self.denoiser_embedding, self.denoiser_mlp[0]),
            nn.LayerNorm(self.denoiser_mlp[0], eps=1e-6),
            nn.GELU(),
        ]
        for index in range(len(self.denoiser_mlp) - 1):
            mlp_list.append(nn.Linear(self.denoiser_mlp[index], self.denoiser_mlp[index + 1]))
            mlp_list.append(nn.LayerNorm(self.denoiser_mlp[index + 1], eps=1e-6))
            mlp_list.append(nn.GELU())
        mlp_list.append(nn.Linear(self.denoiser_mlp[-1], self.denoiser_embedding))
        self.mlp = nn.Sequential(*mlp_list)

    def forward(self, x, time, y, attention_mask=None):
        # Attention Mask 
        if attention_mask is not None:
            # [Batch, 1, 1, Seq_Len]
            attention_mask = attention_mask[:, None, None, :].expand(attention_mask.shape[0], 1, attention_mask.shape[1], attention_mask.shape[1])
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Embedding
        time_emb = self.time_emb(time)
        label_emb = self.label_emb(y, self.training)
        c = torch.add(time_emb, label_emb)

        x = self.norm_before(x)
        attn_output = x.clone()

        for time_emb_layer, attention_layer in zip(self.time_emb_proj_list, self.esm_attention_list):
            mlp_c = time_emb_layer(c)
            mlp_c = rearrange(mlp_c, "b c -> b 1 c")
            scale, shift = mlp_c.chunk(2, dim=-1)

            # Scale & Shift 
            attn_output = torch.add(attn_output * (scale + 1.), shift)

            # Attention 
            layer_outputs = attention_layer(attn_output, attention_mask=attention_mask, output_attentions=False)
            attn_output = layer_outputs[0]

        # ff & output
        x = self.norm_after(torch.add(x, attn_output))
        x = self.mlp(x)
        return x