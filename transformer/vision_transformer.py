import math

import torch
import torch.nn as nn
from torchsummary import summary


class SelfAttention(nn.Module):
    def __int__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.embed_dim).permute(1, # qkv
                                                                   0, # batch
                                                                   2, # channel
                                                                   3) # embed_dim

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.bmm(q, k.transpose(-2, -1)) * self.scale # <q, k> / sqrt(d)
        attn.softmax(dim=-1) # softmax over embedding dim
        x = torch.bmm(attn, v).transpose(1, 2).reshape(B, N, self.embed_dim)
        return x


class MultiheadedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=12, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "Embedding dim must be divisible by number of heads" # why
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, # qkv
                                                                                        0, # batch
                                                                                        3, # channel
                                                                                        1, # num_heads
                                                                                        4) # embed_dim

        q, k, v = torch.chunk(qkv, 3)

        attn = torch.bmm(q, k.transpose(-2, -1)) * self.scale
        attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = torch.bmm(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        x = self.proj_dropout(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, freq=10000.):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embedding_dim, 2).float() *
                        (-math.log(freq) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div) # even terms
        pe[:, 1::2] = torch.cos(position * div) # odd terms
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, attn_dropout=0.0, proj_dropout=0.0, mlp_dropout=0.1,
                 feedforward_dim=3072):
        super().__init__()
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.MHA = MultiheadedSelfAttention(embed_dim, num_heads, attn_dropout, proj_dropout)

        self.ff = nn.Sequential(nn.Linear(embed_dim, feedforward_dim),
                                nn.GELU(), # the Guassian Error Linear Unit
                                nn.Dropout(mlp_dropout),
                                nn.Linear(feedforward_dim, embed_dim),
                                nn.Dropout(mlp_dropout))

    def forward(self, x):
        mha = self.MHA(self.norm_1(x))
        x = x + mha # residual connection (Add)
        x = self.norm_2(x) # LayerNorm

        x2 = self.ff(x)
        x = x + x2 # residual connection (Add)
        return x


class Patching(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.patch = nn.Sequential(nn.Conv2d(in_channels, out_channels=embedding_dim,
                                             kernel_size=(patch_size, patch_size),
                                             stride=(patch_size, patch_size)),
                                   nn.Flatten(2, 3))

    def forward(self, x):
        return self.patch(x).transpose(-2, -1)


class ViT(nn.Module):
    def __init__(self,
                 img_size=224,  # input image size
                 in_channels=3, # number of channels for the image
                 patch_size=16, # how big our patches are
                 embed_dim=768, # embedding dim
                 num_layers=12, # number of transformer encoders
                 num_heads=12,  # number o attention heads
                 attn_dropout=0., # attention dropout
                 proj_dropout=0., # (MHA) projection dropout
                 mlp_dropout=0.1, # (MHA) last layer dropout
                 mlp_ratio=4,     # (MHA) projection FF dimension
                 n_classes=1000,  # number of classes to classify
                 ):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patchAndEmbed = Patching(in_channels, patch_size, embed_dim)
        sequence_length = (img_size // patch_size) ** 2
        hidden_dim = int(embed_dim * mlp_ratio)
        self.class_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        self.pe = nn.Parameter(torch.zeros(1, sequence_length + 1, embed_dim), requires_grad=True)
        self.transformerEncoder = nn.Sequential(*[TransformerEncoderLayer(embed_dim,
                                                                          num_heads,
                                                                          attn_dropout,
                                                                          proj_dropout,
                                                                          mlp_dropout,
                                                                          hidden_dim)
                                                for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = self.patchAndEmbed(x)
        class_token = self.class_embed.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)

        x += self.pe
        x = self.transformerEncoder(x)
        x = x[:, 0]
        x = self.mlp(self.norm(x))
        return x


if __name__ == "__main__":
    vit_ins = ViT()
    print(vit_ins)

    #summary(vit_ins, input_size=(3, 3, 3, 224, 224))