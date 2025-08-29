import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from einops import rearrange

class ConvLinear:
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv(x)
        x = rearrange(x, 'b c t  -> b t c')
        x = self.proj(x)
        return x

class F0Discriminator2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.convs = nn.ModuleList([
            spectral_norm(ConvLinear(hidden_size, hidden_size, 3, 1)),
            spectral_norm(ConvLinear(hidden_size, hidden_size, 7, 3)),
            spectral_norm(ConvLinear(hidden_size, hidden_size, 15, 7)),
        ])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, 4, 1024, activation="gelu", batch_first=True
            ),
            4,
        )
        self.final_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, x, x_mask):
        for conv in self.convs:
            x = x + conv(x)
            x = F.silu(x)
            x = F.layer_norm(x, x.shape[2:])
        # TODO add positional encoding
        x = self.encoder(x, src_key_padding_mask=~x_mask)
        x = self.final_proj(x)
        return x
