import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SAEENDSpkEmbed(nn.Module):
    def __init__(
        self,
        n_speakers,
        in_size,
        n_heads,
        n_units,
        n_layers,
        dim_feedforward=2048,
        dropout=0.5,
        has_pos=False,
        spk_emb_dim=256,
    ):
        """SA-EEND Self-attention based diarization end-to-end model.

        Args:
          n_speakers (int): Number of speakers in recording
          in_size (int): Dimension of input feature vector
          n_heads (int): Number of attention heads
          n_units (int): Number of units in a self-attention block
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
        """
        super(SAEENDSpkEmbed, self).__init__()
        self.n_speakers = n_speakers
        self.in_size = in_size
        self.n_heads = n_heads
        self.n_units = n_units
        self.n_layers = n_layers
        self.has_pos = has_pos

        self.src_mask = None
        self.encoder = nn.Linear(in_size, n_units)
        self.encoder_norm = nn.LayerNorm(n_units)
        if self.has_pos:
            self.pos_encoder = PositionalEncoding(n_units, dropout)
        encoder_layers = TransformerEncoderLayer(n_units, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        # speaker embeddings extraction layers
        for i in range(n_speakers):
            setattr(self, "{}{:d}".format("linear", i), nn.Linear(n_units, spk_emb_dim))

        self.decoder = nn.Linear(n_units + n_speakers * spk_emb_dim, n_speakers)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, activation=None):
        # Since src is pre-padded, the following code is extra,
        # but necessary for reproducibility
        src = nn.utils.rnn.pad_sequence(src, padding_value=-1, batch_first=True)
        # src: (B, T, E)
        src = self.encoder(src)
        src = self.encoder_norm(src)
        # src: (T, B, E)
        src = src.transpose(0, 1)
        if self.has_pos:
            # src: (T, B, E)
            src = self.pos_encoder(src)
        # src: (T, B, E)
        src = self.transformer_encoder(src, self.src_mask)
        # src: (B, T, E)
        src = src.transpose(0, 1)

        # extract and concat speaker embs
        # src: (B, T, E+2*spk_emb)
        spk_embs = []
        for i in range(self.n_speakers):
            spk_emb = getattr(self, "{}{:d}".format("linear", i))(src)
            spk_embs.append(spk_emb)

        for spk_emb in spk_embs:
            src = torch.concat([src, spk_emb], dim=-1)

        # output: (B, T, C)
        output = self.decoder(src)

        if activation:
            output = activation(output)
        return output


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the
    sequence. The positional encodings have the same dimension as the embeddings, so that the two
    can be summed. Here, we use sine and cosine functions of different frequencies.

    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add positional information to each time step of x
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
