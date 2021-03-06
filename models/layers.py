__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['E', 'D', 'C', 'HighwayBlock', 'GatedConvBlock', 'ResidualBlock']

import torch
import torch.nn as nn
import torch.nn.functional as F

from hparams import HParams as hp
import sys


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """Layer Norm."""
        super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # PyTorch LayerNorm seems to be expect (B, T, C)
        y = super(LayerNorm, self).forward(x)
        y = y.permute(0, 2, 1)  # reverse
        return y


class D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, weight_init='none', normalization='weight', nonlinearity='linear'):
        """1D Deconvolution."""
        super(D, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                         stride=2,  # paper: stride of deconvolution is always 2
                                         dilation=dilation)

        if normalization == 'weight':
            self.deconv = nn.utils.weight_norm(self.deconv)
        elif normalization == 'layer':
            self.layer_norm = LayerNorm(out_channels)

        self.nonlinearity = nonlinearity
        if weight_init == 'kaiming':
            nn.init.kaiming_normal_(self.deconv.weight, mode='fan_out', nonlinearity=nonlinearity)
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.deconv.weight, nn.init.calculate_gain(nonlinearity))

    def forward(self, inp, output_size=None):
        x = inp[0]
        speaker_codes = inp[1]
        y = self.deconv(x, output_size=output_size)
        if hasattr(self, 'layer_norm'):
            y = self.layer_norm(y)
        y = F.dropout(y, p=hp.dropout_rate, training=self.training, inplace=True)
        if self.nonlinearity == 'relu':
            y = F.relu(y, inplace=True)
        return (y, speaker_codes)


class LCC(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(LCC, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, x):
        inp = x[0]

        speaker_codes = x[1]
        lcc_gate = self.embedding(speaker_codes)
        lcc_gate = torch.sigmoid(lcc_gate)  # -> 0.5 after sigmoid
        lcc_gate = lcc_gate.permute(0, 2, 1)

        inp = lcc_gate * inp

        return inp


class C(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, use_speaker_code=True, causal=False, weight_init='none', normalization='weight', nonlinearity='linear'):
        """1D convolution.
        The argument 'causal' indicates whether the causal convolution should be used or not.
        """
        super(C, self).__init__()
        self.causal = causal
        if causal:
            self.padding = (kernel_size - 1) * dilation
        else:
            self.padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=1,  # paper: 'The stride of convolution is always 1.'
                              padding=self.padding, dilation=dilation)

        if normalization == 'weight':
            self.conv = nn.utils.weight_norm(self.conv)
        elif normalization == 'layer':
            self.layer_norm = LayerNorm(out_channels)

        self.nonlinearity = nonlinearity
        if weight_init == 'kaiming':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity=nonlinearity)
        elif weight_init == 'xavier':
            nn.init.xavier_uniform_(self.conv.weight, nn.init.calculate_gain(nonlinearity))

        self.use_speaker_code = use_speaker_code
        self.nspeakers = 0
        if 'learn_channel_contributions' in hp.multispeaker:
            self.lcc = LCC(hp.nspeakers, out_channels) # hp.speaker_embedding_size
            self.nspeakers = hp.nspeakers

    def forward(self, inp):

        # as input we receive a tuple containing x at position 0 and the speaker codes at position 1
        x = inp[0]
        speaker_codes = inp[1]

        y = self.conv(x)
        padding = self.padding
        if self.causal and padding > 0:

            y = y[:, :, :-padding]

        if hasattr(self, 'layer_norm'):
            y = self.layer_norm(y)
        y = F.dropout(y, p=hp.dropout_rate, training=self.training, inplace=True)
        if self.nonlinearity == 'relu':
            y = F.relu(y, inplace=True)

        if self.nspeakers > 0 and self.use_speaker_code:
            y = self.lcc((y, speaker_codes))

        # both the y output and speaker code needs to be returned as a tuple
        return (y, speaker_codes)


class E(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(E, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)


class HighwayBlock(nn.Module):
    def __init__(self, d, k, delta, causal=False, weight_init='none', normalization='weight'):
        """Highway Network like layer: https://arxiv.org/abs/1505.00387
        The input and output shapes remain same.
        Args:
            d: input channel
            k: kernel size
            delta: dilation
            causal: causal convolution or not
        """
        super(HighwayBlock, self).__init__()
        self.d = d
        self.C = C(in_channels=d, out_channels=2 * d, kernel_size=k, dilation=delta, causal=causal, weight_init=weight_init, normalization=normalization)

        self.nspeakers = 0
        if 'learn_channel_contributions' in hp.multispeaker:
            self.lcc = LCC(hp.nspeakers, d)  # hp.speaker_embedding_size
            self.nspeakers = hp.nspeakers

    def forward(self, inp):
        x = inp[0]
        speaker_codes = inp[1]
        L, _ = self.C(inp)
        H1 = L[:, :self.d, :]
        H2 = L[:, self.d:, :]
        sigH1 = torch.sigmoid(H1)

        if self.nspeakers > 0:
            H2 = self.lcc((H2, speaker_codes))

        return (sigH1 * H2 + (1 - sigH1) * x, speaker_codes)


class GatedConvBlock(nn.Module):
    def __init__(self, d, k, delta, causal=False, weight_init='none', normalization='weight'):
        """Gated convolutional layer: https://arxiv.org/abs/1612.08083
        The input and output shapes remain same.
        Args:
            d: input channel
            k: kernel size
            delta: dilation
            causal: causal convolution or not
        """
        super(GatedConvBlock, self).__init__()
        self.C = C(in_channels=d, out_channels=2 * d, kernel_size=k, dilation=delta, causal=causal,
                   weight_init=weight_init, normalization=normalization)
        self.glu = nn.GLU(dim=1)

    def forward(self, inp):
        x = inp[0]
        speaker_codes = inp[1]
        L, _ = self.C(inp)
        return (self.glu(L) + x, speaker_codes)


class ResidualBlock(nn.Module):
    def __init__(self, d, k, delta, causal=False, weight_init='none', normalization='weight',
                 widening_factor=2):
        """Residual block: https://arxiv.org/abs/1512.03385
        The input and output shapes remain same.
        Args:
            d: input channel
            k: kernel size
            delta: dilation
            causal: causal convolution or not
        """
        super(ResidualBlock, self).__init__()
        self.C1 = C(in_channels=d, out_channels=widening_factor * d, kernel_size=k, dilation=delta, causal=causal,
                    weight_init=weight_init, normalization=normalization, nonlinearity='relu')
        self.C2 = C(in_channels=widening_factor * d, out_channels=d, kernel_size=k, dilation=delta, causal=causal,
                    weight_init=weight_init, normalization=normalization, nonlinearity='relu')

    def forward(self, x):
        return self.C2(self.C1(x)) + x
