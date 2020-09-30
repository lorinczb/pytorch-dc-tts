"""
Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara
Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
https://arxiv.org/abs/1710.08969

SSRN Network.
"""
__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['SSRN']

import torch
import torch.nn as nn
import torch.nn.functional as F

from hparams import HParams as hp
from .layers_ssrn import LayerNorm, GatedConvBlock, ResidualBlock


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

    def forward(self, x, output_size=None):
        y = self.deconv(x, output_size=output_size)
        if hasattr(self, 'layer_norm'):
            y = self.layer_norm(y)
        y = F.dropout(y, p=hp.dropout_rate, training=self.training, inplace=True)
        if self.nonlinearity == 'relu':
            y = F.relu(y, inplace=True)
        return y


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

    def forward(self, x):
        L = self.C(x)
        H1 = L[:, :self.d, :]
        H2 = L[:, self.d:, :]
        sigH1 = F.sigmoid(H1)
        return sigH1 * H2 + (1 - sigH1) * x


class C(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, causal=False, weight_init='none', normalization='weight', nonlinearity='linear'):
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

    def forward(self, x):
        y = self.conv(x)
        padding = self.padding
        if self.causal and padding > 0:
            y = y[:, :, :-padding]

        if hasattr(self, 'layer_norm'):
            y = self.layer_norm(y)
        y = F.dropout(y, p=hp.dropout_rate, training=self.training, inplace=True)
        if self.nonlinearity == 'relu':
            y = F.relu(y, inplace=True)
        return y


def Conv(in_channels, out_channels, kernel_size, dilation, nonlinearity='linear'):
    return C(in_channels, out_channels, kernel_size, dilation, causal=False,
             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization, nonlinearity=nonlinearity)


def DeConv(in_channels, out_channels, kernel_size, dilation, nonlinearity='linear'):
    return D(in_channels, out_channels, kernel_size, dilation,
             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization, nonlinearity=nonlinearity)


def BasicBlock(d, k, delta):
    if hp.ssrn_basic_block == 'gated_conv':
        return GatedConvBlock(d, k, delta, causal=False,
                              weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization)
    elif hp.ssrn_basic_block == 'highway':
        return HighwayBlock(d, k, delta, causal=False,
                            weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization)
    else:
        return ResidualBlock(d, k, delta, causal=False,
                             weight_init=hp.ssrn_weight_init, normalization=hp.ssrn_normalization,
                             widening_factor=1)


class SSRN(nn.Module):
    def __init__(self, c=hp.c, f=hp.n_mels, f_prime=(1 + hp.n_fft // 2)):
        """Spectrogram super-resolution network.
        Args:
            c: SSRN dim
            f: Number of mel bins
            f_prime: full spectrogram dim
        Input:
            Y: (B, f, T) predicted melspectrograms
        Outputs:
            Z_logit: logit of Z
            Z: (B, f_prime, 4*T) full spectrograms
        """
        super(SSRN, self).__init__()
        self.layers = nn.Sequential(
            Conv(f, c, 1, 1),

            BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),

            DeConv(c, c, 2, 1), BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),
            DeConv(c, c, 2, 1), BasicBlock(c, 3, 1), BasicBlock(c, 3, 3),

            Conv(c, 2 * c, 1, 1),

            BasicBlock(2 * c, 3, 1), BasicBlock(2 * c, 3, 1),

            Conv(2 * c, f_prime, 1, 1),

            BasicBlock(f_prime, 1, 1),

            Conv(f_prime, f_prime, 1, 1)
        )

    def forward(self, x):
        Z_logit = self.layers(x)
        Z = torch.sigmoid(Z_logit)
        return Z_logit, Z