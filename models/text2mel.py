"""
Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara
Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention
https://arxiv.org/abs/1710.08969

Text2Mel Network.
"""
__author__ = 'Erdene-Ochir Tuguldur'
__all__ = ['Text2Mel']

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hparams import HParams as hp
from .layers import E, C, HighwayBlock, GatedConvBlock, ResidualBlock


def Conv(in_channels, out_channels, kernel_size, dilation, use_speaker_code=True, causal=False, nonlinearity='linear'):
    return C(in_channels, out_channels, kernel_size, dilation, use_speaker_code, causal=causal,
             weight_init=hp.text2mel_weight_init, normalization=hp.text2mel_normalization, nonlinearity=nonlinearity)


def BasicBlock(d, k, delta, causal=False):
    if hp.text2mel_basic_block == 'gated_conv':
        return GatedConvBlock(d, k, delta, causal=causal,
                              weight_init=hp.text2mel_weight_init, normalization=hp.text2mel_normalization)
    elif hp.text2mel_basic_block == 'highway':
        return HighwayBlock(d, k, delta, causal=causal,
                            weight_init=hp.text2mel_weight_init, normalization=hp.text2mel_normalization)
    else:
        return ResidualBlock(d, k, delta, causal=causal,
                             weight_init=hp.text2mel_weight_init, normalization=hp.text2mel_normalization,
                             widening_factor=2)


def CausalConv(in_channels, out_channels, kernel_size, dilation, nonlinearity='linear', use_speaker_code=True):
    return Conv(in_channels, out_channels, kernel_size, dilation, use_speaker_code, causal=True, nonlinearity=nonlinearity)


def CausalBasicBlock(d, k, delta):
    return BasicBlock(d, k, delta, causal=True)


class TextEnc(nn.Module):

    def __init__(self, vocab, e=hp.e, d=hp.d):
        """Text encoder network.
        Args:
            vocab: vocabulary
            e: embedding dim
            d: Text2Mel dim
        Input:
            L: (B, N) text inputs
        Outputs:
            K: (B, d, N) keys
            V: (N, d, N) values
        """
        super(TextEnc, self).__init__()
        self.d = d
        self.embedding = E(len(vocab), e)
        self.e = e
        if 'text_encoder_input' in hp.multispeaker:
            self.speaker_embedding = E(hp.nspeakers, hp.speaker_embedding_size)
            # the speaker embedding size needs to be added, and this will be used as input for the first conv layer
            self.e = e + hp.speaker_embedding_size

        self.layers = nn.Sequential(
            Conv(self.e, 2 * d, 1, 1, nonlinearity='relu'),

            Conv(2 * d, 2 * d, 1, 1),

            BasicBlock(2 * d, 3, 1), BasicBlock(2 * d, 3, 3), BasicBlock(2 * d, 3, 9), BasicBlock(2 * d, 3, 27),
            BasicBlock(2 * d, 3, 1), BasicBlock(2 * d, 3, 3), BasicBlock(2 * d, 3, 9), BasicBlock(2 * d, 3, 27),

            BasicBlock(2 * d, 3, 1), BasicBlock(2 * d, 3, 1),

            # needs to be moved outside because of text_encoder_towards_end
            # BasicBlock(2 * d, 1, 1), BasicBlock(2 * d, 1, 1)
        )

        self.d2 = 2 * d
        if 'text_encoder_towards_end' in hp.multispeaker:
            self.d2 = 2 * d + hp.speaker_embedding_size
            self.speaker_conv = Conv(self.d2, self.d2, 1, 1)

        # 2 * d changed to d2 because if the size need to be increased with the speaker embedding
        self.BasicBlock_Penultimate = BasicBlock(self.d2, 1, 1)
        self.BasicBlock_Last = BasicBlock(self.d2, 1, 1)

    def forward(self, x):
        text = x[0]
        speaker_codes = x[1]
        out = self.embedding(text)

        if 'text_encoder_input' in hp.multispeaker:
            speaker_codes_time = torch.repeat_interleave(speaker_codes, text.shape[1], 1)
            speaker_reps = self.speaker_embedding(speaker_codes_time)
            out = torch.cat((out, speaker_reps), -1)

        # simple solution just to add embedding for speakers and then concat to out
        out = out.permute(0, 2, 1)  # change to (B, e, N)
        out = self.layers((out, speaker_codes))  # (B, 2*d, N)

        if 'text_encoder_towards_end' in hp.multispeaker:
            out = out[0]
            speaker_codes_time = torch.repeat_interleave(speaker_codes, text.shape[1], 1)
            speaker_reps = self.speaker_embedding(speaker_codes_time)
            speaker_reps = speaker_reps.permute(0, 2, 1)
            out = torch.cat((out, speaker_reps), 1)
            out = self.speaker_conv((out, speaker_codes))

        out = self.BasicBlock_Penultimate(out)
        out = self.BasicBlock_Last(out)

        out = out[0]
        K = out[:, :self.d, :]  # (B, d, N)
        V = out[:, self.d:, :]  # (B, d, N)
        return K, V


class AudioEnc(nn.Module):
    def __init__(self, d=hp.d, f=hp.n_mels):
        """Audio encoder network.
        Args:
            d: Text2Mel dim
            f: Number of mel bins
        Input:
            S: (B, f, T) melspectrograms
        Output:
            Q: (B, d, T) queries
        """
        super(AudioEnc, self).__init__()

        self.CausalConv = CausalConv(f, d, 1, 1, nonlinearity='relu')
        self.d = d
        if 'audio_encoder_input' in hp.multispeaker:
            self.speaker_embedding = E(hp.nspeakers, hp.speaker_embedding_size)
            # the input size needs to be increased with the speaker embedding size
            self.d = d + hp.speaker_embedding_size

        self.layers = nn.Sequential(
            # this is moved to be in front of the embedding layer
            # CausalConv(f, d, 1, 1, nonlinearity='relu'),

            CausalConv(self.d, d, 1, 1, nonlinearity='relu'),
            CausalConv(d, d, 1, 1),

            CausalBasicBlock(d, 3, 1), CausalBasicBlock(d, 3, 3), CausalBasicBlock(d, 3, 9), CausalBasicBlock(d, 3, 27),
            CausalBasicBlock(d, 3, 1), CausalBasicBlock(d, 3, 3), CausalBasicBlock(d, 3, 9), CausalBasicBlock(d, 3, 27),

            CausalBasicBlock(d, 3, 3), CausalBasicBlock(d, 3, 3),
        )

    def forward(self, x):

        x = self.CausalConv(x)

        out = x[0]
        speaker_codes = x[1]

        if 'audio_encoder_input' in hp.multispeaker:
            speaker_codes_time = torch.repeat_interleave(speaker_codes, out.shape[2], 1)
            speaker_reps = self.speaker_embedding(speaker_codes_time)
            speaker_reps = speaker_reps.permute(0, 2, 1)
            out = torch.cat((out, speaker_reps), 1)

        return self.layers((out,speaker_codes))

        # return self.layers(x)


class AudioDec(nn.Module):
    def __init__(self, d=hp.d, f=hp.n_mels):
        """Audio decoder network.
        Args:
            d: Text2Mel dim
            f: Number of mel bins
        Input:
            R_prime: (B, 2d, T) [V*Attention, Q] paper says: "we found it beneficial in our pilot study."
        Output:
            Y: (B, f, T)
        """
        super(AudioDec, self).__init__()

        self.d2 = 2 * d
        # if this option is used the speaker embedding size needs to be added, as the input received
        # will contain the speaker embedding concatenated
        if 'text_encoder_towards_end' in hp.multispeaker:
            self.d2 = 2 * d + hp.speaker_embedding_size

        self.CausalConv = CausalConv(self.d2, d, 1, 1, use_speaker_code=False)

        self.d = d
        if 'audio_decoder_input' in hp.multispeaker:
            self.speaker_embedding = E(hp.nspeakers, hp.speaker_embedding_size)
            self.d = d + hp.speaker_embedding_size

        self.layers = nn.Sequential(
            # this is moved outside of layers as the speaker embedding layer is inserted after
            # the first CausalConv layer
            # CausalConv(2 * d, d, 1, 1, use_speaker_code=False),

            CausalBasicBlock(self.d, 3, 1), CausalBasicBlock(self.d, 3, 3), CausalBasicBlock(self.d, 3, 9), CausalBasicBlock(self.d, 3, 27),

            CausalBasicBlock(self.d, 3, 1), CausalBasicBlock(self.d, 3, 1),

            CausalBasicBlock(self.d, 1, 1),
            CausalConv(self.d, self.d, 1, 1, nonlinearity='relu'),

            CausalConv(self.d, f, 1, 1)
        )

    def forward(self, x):

        x = self.CausalConv(x)

        out = x[0]
        speaker_codes = x[1]

        if 'audio_decoder_input' in hp.multispeaker:
            speaker_codes_time = torch.repeat_interleave(speaker_codes, out.shape[2], 1)
            speaker_reps = self.speaker_embedding(speaker_codes_time)
            speaker_reps = speaker_reps.permute(0, 2, 1)
            out = torch.cat((out, speaker_reps), 1)

        return self.layers((out, speaker_codes))

        # return self.layers(x)


class Text2Mel(nn.Module):
    def __init__(self, vocab, d=hp.d):
        """Text to melspectrogram network.
        Args:
            vocab: vocabulary
            d: Text2Mel dim
        Input:
            L: (B, N) text inputs
            S: (B, f, T) melspectrograms
        Outputs:
            Y_logit: logit of Y
            Y: predicted melspectrograms
            A: (B, N, T) attention matrix
        """
        super(Text2Mel, self).__init__()
        self.d = d
        self.text_enc = TextEnc(vocab)
        self.audio_enc = AudioEnc()
        self.audio_dec = AudioDec()

    def forward(self, L, S, speaker_codes, monotonic_attention=False):

        K, V = self.text_enc((L, speaker_codes))
        Q, _ = self.audio_enc((S, speaker_codes))
        A = torch.bmm(K.permute(0, 2, 1), Q) / np.sqrt(self.d)

        if monotonic_attention:
            # TODO: vectorize instead of loops
            B, N, T = A.size()
            for i in range(B):
                prva = -1  # previous attention
                for t in range(T):
                    _, n = torch.max(A[i, :, t], 0)
                    # print (n, prva, n-prva)
                    if not (-1 <= n - prva <= 3) and 0:
                        A[i, :, t] = -2 ** 20  # some small numbers
                        A[i, min(N - 1, prva + 1), t] = 1
                    _, prva = torch.max(A[i, :, t], 0)

        A = F.softmax(A, dim=1)
        R = torch.bmm(V, A)
        R_prime = torch.cat((R, Q), 1)
        Y_logit, _ = self.audio_dec((R_prime, speaker_codes))
        Y = torch.sigmoid(Y_logit)
        return Y_logit, Y, A
