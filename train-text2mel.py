#!/usr/bin/env python
"""Train the Text2Mel network. See: https://arxiv.org/abs/1710.08969"""
__author__ = 'Erdene-Ochir Tuguldur'

import sys, os
import time
import argparse
from tqdm import *

import numpy as np
import torch
import torch.nn.functional as F

# project imports
from models import Text2Mel
from hparams import HParams as hp
from logger import Logger
from utils import get_last_checkpoint_file_name, load_checkpoint, load_and_transform_checkpoint, save_checkpoint
from datasets.data_loader import Text2MelDataLoader
from pretrained_voxceleb_model.ResNetSE34L import *
from pretrained_voxceleb_model.SpeakerNet import SpeakerNet
from pretrained_voxceleb_model.tuneThreshold import tuneThresholdfromScore

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, choices=['ljspeech', 'mbspeech', 'swara', 'swara_test'], help='dataset name')
args = parser.parse_args()

if args.dataset == 'ljspeech':
    from datasets.lj_speech import vocab, LJSpeech as SpeechDataset
elif args.dataset == 'mbspeech':
    from datasets.mb_speech import vocab, MBSpeech as SpeechDataset
elif args.dataset == 'swara':
    from datasets.swara import vocab, SWARA as SpeechDataset
elif args.dataset == 'swara_test':
    from datasets.swara_test import vocab, SWARA as SpeechDataset
else:
    print('No such dataset')
    sys.exit(1)

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
if use_gpu:
    torch.backends.cudnn.benchmark = True

train_data_loader = Text2MelDataLoader(text2mel_dataset=SpeechDataset(['texts', 'mels', 'mel_gates', 'speakers']),
                                       batch_size=32, mode='train')
valid_data_loader = Text2MelDataLoader(text2mel_dataset=SpeechDataset(['texts', 'mels', 'mel_gates', 'speakers']),
                                       batch_size=32, mode='valid')

text2mel = Text2Mel(vocab).cuda()

optimizer = torch.optim.Adam(text2mel.parameters(), lr=hp.text2mel_lr)

start_timestamp = int(time.time() * 1000)
start_epoch = 0
global_step = 0

logger = Logger(args.dataset, 'text2mel')


# load the last checkpoint if exists
last_checkpoint_file_name = get_last_checkpoint_file_name(logger.logdir)
if last_checkpoint_file_name:
    print("loading the last checkpoint: %s" % last_checkpoint_file_name)
    # this will load a pretrained model and use the weights of layers that exist in the current model
    start_epoch, global_step = load_and_transform_checkpoint(last_checkpoint_file_name, text2mel, optimizer)
    # start_epoch, global_step = load_checkpoint(last_checkpoint_file_name, text2mel, optimizer)

# adding speaker recognition model load and loss calculation
if hp.use_additional_speaker_loss:
    s = SpeakerNet()
    s.loadParameters(hp.speaker_recognition_model_path)
    s.eval()
    s = s.cuda()
    f = open(logger.logdir + "/eer_log.txt", "a")


def get_lr():
    return optimizer.param_groups[0]['lr']


def lr_decay(step, warmup_steps=4000):
    new_lr = hp.text2mel_lr * warmup_steps ** 0.5 * min((step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
    optimizer.param_groups[0]['lr'] = new_lr


def train(train_epoch, phase='train'):
    global global_step

    lr_decay(global_step)
    print("epoch %3d with lr=%.02e" % (train_epoch, get_lr()))

    text2mel.train() if phase == 'train' else text2mel.eval()
    torch.set_grad_enabled(True) if phase == 'train' else torch.set_grad_enabled(False)
    data_loader = train_data_loader if phase == 'train' else valid_data_loader

    it = 0
    running_loss = 0.0
    running_l1_loss = 0.0
    running_att_loss = 0.0
    running_eer_loss = 0.0

    pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size, disable=hp.disable_progress_bar)
    for batch in pbar:
        L, S, gates, speakers = batch['texts'], batch['mels'], batch['mel_gates'], batch['speakers']
        S = S.permute(0, 2, 1)  # TODO: because of pre processing

        B, N = L.size()  # batch size and text count
        _, n_mels, T = S.size()  # number of melspectrogram bins and time

        assert gates.size(0) == B  # TODO: later remove
        assert gates.size(1) == T

        S_shifted = torch.cat((S[:, :, 1:], torch.zeros(B, n_mels, 1)), 2)

        S.requires_grad = False
        S_shifted.requires_grad = False
        gates.requires_grad = False

        def W_nt(_, n, t, g=0.2):
            return 1.0 - np.exp(-((n / float(N) - t / float(T)) ** 2) / (2 * g ** 2))

        W = np.fromfunction(W_nt, (B, N, T), dtype=np.float32)
        W = torch.from_numpy(W)

        L = L.cuda()
        S = S.cuda()
        S_shifted = S_shifted.cuda()
        W = W.cuda()
        gates = gates.cuda()
        speakers = speakers.cuda()

        Y_logit, Y, A = text2mel(L, S, speakers)

        l1_loss = F.l1_loss(Y, S_shifted)
        masks = gates.reshape(B, 1, T).float()
        att_loss = (A * W * masks).mean()

        if hp.use_additional_speaker_loss:
            y_embedding = s.__S__.forward(Y)
            sc, lab, _ = s.evaluateY(y_embedding, speakers)
            # print("scores: ", sc)
            result = tuneThresholdfromScore(sc, lab, [1, 0.1])
            eer_loss = result[1]/100
            f.write("train_epoch: %d, it: %d, eer_loss: %f \n" % (train_epoch, it, eer_loss))
        else:
            eer_loss = 0

        loss = l1_loss + att_loss + eer_loss

        if phase == 'train':
            lr_decay(global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        it += 1

        if hp.use_additional_speaker_loss:
            loss, l1_loss, att_loss, eer_loss = loss.item(), l1_loss.item(), att_loss.item(), eer_loss.item()
        else:
            loss, l1_loss, att_loss = loss.item(), l1_loss.item(), att_loss.item()

        running_loss += loss
        running_l1_loss += l1_loss
        running_att_loss += att_loss

        if hp.use_additional_speaker_loss:
            running_eer_loss += eer_loss

        if phase == 'train':
            # update the progress bar
            pbar.set_postfix({
                'l1': "%.05f" % (running_l1_loss / it),
                'att': "%.05f" % (running_att_loss / it),
                'eer': "%.05f" % (running_eer_loss / it)
            })
            logger.log_step(phase, global_step, {'loss_l1': l1_loss, 'loss_att': att_loss, 'loss_eer': eer_loss},
                            {'mels-true': S[:1, :, :], 'mels-pred': Y[:1, :, :], 'attention': A[:1, :, :]})
            if global_step % 5000 == 0:
                # checkpoint at every 5000th step
                save_checkpoint(logger.logdir, train_epoch, global_step, text2mel, optimizer)

    epoch_loss = running_loss / it
    epoch_l1_loss = running_l1_loss / it
    epoch_att_loss = running_att_loss / it
    epoch_eer_loss = running_eer_loss / it

    if hp.use_additional_speaker_loss:
        f.write('***************** Total epoch eer loss: %f *****************\n' % (epoch_eer_loss))

    logger.log_epoch(phase, global_step, {'loss_l1': epoch_l1_loss, 'loss_att': epoch_att_loss, 'loss_eer': epoch_eer_loss})

    return epoch_loss


since = time.time()
epoch = start_epoch
while True:
    train_epoch_loss = train(epoch, phase='train')

    time_elapsed = time.time() - since
    time_str = 'total time elapsed: {:.0f}h {:.0f}m {:.0f}s '.format(time_elapsed // 3600, time_elapsed % 3600 // 60,
                                                                     time_elapsed % 60)
    print("train epoch loss %f, step=%d, %s" % (train_epoch_loss, global_step, time_str))

    valid_epoch_loss = train(epoch, phase='valid')
    print("valid epoch loss %f" % valid_epoch_loss)

    epoch += 1
    if global_step >= hp.text2mel_max_iteration:
        print("max step %d (current step %d) reached, exiting..." % (hp.text2mel_max_iteration, global_step))
        sys.exit(0)
