#!/usr/bin/python
#-*- coding: utf-8 -*-

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from pretrained_voxceleb_model.tuneThreshold import tuneThresholdfromScore
from pretrained_voxceleb_model.DatasetLoader import loadWAV
from pretrained_voxceleb_model.loss import *
from hparams import HParams as hp

def initialize_validation_file_list(listfilename):
    lines = []
    files = []

    ## Read all lines
    with open(listfilename) as listfile:
        while True:
            line = listfile.readline();
            if (not line):
                break;

            data = line.split();

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0, 1)] + data

            files.append(data[1])
            files.append(data[2])
            lines.append(line)

    setfiles = list(set(files))
    setfiles.sort()

    return setfiles, lines

class SpeakerNet(nn.Module):

    # def __init__(self, lr = 0.0001, margin = 1, scale = 1, hard_rank = 0, hard_prob = 0, model="ResNetSE34L", nOut = 512, nSpeakers = 1000, optimizer = 'adam', encoder_type = 'SAP', normalize = True, trainfunc='softmax', n_mels=40, log_input=True, **kwargs):
    def __init__(self, lr=0.001, margin=0.3, scale=30, hard_rank=10, hard_prob=0.5, model="ResNetSE34L", nOut=512,
                     nSpeakers=18, optimizer='adam', encoder_type='SAP', normalize=True, trainfunc='softmax',
                     n_mels=80, log_input=True, **kwargs):
        super(SpeakerNet, self).__init__();

        argsdict = {'nOut': nOut, 'encoder_type':encoder_type, 'nClasses':nSpeakers, 'margin':margin, 'scale':scale, 'hard_prob':hard_prob, 'hard_rank':hard_rank, 'log_input':log_input, 'n_mels':n_mels}
        print(argsdict)

        SpeakerNetModel = importlib.import_module('pretrained_voxceleb_model.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**argsdict).cuda()

        if optimizer == 'adam':
            self.__optimizer__ = torch.optim.Adam(self.parameters(), lr = lr)
        elif optimizer == 'sgd':
            self.__optimizer__ = torch.optim.SGD(self.parameters(), lr = lr, momentum = 0.9, weight_decay=5e-5)
        else:
            raise ValueError('Undefined optimizer.')

        self.test_files = self.create_vgg_train_file_dict()

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.train();
        return 0

    def create_vgg_train_file_dict(self):

        # creating a dictionary of the vgg train files for each speaker
        speaker_train_samples = {}
        vgg_train_files, vgg_train_speakers = [], []
        with open(hp.test_file_list) as f:
            for line in f:
                file_name = line.strip().replace(".wav","")
                speaker = hp.speaker2ix[file_name.split("_")[0].upper()]
                vgg_train_files.append(file_name)
                vgg_train_speakers.append(speaker)
        for i in range(len(vgg_train_files)):
            spk_id = int(vgg_train_speakers[i])
            if spk_id in speaker_train_samples:
                speaker_train_samples[spk_id].append(vgg_train_files[i].strip())
            else:
                speaker_train_samples[spk_id] = [vgg_train_files[i].strip()]

        return speaker_train_samples

    # evaluate Y generated from the dc tts network
    def evaluateY(self, Y, speakers):

        self.eval()
        feats = {}
        batch_length = Y.shape[0]
        lines = []

        for i in range(batch_length):
            speaker = speakers[i]
            feats['Y'+str(i)] = Y[i] # [:,:,:40]

            # same speakers
            speaker_item = speaker.item()
            comp_file_path = random.choices(self.test_files[speaker_item], k=hp.same_spk_samples)
            for j in range(len(comp_file_path)):
                # print("file: ", comp_file_path[j])
                feats[comp_file_path[j]] = numpy.load(hp.embedding_files + comp_file_path[j] + ".npy")
                # print(feats[comp_file_path[j]].shape)
                lines.append("1 " + "Y" + str(i) + " " + comp_file_path[j] + "\n")

            # other speakers
            comp_file_path = []
            for spk in sorted(self.test_files):
                if spk == speaker_item:
                    continue
                comp_file_path.extend(random.choices(self.test_files[spk], k=hp.other_spk_samples))

            for j in range(len(comp_file_path)):
                # print("file: ", comp_file_path[j])
                feats[comp_file_path[j]] = numpy.load(hp.embedding_files + comp_file_path[j] + ".npy")
                # print(feats[comp_file_path[j]].shape)
                lines.append("0 " + "Y" + str(i) + " " + comp_file_path[j] + "\n")

        # lines = ['1 Y0 bas_rnd3_127', '1 Y0 bas_rnd2_371', '0 Y0 bea_rnd3_166', '0 Y0 cau_rnd2_038', '0 Y0 dcs_rnd3_031', '0 Y0 ddm_rnd2_335', '0 Y0 eme_rnd2_457', '0 Y0 fds_rnd1_014', '0 Y0 htm_rnd1_067', '0 Y0 ips_rnd2_184', '0 Y0 maria_rnd2_301', '0 Y0 pcs_rnd2_108', '0 Y0 pmm_rnd1_274', '0 Y0 pss_rnd1_044', '0 Y0 rms_rnd1_256', '0 Y0 sam_rnd2_365', '0 Y0 sds_rnd1_228', '0 Y0 sgs_rnd1_238', '0 Y0 tim_rnd2_381', '0 Y0 tss_rnd1_433']
        # lines = ['1 Y0 fds_rnd2_432', '1 Y0 fds_rnd1_257', '0 Y0 bas_rnd2_400', '0 Y0 bea_rnd2_142', '0 Y0 cau_rnd2_120', '0 Y0 dcs_rnd3_431', '0 Y0 ddm_rnd1_090', '0 Y0 eme_rnd3_228', '0 Y0 htm_rnd1_204', '0 Y0 ips_rnd2_366', '0 Y0 maria_rnd1_018', '0 Y0 pcs_rnd2_417', '0 Y0 pmm_rnd1_353', '0 Y0 pss_rnd2_093', '0 Y0 rms_rnd2_383', '0 Y0 sam_rnd1_212', '0 Y0 sds_rnd2_199', '0 Y0 sgs_rnd2_467', '0 Y0 tim_rnd2_287', '0 Y0 tss_rnd2_371', '1 Y1 sam_rnd2_149', '1 Y1 sam_rnd2_125', '0 Y1 bas_rnd3_443', '0 Y1 bea_rnd1_446', '0 Y1 cau_rnd2_043', '0 Y1 dcs_rnd1_223', '0 Y1 ddm_rnd1_120', '0 Y1 eme_rnd3_062', '0 Y1 fds_rnd2_378', '0 Y1 htm_rnd2_317', '0 Y1 ips_rnd2_172', '0 Y1 maria_rnd1_305', '0 Y1 pcs_rnd1_478', '0 Y1 pmm_rnd1_140', '0 Y1 pss_rnd3_297', '0 Y1 rms_rnd2_461', '0 Y1 sds_rnd2_090', '0 Y1 sgs_rnd1_487', '0 Y1 tim_rnd1_379', '0 Y1 tss_rnd1_202']
        # for i in range(len(lines)):
        #     l, _, f = lines[i].split(" ")
        #     feats[f] = numpy.load(hp.embedding_files + f + ".npy")

        # print(lines)

        all_scores = []
        all_labels = []
        all_trials = []

        # Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0, 1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = torch.Tensor(feats[data[2]]).cuda()

            ref_feat = ref_feat.unsqueeze(0)

            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

            dist = F.pairwise_distance(ref_feat.unsqueeze(-1),
                                       com_feat.unsqueeze(-1).transpose(0, 2)).detach().cpu().numpy()

            score = -1 * numpy.mean(dist)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

        return all_scores, all_labels, all_trials

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Update learning rate
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def updateLearningRate(self, alpha):

        learning_rate = []
        for param_group in self.__optimizer__.param_groups:
            param_group['lr'] = param_group['lr']*alpha
            learning_rate.append(param_group['lr'])

        return learning_rate


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        torch.save(self.state_dict(), path);

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict()
        loaded_state = torch.load(path)
        print("Loading parameters from: ", path)
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

