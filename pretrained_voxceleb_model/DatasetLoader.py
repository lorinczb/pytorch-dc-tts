#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torchaudio
import numpy
import random
import pdb
import os
import threading
import time
import math
from scipy.io import wavfile
from queue import Queue

# torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, win_length=800, hop_length=200,
#                                                    window_fn=torch.hamming_window, n_mels=80)


def round_down(num, divisor):
    return num - (num%divisor)


def get_spectrogram_dctts(fpath):
    import librosa
    import numpy as np
    y, sr = librosa.load(fpath, sr=16000)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=2048,
                          hop_length=200,
                          win_length=800)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(16000, 2048, 80)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    max_db = 100
    ref_db = 20
    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    mel = mel[::4, :]
    return torch.Tensor(mel)


def loadWAV(filename, max_frames, evalmode=True, num_eval=10, n_mels=80):

    # # Maximum audio length
    # max_audio = max_frames * 160 + 240
    # print("max_frames: ", max_frames)
    # # Read wav file and convert to torch tensor
    # sample_rate, audio  = wavfile.read(filename)
    #
    # audiosize = audio.shape[0]
    #
    # if audiosize <= max_audio:
    #     shortage    = math.floor( ( max_audio - audiosize + 1 ) / 2 )
    #     audio       = numpy.pad(audio, (shortage, shortage), 'constant', constant_values=0)
    #     audiosize   = audio.shape[0]
    #
    # if evalmode:
    #     startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    # else:
    #     startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    #
    # feats = []
    # if evalmode and max_frames == 0:
    #     feats.append(audio)
    # else:
    #     for asf in startframe:
    #         feats.append(audio[int(asf):int(asf)+max_audio])
    #
    # feat = numpy.stack(feats,axis=0)
    #
    # feat = torch.FloatTensor(feat)

    mel_input = get_spectrogram_dctts(filename) + 1e-6
    mel_input = mel_input.permute(1, 0)
    mel_input = mel_input.unsqueeze(0)

    return mel_input


class DatasetLoader(object):
    def __init__(self, dataset_file_name, batch_size, max_frames, max_seg_per_spk, nDataLoaderThread, nPerSpeaker, train_path, maxQueueSize = 10, **kwargs):
        self.dataset_file_name = dataset_file_name;
        self.nWorkers = nDataLoaderThread;
        self.max_frames = max_frames;
        self.max_seg_per_spk = max_seg_per_spk;
        self.batch_size = batch_size;
        self.maxQueueSize = maxQueueSize;

        self.data_dict = {};
        self.data_list = [];
        self.nFiles = 0;
        self.nPerSpeaker = nPerSpeaker; ## number of clips per sample (e.g. 1 for softmax, 2 for triplet or pm)

        self.dataLoaders = [];
        
        ### Read Training Files...
        with open(dataset_file_name) as dataset_file:
            while True:
                line = dataset_file.readline();
                if not line:
                    break;
                
                data = line.split();
                speaker_name = data[0];
                filename = os.path.join(train_path,data[1]);

                if not (speaker_name in self.data_dict):
                    self.data_dict[speaker_name] = [];

                self.data_dict[speaker_name].append(filename);

        ### Initialize Workers...
        self.datasetQueue = Queue(self.maxQueueSize);

    def dataLoaderThread(self, nThreadIndex):
        
        index = nThreadIndex*self.batch_size;

        if(index >= self.nFiles):
            return;

        while(True):
            if(self.datasetQueue.full() == True):
                time.sleep(1.0);
                continue;

            in_data = [];
            for ii in range(0,self.nPerSpeaker):
                feat = []
                for ij in range(index,index+self.batch_size):
                    feat.append(loadWAV(self.data_list[ij][ii], self.max_frames, evalmode=False));
                in_data.append(torch.cat(feat, dim=0));

            in_label = numpy.asarray(self.data_label[index:index+self.batch_size]);
            
            self.datasetQueue.put([in_data, in_label]);

            index += self.batch_size*self.nWorkers;

            if(index+self.batch_size > self.nFiles):
                break;

    def __iter__(self):

        dictkeys = list(self.data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = self.data_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            
            rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        self.data_list  = [flattened_list[i] for i in mixmap]
        self.data_label = [flattened_label[i] for i in mixmap]
        
        ## Iteration size
        self.nFiles = len(self.data_label);

        ### Make and Execute Threads...
        for index in range(0, self.nWorkers):
            self.dataLoaders.append(threading.Thread(target = self.dataLoaderThread, args = [index]));
            self.dataLoaders[-1].start();

        return self;


    def __next__(self):

        while(True):
            isFinished = True;
            
            if(self.datasetQueue.empty() == False):
                return self.datasetQueue.get();
            for index in range(0, self.nWorkers):
                if(self.dataLoaders[index].is_alive() == True):
                    isFinished = False;
                    break;

            if(isFinished == False):
                time.sleep(1.0);
                continue;


            for index in range(0, self.nWorkers):
                self.dataLoaders[index].join();

            self.dataLoaders = [];
            raise StopIteration;


    def __call__(self):
        pass;

    def getDatasetName(self):
        return self.dataset_file_name;

    def qsize(self):
        return self.datasetQueue.qsize();