"""Data loader for the LJSpeech dataset. See: https://keithito.com/LJ-Speech-Dataset/"""
import os
import re
import codecs
import unicodedata
import numpy as np

from torch.utils.data import Dataset
from hparams import HParams as hp

vocab = "PSE aăâbcdefghiîjklmnopqrsștțuvwxyz'.?"  # P: Padding, S: SOS, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}
speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))


def text_normalize(text):
    # text = ''.join(char for char in unicodedata.normalize('NFD', text)
    #                if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


def read_metadata(metadata_file):
    fnames, text_lengths, texts, speakers = [], [], [], []
    transcript = os.path.join(metadata_file)
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for line in lines:
        fname, _, text, speaker = line.strip().split("|")

        fnames.append(fname)

        text = "S" + text_normalize(text) + "E"  # S: SOS, E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.long))

        speaker_ix = [speaker2ix[speaker]]
        speakers.append(np.array(speaker_ix, np.long))

    return fnames, text_lengths, texts, speakers


def get_test_data(sentences, max_n):
    normalized_sentences = ["S" + text_normalize(line).strip() + "E" for line in sentences]  # text normalization, E: EOS
    texts = np.zeros((len(normalized_sentences), max_n + 2), np.long)
    print(normalized_sentences, len(normalized_sentences))
    for i, sent in enumerate(normalized_sentences):
        print(sent, [c for c in sent])
        print(len(sent), texts.shape)
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts


def get_speaker_data(speakers):
    return np.array([speaker2ix[speakers]], np.long)


class SWARA(Dataset):
    def __init__(self, keys, dir_name='SWARA_wav_16k'):
        self.keys = keys
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        self.data_path = hp.data_path
        self.fnames, self.text_lengths, self.texts, self.speakers = read_metadata(os.path.join(self.data_path, 'metadata_swara_4ophelia.csv'))

    def slice(self, start, end):
        self.fnames = self.fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]
        self.speakers = self.speakers[start:end]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        data = {}
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'mels' in self.keys:
            # (39, 80)
            data['mels'] = np.load(os.path.join(hp.mels_path, 'mels', "%s.npy" % self.fnames[index]))
        if 'mags' in self.keys:
            # (39, 80)
            data['mags'] = np.load(os.path.join(hp.mags_path, 'mags', "%s.npy" % self.fnames[index]))
        if 'mel_gates' in self.keys:
            data['mel_gates'] = np.ones(data['mels'].shape[0], dtype=np.int)  # TODO: because pre processing!
        if 'mag_gates' in self.keys:
            data['mag_gates'] = np.ones(data['mags'].shape[0], dtype=np.int)  # TODO: because pre processing!
        if 'speakers' in self.keys:
            data['speakers'] = self.speakers[index]
        return data
