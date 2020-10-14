#!/usr/bin/env python3
"""Synthetize sentences into speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import argparse
from tqdm import *

import numpy as np
import torch

from models import Text2Mel, SSRN
from hparams import HParams as hp
from audio import save_to_wav
from utils import get_last_checkpoint_file_name, load_checkpoint, save_to_png

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, choices=['ljspeech', 'mbspeech', 'swara', 'swara_test'], help='dataset name')
args = parser.parse_args()
speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

if args.dataset == 'ljspeech':
    from datasets.lj_speech import vocab, get_test_data

    SENTENCES = [
        "The birch canoe slid on the smooth planks.",
        "Glue the sheet to the dark blue background.",
        "It's easy to tell the depth of a well.",
        "These days a chicken leg is a rare dish.",
        "Rice is often served in round bowls.",
        "The juice of lemons makes fine punch.",
        "The box was thrown beside the parked truck.",
        "The hogs were fed chopped corn and garbage.",
        "Four hours of steady work faced us.",
        "Large size in stockings is hard to sell.",
        "The boy was there when the sun rose.",
        "A rod is used to catch pink salmon.",
        "The source of the huge river is the clear spring.",
        "Kick the ball straight and follow through.",
        "Help the woman get back to her feet.",
        "A pot of tea helps to pass the evening.",
        "Smoky fires lack flame and heat.",
        "The soft cushion broke the man's fall.",
        "The salt breeze came across from the sea.",
        "The girl at the booth sold fifty bonds."
    ]
elif args.dataset == 'swara':
    from datasets.swara import vocab, get_test_data, get_speaker_data

    spk = ['BAS', 'BEA', 'CAU', 'DCS', 'DDM', 'EME', 'FDS', 'HTM', 'IPS', 'MARIA', 'PCS',
     'PMM', 'PSS', 'RMS', 'SAM', 'SDS', 'SGS', 'TSS']
    SENTENCES = [
        "Se așteaptă ca acest mânz să fie unul dintre cele mai importante exemplare din rasa sa, din punct de vedere genetic.|"+sp for sp in spk]
        # "La automatele de vândut cartele se poate achita cu numerar.|"+sp for sp in spk]
        # "Ana are mere.|EME",
        # "de asemenea contează și dacă imobilul este la stradă sau nu.|MARIA",
        # "de aceea spune medicul autoritățile ar trebui să schimbe tonul discuțiilor despre coronavirus.|BAS",
        # "Diaconu a explicat că a fi asimptomatic este o raritate în orice boală, dar că să mărturisești că ai simptome poate duce la stigmă socială, de aceea mulți oameni preferă să tacă și să ascundă ceea ce li se întâmplă.|SAM",
        # "Suntem pregătiți să dăm în trafic cu călători Magistrala de metroul Drumul Taberei, Eroilor în cursul zilei de mâine|BEA"
        # "După semnarea celor două procese verbale de recepție la terminarea lucrărilor, respectiv la punerea în funcțiune, pe data de unu septembrie, am anunțat public",
        # "După îndeplinirea tuturor procedurilor administrative, care au mai rămas de efectuat de către beneficiar",
        # "O altă activitate necesară constă în predarea obiectivelor de pază pe staţii şi interstaţii",
        # "Astfel, în aceasta perioadă, operatorii de telefonie mobilă au finalizat instalarea reţelelor de comunicaţii în subteran.",
        # "La automatele de vândut cartele se poate achita cu numerar.",
        # "Achitarea cu card bancar urmează să se implementeze cât mai curând posibil împreună cu banca suport.",
        # "Este necesar a se realiza o soluţie de comunicaţie stabilă care presupune atât configurarea cât şi instalarea unor echipamente şi chei de criptare.",
        # "De asemenea se predau toate spaţiile publice de la firmele de curăţenie ale constructorului.",
        # "Au semnat procesele-verbale în urma cărora această magistrală de metrou poate fi dată în exploatare şi, totodată, în trafic cu călători.",
        # "Tehnologiile avansate de reproducere, inclusiv clonarea, pot salva speciile, permițându-ne să restabilim diversitatea genetică ce altfel s-ar fi pierdut în timp",
        # "Când va crește, Kurt va fi mutat în Parcul Safari de la grădina zoologică, pentru reproducere.",
        # "Se așteaptă ca acest mânz să fie unul dintre cele mai importante exemplare din rasa sa, din punct de vedere genetic."
    # ]
elif args.dataset == 'swara_test':
    from datasets.swara_test import vocab, get_test_data

    spk = ['BAS', 'BEA', 'CAU', 'DCS', 'DDM', 'EME', 'FDS', 'HTM', 'IPS', 'MARIA', 'PCS',
           'PMM', 'PSS', 'RMS', 'SAM', 'SDS', 'SGS', 'TSS']
    SENTENCES = [
        "ana are mere și pere.|" + sp for sp in spk]
    # ]
elif args.dataset == 'mbspeech':
    from datasets.mb_speech import vocab, get_test_data

    SENTENCES = [
        "Нийслэлийн прокурорын газраас төрийн өндөр албан тушаалтнуудад холбогдох зарим эрүүгийн хэргүүдийг шүүхэд шилжүүлэв.",
        "Мөнх тэнгэрийн хүчин дор Монгол Улс цэцэглэн хөгжих болтугай.",
        "Унасан хүлгээ түрүү магнай, аман хүзүүнд уралдуулж, айрагдуулсан унаач хүүхдүүдэд бэлэг гардууллаа.",
        "Албан ёсоор хэлэхэд “Монгол Улсын хэрэг эрхлэх газрын гэгээнтэн” гэж нэрлээд байгаа зүйл огт байхгүй.",
        "Сайн чанарын бохирын хоолой зарна.",
        "Хараа тэглэх мэс заслын дараа хараа дахин муудах магадлал бага.",
        "Ер нь бол хараа тэглэх мэс заслыг гоо сайхны мэс засалтай адилхан гэж зүйрлэж болно.",
        "Хашлага даван, зүлэг гэмтээсэн жолоочийн эрхийг хоёр жилээр хасжээ.",
        "Монгол хүн бидний сэтгэлийг сорсон орон. Энэ бол миний төрсөн нутаг. Монголын сайхан орон.",
        "Постройка крейсера затягивалась из-за проектных неувязок, необходимости."
    ]

torch.set_grad_enabled(False)

text2mel = Text2Mel(vocab).eval()
text2mel = text2mel.cuda()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-text2mel' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-text2mel/step-020K.pth' % args.dataset
if last_checkpoint_file_name:
    print("loading text2mel checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, text2mel, None)
else:
    print("text2mel not exits")
    sys.exit(1)

ssrn = SSRN().eval()
ssrn = ssrn.cuda()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-ssrn' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-ssrn/step-005K.pth' % args.dataset
if last_checkpoint_file_name:
    print("loading ssrn checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, ssrn, None)
else:
    print("ssrn not exits")
    sys.exit(1)

# synthetize by one by one because there is a batch processing bug!
for i in range(len(SENTENCES)):

    sentence = SENTENCES[i].split("|")[0]
    speaker = SENTENCES[i].split("|")[1]
    sentences = [sentence]

    speakers = [speaker]
    print (speaker)
    speaker_ix = [speaker2ix[speaker]]
    speakers = [speaker_ix]
    print (speaker2ix)
    print (speakers)

    max_N = len(sentence)
    L = torch.from_numpy(get_test_data(sentences, max_N))
    zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32))
    Y = zeros
    A = None

    speakers = torch.from_numpy(np.array(speakers))

    speakers = speakers.cuda()
    L = L.cuda()
    Y = Y.cuda()
    zeros = zeros.cuda()

    for t in tqdm(range(hp.max_T)):
        _, Y_t, A = text2mel(L, Y, speakers, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    _, Z = ssrn(Y)

    Y = Y.cpu().detach().numpy()
    A = A.cpu().detach().numpy()
    Z = Z.cpu().detach().numpy()

    save_to_png('samples/%d-att.png' % (i + 1), A[0, :, :])
    save_to_png('samples/%d-mel.png' % (i + 1), Y[0, :, :])
    save_to_png('samples/%d-mag.png' % (i + 1), Z[0, :, :])
    # import matplotlib.pyplot as plt
    # a = self.embeddings(torch.tensor([x for x in range(10)]).cuda())
    # fig, ax = plt.subplots()
    # plt.imshow(A[0,:,:23])
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels=[x for x in sentence]
    #
    # ax.set_yticklabels(labels[::-1])
    # plt.show()
    # save_to_wav(Z[0, :, :].T, 'samples/%d-wav.wav' % (i + 1))
    print('saving for speaker: ', speaker)
    save_to_wav(Z[0, :, :].T, 'samples/%d-%s-wav.wav' % ((i + 1), speaker))
