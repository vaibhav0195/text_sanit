import os
from io import open
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def save(self,path):
        import json
        with open(path+"/idx2word.json", 'w') as f:
            # indent=2 is not needed but makes the file human-readable
            # if the data is nested
            json.dump(self.idx2word, f, indent=2)

    def load(self,path):
        import json
        with open(path, 'r') as f:
            self.idx2word = json.load(f)

        for idx,word in enumerate(self.idx2word):
            self.word2idx[word] = idx

class Corpus(object):
    def __init__(self, path,eval=None):
        if eval is None:
            self.dictionary = Dictionary()
            self.train = self.tokenize(os.path.join(path, 'traindata.txt'))
            self.valid = self.tokenize(os.path.join(path, 'validsubset.txt'))
        else:
            self.dictionary = Dictionary()
            self.dictionary.load(eval)
        # self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids

class CorpusRedacted(object):
    def __init__(self, path,originalDir,original_dict=None,sensitive=None):
        self._sensitive = sensitive
        trainSent, valSent = self.getSentences(path,originalDir)
        if original_dict is None:
            self.dictionary = Dictionary()
            self.creat_dict()
        else:
            self.dictionary = original_dict
        # trainSent, valSent = self.getSentences(path,originalDir)
        # print(valSent[:2])
        self.train = self.tokenize_redacted(trainSent)
        self.valid = self.tokenize_redacted(valSent)

    def getSentences(self,path,originalDir):
        sentOutPair = np.load(path+"/sentOut.npy", allow_pickle=True)
        sents = []
        allsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPair:
            # allsents.append(sentIp)
            if self._sensitive:
                if float(gtSentiment) == 0:
                    sents.append(sentIp)
            else:
                if float(gtSentiment) == 1:
                    sents.append(sentIp)

        sentOutPairEval = np.load(originalDir+"/0/sentOut.npy", allow_pickle=True)
        print(sentOutPairEval.shape)
        evalsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPairEval:
            allsents.append(sentIp)
            if float(gtSentiment) == 0:
                evalsents.append(sentIp)
        self.original_sents = allsents
        numSents = len(sents)
        numValSents = int(numSents*0.1)
        return sents[numValSents:],evalsents[:numValSents]

    def creat_dict(self):
        for sentIp in self.original_sents:
            words = sentIp.split() + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)
        self.dictionary.add_word("maskgusaintcd")

    def tokenize_redacted(self, sentences):
        """Tokenizes a text file."""
        # assert os.path.exists(path)
        # Add words to the dictionary
        # Tokenize file content
        idss = []
        for sentIp in sentences:
            # if gtSentiment == 0:
            words = sentIp.split() + ['<eos>']
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

        return ids
