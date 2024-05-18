import os
from io import open
import torch
import numpy as np
# from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader, Dataset

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

class DatasetCSV(Dataset):
    def __init__(self, originalDir,batch_size,device,bptt,train=False):
        self.bptt = bptt
        self.device = device
        self.dictionary = Dictionary()
        self.batch_size = batch_size
        trainSent, valSent = self.getSentences( originalDir)
        trainSent = self.tokenize_redacted(trainSent)
        valSent = self.tokenize_redacted(valSent)
        if train:
            self.sents = trainSent
        else:
            self.sents = valSent

        self.batchify(batch_size)
        self.make_batch_perfect()
        debug_flag = False
        if debug_flag:
            print(self.data[0].size())
            print(self.data[1].size())
            print(self.target[0].size())
            print(self.target[1].size())

    def batchify(self, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = self.sents.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self.sents.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        self.sents = data

    def getSentences(self, originalDir):

        sentOutPairEval = np.load(originalDir + "/0/sentOut.npy", allow_pickle=True)
        # print(sentOutPairEval.shape)
        evalsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPairEval:
            # print(sentIp,gtSentiment)
            if float(gtSentiment) == 0:
                evalsents.append(sentIp)

        numSents = len(evalsents)
        numValSents = int(numSents * 0.1)
        return evalsents[numValSents:], evalsents[:numValSents]

    def tokenize_redacted(self, sentences):
        """Tokenizes a text file."""
        # assert os.path.exists(path)
        # Add words to the dictionary
        # sentOutPair = np.load(path, allow_pickle=True)
        for sentIp in sentences:
            words = sentIp.split() + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)

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


    def __getitem__(self, i):
        # print(self.data[i],self.target[i])
        # print("*"*10)
        return self.data[i],self.target[i]

    def make_batch_perfect(self):
        data = [] # each row is data
        target = [] # each row is data
        for i in range(0, self.sents.size(0) - 1, self.bptt):
            seq_len = min(self.bptt, self.sents.size(0) - 1 - i)
            # self.sents[i],self.sents[i+1:i+1+seq_len].view(-1)
            # print(self.sents[i:i+seq_len].size()) # 35,10
            # print(self.sents[i+1:i+1+seq_len].view(-1).size()) # 350
            data_to_append = torch.t(self.sents[i:i+seq_len])
            target_to_append = torch.t(self.sents[i+1:i+1+seq_len])
            for row_idx in range(data_to_append.size()[0]):
                row_to_append = data_to_append[row_idx,:]
                if row_to_append.size()[0] != self.bptt:
                    continue
                target_row_to_append = target_to_append[row_idx]
                data.append(row_to_append)
                target.append(target_row_to_append)
            # data.append(self.sents[i:i+seq_len])
            # target.append(self.sents[i+1:i+1+seq_len].view(-1))
        # print((data))
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

class DatasetCSV_redcated(Dataset):
    def __init__(self, originalDir,batch_size,device,bptt,train=False,maskperc=30,sensitive=True):
        self.bptt = bptt
        self.sensitive = sensitive
        self._maskperc = maskperc
        self.device = device
        self.dictionary = Dictionary()
        self.batch_size = batch_size
        trainSent, valSent = self.getSentences( originalDir)
        trainSent = self.tokenize_redacted(trainSent)
        valSent = self.tokenize_redacted(valSent)
        if train:
            self.sents = trainSent
        else:
            self.sents = valSent

        self.batchify(batch_size)
        self.make_batch_perfect()
        debug_flag = False
        if debug_flag:
            print(self.data[0].size())
            print(self.data[1].size())
            print(self.target[0].size())
            print(self.target[1].size())

    def batchify(self, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = self.sents.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = self.sents.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        self.sents = data

    def getSentences(self, originalDir):

        sentOutPairEval = np.load(originalDir + "/0/sentOut.npy", allow_pickle=True)
        sentOutPairTrain = np.load(originalDir + "/{}/sentOut.npy".format(self._maskperc), allow_pickle=True)
        # print(sentOutPairEval.shape)
        evalsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPairEval:
            # print(sentIp,gtSentiment)
            if self.sensitive:
                if float(gtSentiment) == 0:
                    evalsents.append(sentIp)
            else:
                if float(gtSentiment) == 1:
                    evalsents.append(sentIp)
        trainsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPairTrain:
            # print(sentIp,gtSentiment)
            if self.sensitive:
                if float(gtSentiment) == 0:
                    trainsents.append(sentIp)
            else:
                if float(gtSentiment) == 1:
                    trainsents.append(sentIp)

        numSents = len(evalsents)
        numValSents = int(numSents * 0.1)
        return trainsents[numValSents:], evalsents[:numValSents]

    def tokenize_redacted(self, sentences):
        """Tokenizes a text file."""
        # assert os.path.exists(path)
        # Add words to the dictionary
        # sentOutPair = np.load(path, allow_pickle=True)
        for sentIp in sentences:
            words = sentIp.split() + ['<eos>']
            for word in words:
                self.dictionary.add_word(word)

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


    def __getitem__(self, i):
        # print(self.data[i],self.target[i])
        # print("*"*10)
        return self.data[i],self.target[i]

    def make_batch_perfect(self):
        data = []
        target = []
        for i in range(0, self.sents.size(0) - 1, self.bptt):
            seq_len = min(self.bptt, self.sents.size(0) - 1 - i)
            # self.sents[i],self.sents[i+1:i+1+seq_len].view(-1)
            # print(self.sents[i:i+seq_len].size()) # 35,10
            # print(self.sents[i+1:i+1+seq_len].view(-1).size()) # 350
            data_to_append = torch.t(self.sents[i:i+seq_len]) # 10,35
            target_to_append = torch.t(self.sents[i+1:i+1+seq_len]) # 10,35
            for row_idx in range(data_to_append.size()[0]):
                row_to_append = data_to_append[row_idx,:]
                if row_to_append.size()[0] != self.bptt:
                    continue
                target_row_to_append = target_to_append[row_idx,:]
                data.append(row_to_append)
                target.append(target_row_to_append)
            # data.append(self.sents[i:i+seq_len])
            # target.append(self.sents[i+1:i+1+seq_len].view(-1))
        # print((data))
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)


