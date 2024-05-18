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
    def __init__(self, path,originalDir,original_dict=None,sensitive=True):
        self.sensitive = sensitive
        trainSent, valSent = self.getSentences(path,originalDir)
        if original_dict is None:
            self.dictionary = Dictionary()
            self.creat_dict()
        else:
            self.dictionary = original_dict
        # print(valSent[:2])
        self.train = self.tokenize_redacted(trainSent)
        self.valid = self.tokenize_redacted(valSent)

    def getSentences(self,path,originalDir):
        sentOutPair = np.load(path+"/sentOut.npy", allow_pickle=True)
        sents = []
        allsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPair:
            # print(sentIp,gtSentiment)

            if self.sensitive:
                if float(gtSentiment) == 0:
                    sents.append(sentIp)
            else:
                if float(gtSentiment) == 0:
                    sents.append(sentIp)
            # break
        sentOutPairEval = np.load(originalDir+"/0/sentOut.npy", allow_pickle=True)
        print(sentOutPairEval.shape)
        evalsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPairEval:
            # print(sentIp,gtSentiment)
            allsents.append(sentIp)
            if self.sensitive:
                if float(gtSentiment) == 0:
                    evalsents.append(sentIp)
            else:
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

class CorpusRedacted_original(object):
    def __init__(self, path,originalDir,original_dict=None,sensitive=True):
        self.sensitive = sensitive
        # self._bptt = bptt
        redacted_sents, original_sents = self.getSentences(path,originalDir)
        self.get_updated_redacted(redacted_sents,original_sents)
        if original_dict is None:
            self.dictionary = Dictionary()
            self.creat_dict()
        else:
            self.dictionary = original_dict
        # print(valSent[:2])
        self.train = self.tokenize_redacted(original_sents)
        self.valid = self.tokenize_redacted(self.redacted_setns)

    def compare_redact(self,masked_sent, original_sent):
        words_in_original = original_sent.split()
        words_in_masked = masked_sent.split()
        idx_masked = 0
        new_sent = ""
        for idx in range(0, len(words_in_original)):
            word_at_idx = words_in_original[idx]
            redacted_word_at_idx = words_in_masked[idx_masked]

            if word_at_idx == redacted_word_at_idx:
                word_to_use = word_at_idx
                idx_masked = idx_masked + 1
            else:
                if redacted_word_at_idx == "maskgusaintcd":
                    if idx_masked == len(words_in_masked) - 1:
                        word_to_use = "maskgusaintcd"
                    else:
                        idx_new = idx_masked + 1
                        if words_in_masked[idx_new] == word_at_idx:
                            word_to_use = word_at_idx
                            idx_masked = idx_masked + 2
                        else:
                            word_to_use = "maskgusaintcd"
                else:
                    # print(word_at_idx,redacted_word_at_idx)
                    word_to_use = "maskgusaintcd"
            new_sent = new_sent + " " + word_to_use
        return new_sent

    def get_updated_redacted(self,redacted_sents,original_sents):
        # print(len(redacted_sents),len(original_sents))
        updated_redact_sents = []
        for idx,redacted_sent in enumerate(redacted_sents):
            original_sent_idx = original_sents[idx]
            updated_masked_sentence = self.compare_redact(redacted_sent,original_sent_idx)
            updated_redact_sents.append(updated_masked_sentence)
        self.redacted_setns = updated_redact_sents

    def getSentences(self,path,originalDir):
        sentOutPair = np.load(path+"/sentOut.npy", allow_pickle=True)
        sents = []
        allsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPair:
            # print(sentIp,gtSentiment)
            if self.sensitive:
                if float(gtSentiment) == 0:
                    sents.append(sentIp)
            else:
                if float(gtSentiment) == 0:
                    sents.append(sentIp)
        sentOutPairEval = np.load(originalDir+"/0/sentOut.npy", allow_pickle=True)
        # print(sentOutPairEval.shape)
        evalsents = []
        for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPairEval:

            allsents.append(sentIp)
            if self.sensitive:
                if float(gtSentiment) == 0:
                    evalsents.append(sentIp)
            else:
                if float(gtSentiment) == 0:
                    evalsents.append(sentIp)

        numSents = len(sents)
        numValSents = int(numSents * 0.1)


        return sents[:numValSents],evalsents[:numValSents]

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

class CorpusMaskLeak(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def getMaskSentPair(self,wordmapJson,textFilePathOrignal,numWords):
        import json
        import numpy as np

        def createabbervationlist(wordMap):
            abbervations = {}
            nonsensAbb = {}
            for data in wordMap:
                label = wordMap[data]
                if label.lower().endswith("cancer") or label.lower().endswith("tumor") or label.lower().endswith(
                        "carcinoma") or "hiv" in label.lower():
                    abbervations[data.lower()] = label
                else:
                    nonsensAbb[data.lower()] = label
            # print(tot, len(tot))
            return abbervations, nonsensAbb

        def getMaskedInp(valDatalines, abbr, maskToken="<eos>", numletters=4):
            """
            this functions replaces the abbervations with the mask token
            :param valDatalines: text line by line here the diseases names should be in the abbervations fmt. NOT THE full name
            as we will be replacing them with mask token.
            :param abbr: dictionary abbervations
            :param maskToken: mask token which the model thinks is the token
            :param numletters: number of letters to keep before the mask token
            :return:
            """
            sentLabel = []
            for sentence in valDatalines:
                wordsList = sentence.split(" ")
                for idx, words in enumerate(wordsList):
                    if words in abbr:
                        # found the abbervation now replace it with mask and chose numletters previous words.
                        if (idx - numletters + 1 > 0):
                            wordsToConsider = wordsList[idx - numletters + 1:idx]
                            for tword in wordsToConsider:
                                self.dictionary.add_word(tword)
                            for tword in abbr[words].lower().split():
                                self.dictionary.add_word(tword)
                            strWords = ' '.join([str(elem) for elem in wordsToConsider])
                            sentLabel.append([strWords + " " + maskToken, abbr[words].lower()])

            sentLabel = np.asarray(sentLabel)
            return sentLabel

        with open(wordmapJson) as json_file:
            wordmap = json.load(json_file)

        abbr, _ = createabbervationlist(wordmap)

        with open(textFilePathOrignal) as f:
            validDataLines = [line.rstrip('\n') for line in f]

        maskToken = "<eos>"
        sentLab = getMaskedInp(validDataLines,abbr,maskToken,numWords)
        self.dictionary.add_word(maskToken)
        idss = []
        oIdss = []
        for line,op in sentLab:
            words = line.split()
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))


            wordsO = op.split()
            oids = []
            for wordo in wordsO:
                oids.append(self.dictionary.word2idx[wordo])
            oIdss.append(torch.tensor(oids).type(torch.int64))

        ids = torch.cat(idss)
        oids = torch.cat(oIdss)
        return ids,oids
