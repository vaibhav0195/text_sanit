import numpy as np
import torchtext
from torchtext.vocab import Vectors
from collections import  Counter
import re
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import os
from sentence_transformers import SentenceTransformer


def create_clean_counter(input_data, add_space_split=False):
    # create phrases persent in the data to get the embeedings for those words
    phrase_count = Counter()
    for original_text in input_data:

        # original_text = " ".join(review)
        text = original_text.replace(
            " ' ", "").replace("'", "").replace("/", " ").replace("  ", " ").replace('"', '')
        if add_space_split:
            text = re.split('\!|\,|\n|\.|\?|\-|\;|\:|\(|\)|\s', text)
        else:
            text = re.split('\!|\,|\n|\.|\?|\-|\;|\:|\(|\)', text)
        sentences = [x.strip() for x in text if x.strip()]
        for sentence in sentences:
            phrase_count[sentence] += 1
    return phrase_count

def clean_example(example):
    # clean the input example
    original_text = " ".join(example)
    clean_text = original_text.replace(
        "'", " ").replace("/", " ").replace("  ", " ").replace('"', '')
    text = re.split('\!|\,|\n|\.|\?|\-|\;|\:|\(|\)', clean_text)
    return text

def convert_sent_to_embs(sent,train_vocab):
    embs = []
    for word in sent.split(" "):
        original_vec = train_vocab.vectors[train_vocab.stoi[word]]
        embs.append(original_vec.tolist())
    embs = np.asarray(embs)

    embs = np.average(embs, axis=0)
    # print(embs.shape)
    return embs

def get_glove_for_sents(p_text,q_text,vocab):
    p_feat = []
    q_feat = []
    for text in p_text:
        p_feat.append(convert_sent_to_embs(text,vocab))

    for text in q_text:
        q_feat.append(convert_sent_to_embs(text,vocab))
    p_feat = np.asarray(p_feat)
    q_feat = np.asarray(q_feat)
    return p_feat,q_feat

def getPqFromNpyData(npyFilePath):
    """
    get the predictions of each point from p and q distribution
    Parameters
    ----------
    npyFilePath

    Returns
    -------

    """
    sentOutPair = np.load(npyFilePath, allow_pickle=True)
    p_pred = []
    q_pred = []
    p_text = []
    q_text = []
    for sentIp, gtSentiment, predictedSentiment, actualSent in sentOutPair:
        # sentIpLength = len(sentIp.split())
        # sentLengths.append(sentIpLength)
        if float(gtSentiment) == 0:
            # num_p_padded.append(paded)
            p_text.append(sentIp)
            p_pred.append(float(predictedSentiment[0]))
        else:
            # num_q_padded.append(paded)
            q_text.append(sentIp)
            q_pred.append(float(predictedSentiment[0]))
    # print("Number of paded sentence in p {} and q {}".format(sum(num_p_padded),sum(num_q_padded)))
    print("length of p dataset {}, length of q dataset {}".format(len(p_text),len(q_text)))
    return p_text,q_text,p_pred, q_pred


def normaliseData(p,q,norm='l2', whiten=False,
                         pca_max_data=-1,
                         explained_variance=0.9,seed=25):
    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    # varData = np.sum(np.var(data1,axis=0))
    # if varData > 0.5:
    pca = PCA(n_components=None, whiten=whiten, random_state=seed + 1)
    pca.fit(data1)
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    data1 = pca.transform(data1)[:, :idx + 1]
    p_data = data1[q.shape[0]: , :]
    q_data = data1[:q.shape[0], :]
    return p_data,q_data


def get_glove_features(p_text_sampled,q_text_sampled,vectors):
    reviews = []
    reviews.extend(p_text_sampled)
    reviews.extend(q_text_sampled)
    phrase_count_complete = create_clean_counter(reviews, add_space_split=True)
    train_vocab = torchtext.vocab.Vocab(counter=phrase_count_complete)
    train_vocab.load_vectors(vectors)
    p_feat_sampled_glove, q_feat_sampled_glove = get_glove_for_sents(p_text_sampled, q_text_sampled, train_vocab)
    p_feat_sampled_glove, q_feat_sampled_glove = normaliseData(p_feat_sampled_glove, q_feat_sampled_glove)
    return p_feat_sampled_glove,q_feat_sampled_glove

def get_sent_features(p_text_sampled,q_text_sampled,sent_trans):
    p_feat_sampled_glove = sent_trans.encode(p_text_sampled)
    q_feat_sampled_glove = sent_trans.encode(q_text_sampled)
    # p_feat_sampled_glove, q_feat_sampled_glove = normaliseData(p_feat_sampled_glove, q_feat_sampled_glove)
    return p_feat_sampled_glove,q_feat_sampled_glove



if __name__ == '__main__':
    # dirNames = ["smart_masking_redit_suicide","smartMaskingPoliticalDataset","smartMaskingValidationMedalSepTrain_diff_length"]
    dirNames = ["smartMaskingIMDB_MEdal"]
    sentenceTransformerModel = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    n_bins = 5

    for dirName in dirNames:
        save_dir = "emebeddings_sent/{}_sent_embs".format(dirName)
        print(dirName)
        # outDir = "/Users/vaibhavgusain/ML/bartExps/{}".format(dirName)
        dataToPlot = []

        for maskPerc in [0,10,20,30,40,50,60,70,80,90]:
            embs_save_dir = save_dir+"/{}".format(maskPerc)
            os.makedirs(embs_save_dir,exist_ok=True)
            print("********{}_{}*********".format(dirName, maskPerc))
            numWordsToMask = float(maskPerc)
            npyDataFilePath = "/Users/vaibhavgusain/ML/bartExps/"+dirName + "/{}/sentOut.npy".format(maskPerc)
            p_text,q_text, p_pred, q_pred = getPqFromNpyData(npyDataFilePath)
            p_feat, q_feat = get_sent_features(p_text, q_text,sentenceTransformerModel)
            np.save("{}/p_sent.npy".format(embs_save_dir), p_text)
            np.save("{}/q_sent.npy".format(embs_save_dir), q_text)
            np.save("{}/p_embs.npy".format(embs_save_dir), p_feat)
            np.save("{}/q_embs.npy".format(embs_save_dir), q_feat)


    pass

