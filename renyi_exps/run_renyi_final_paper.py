import os

import numpy as np
from estimator import estimate_rynei as est_r
import random
import mauve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def getRandomWithReplacement(inp_feat):
    inp_ret = []
    numSamplesretinp = []
    for i in range(inp_feat.shape[0]):
        numSamplesretinp.append(i)
    idxsSampleToUse = random.choices(numSamplesretinp,k=inp_feat.shape[0])
    for idx_inp in idxsSampleToUse:
        inp_ret.append(inp_feat[idx_inp])
    return inp_ret,idxsSampleToUse

def get_sent_features(p_text_sampled,q_text_sampled,sent_trans):
    p_feat_sampled_glove = sent_trans.encode(p_text_sampled)
    q_feat_sampled_glove = sent_trans.encode(q_text_sampled)
    p_feat_sampled_glove, q_feat_sampled_glove = normaliseData(p_feat_sampled_glove, q_feat_sampled_glove)
    return p_feat_sampled_glove,q_feat_sampled_glove

def getInpfeatures(featuresPath):
    features = np.load(featuresPath, allow_pickle=True)
    return features

def mauve_divergence(p_feats,q_feats,k=5):
    out = mauve.compute_mauve(p_features=p_feats, q_features=q_feats,
                              num_buckets=k, verbose=False, mauve_scaling_factor=1.0,kmeans_explained_var=1)
    divergence = np.max([-np.log(out.divergence_curve[1, 1]), -np.log(out.divergence_curve[-2, 0])])
    return divergence

def divergence_Knn(p_feats,q_feats,k=5,alpha=2):
    return max(est_r(p_feats, q_feats,alpha=alpha, k=k,rounding=5),est_r(q_feats, p_feats, k=k,rounding=5,alpha=alpha))

def get_divergence_roc(p_feat_glove,q_feat_glove,gt,pred,alpha=2):

    k=5
    divergence_glove =  divergence_Knn(p_feat_glove, q_feat_glove, k=k,alpha=alpha)
    y_pt = roc_auc_score(gt,pred)
    return divergence_glove, y_pt


def getVectoriserAndModel(inpDir):

    import pickle
    with open('{}/vectorizer.pickle'.format(inpDir), 'rb') as f:
        vectorizer = pickle.load(f)
    with open('{}/model.pickle'.format(inpDir), 'rb') as f:
        clf = pickle.load(f)
    return clf, vectorizer

def normaliseData(p,q,norm='l2', whiten=False,
                         pca_max_data=-1,
                         explained_variance=0.9,seed=25):

    data1 = np.vstack([q, p])
    if norm in ['l2', 'l1']:
        data1 = normalize(data1, norm=norm, axis=1)
    pca = PCA(n_components=None, whiten=whiten, random_state=seed + 1)
    pca.fit(data1)
    s = np.cumsum(pca.explained_variance_ratio_)
    idx = np.argmax(s >= explained_variance)  # last index to consider
    data1 = pca.transform(data1)[:, :idx + 1]
    p_data = data1[q.shape[0]: , :]
    q_data = data1[:q.shape[0], :]
    return p_data,q_data

if __name__ == '__main__':
    # "/Users/vaibhavgusain/ML/kl_div_exps_git/smartMaskingIMDB_MEdal_embs_not_norm"
    # dirNames = ["random_amazon_utility_redaction_embs_not_norm_glove","smart_amazon_utility_redaction_embs_not_norm_glove"]
    # dirNames = ["smartMaskingValidationMedal_various_length_rm_single_word_embs"]
    # dirNames = ["smartMaskingValidationMedal_various_length_rm_single_word_embs"]
    # dirNames = ["smartMaskingValidationMedal_using_random_split_val_embs"]
    # dirNames = ["smartMaskingValidationMedal_using_random_split_embs"]
    # dirNames = ["smart_amazon_utility_redaction_old_embs_not_norm"]
    # dirNames = ["smart_amazon_utility_redaction_embs_not_norm"]
    # dirNames = ["randomMaskingEqSmartMedal_embs_not_norm"]
    # dirNames = ["smartMaskingValidationMedalsame_length_embs_not_norm"]
    main_data_dir = "/Users/vaibhavgusain/ML/KL_DPSGD/emebeddings_sent"
    alpha = 2
    map_vector_dirname = {"smart_amazon_utility_redaction_embs_not_norm":"smart_amazon_utility_redaction",
                          "smart_masking_redit_suicide_sent_embs":"smart_masking_redit_suicide",
                          "smartMaskingPoliticalDataset_sent_embs":"smartMaskingPoliticalDataset",
                          "smartMaskingValidationMedalSepTrain_diff_length_sent_embs":"smartMaskingValidationMedalSepTrain_diff_length"}
    for dirName in os.listdir(main_data_dir):
        dataToPlot = []
        model_dir = map_vector_dirname[dirName]
        # vectoriser_dir_name = dirName.split("_")[:]
        # for i in range(len(vectoriser_dir_name))
        if "amazon" in dirName:
            clf, vectoriser = getVectoriserAndModel("{}".format(model_dir))
        else:
            clf, vectoriser = getVectoriserAndModel("/Users/vaibhavgusain/ML/bartExps/{}".format(model_dir))
        # for maskPerc in [90]:
        print(dirName)
        for maskPerc in [0,10,20,30,40,50,60,70,80,90,100]:
            divergences = []
            roc_score = []
            if maskPerc == 100:
                for bootstap_idx in range(25):
                    divergences.append(0)
                    roc_score.append(0.5)
                divergences = np.asarray(divergences)
                roc_score = np.asarray(roc_score)
                dataToPlot.append([maskPerc,np.mean(divergences),
                                   np.std(divergences),np.mean(roc_score),np.std(roc_score)])
                continue
            # print("********{}_{}*********".format(dirName, maskPerc))
            # eps_to_use = eps_dict[dirName][maskPerc]
            numWordsToMask = float(maskPerc)
            p_data_path = main_data_dir+"/"+dirName+"/{}/p_embs.npy".format(maskPerc)
            q_data_path = main_data_dir+"/"+dirName+"/{}/q_embs.npy".format(maskPerc)

            p_data = getInpfeatures(p_data_path)
            q_data = getInpfeatures(q_data_path)
            p_text = getInpfeatures(main_data_dir+"/"+dirName+"/{}/p_sent.npy".format(maskPerc))
            q_text = getInpfeatures(main_data_dir+"/"+dirName+"/{}/q_sent.npy".format(maskPerc))

            p_data,q_data = normaliseData(p_data,q_data)
            # map_alpha = {}
            for bootstap_idx in range(25):
                # p_feat_sampled_updated, _ = getRandomWithReplacement(p_data_updated)
                # q_feat_sample_updated, _ = getRandomWithReplacement(q_data_updated)
                p_feat_sampled, btstrap_p_idx = getRandomWithReplacement(p_data)
                q_feat_sampled, btstrap_q_idx = getRandomWithReplacement(q_data)
                p_text_sampled = p_text[btstrap_p_idx]
                q_text_sampled = q_text[btstrap_q_idx]
                gt_n = [0 for _ in p_text_sampled]
                for temp_ in q_text_sampled:
                    gt_n.append(1)
                test_data = []
                test_data.extend(p_text_sampled)
                test_data.extend(q_text_sampled)
                test_data_transformed = vectoriser.transform(test_data)
                prediction_n = clf.predict(test_data_transformed)
                p_feat_sampled = np.asarray(p_feat_sampled)
                q_feat_sampled = np.asarray(q_feat_sampled)
                # p_feat_sampled,q_feat_sampled = normaliseData(p_feat_sampled,q_feat_sampled)

                # for alpha in range(2,7,2):
                divergence_glove, ypt = get_divergence_roc(p_feat_sampled, q_feat_sampled,
                                                           gt_n,prediction_n,alpha=alpha)
                # divergences_glove.append([alpha,divergence_glove])
                # _, ypt = get_divergence_roc(p_feat_sampled, q_feat_sampled,
                #                                                gt_n,prediction_n,alpha=2)
                roc_score.append(ypt)
                divergences.append(divergence_glove)
            dataToPlot.append([maskPerc, np.mean(divergences),
                               np.std(divergences), np.mean(roc_score), np.std(roc_score)])

        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        dataToPlot_new = np.asarray(dataToPlot)
        ax.errorbar(dataToPlot_new[:, 0], dataToPlot_new[:, 1], dataToPlot_new[:, 2], label="Renyi-divergence alpha={}".format(alpha),color='g')
        ax2.errorbar(dataToPlot_new[:, 0], dataToPlot_new[:, 3], dataToPlot_new[:, 4], label="Attack Accuracy".format(alpha),color='r')
        ax.set_xlabel('WordsMasked %')
        ax.set_ylabel('Divergence')
        ax2.set_ylabel('Attack Accuracy')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        # plt.legend()
        plt.tight_layout()
        # plt.show()
        # with open('test.npy', 'wb') as f:
        np.save("renyi_plots/{}.npy".format(model_dir), dataToPlot_new)
        plt.savefig("renyi_plots/{}_final_renyi_plot.eps".format(dirName), format="eps", bbox_inches='tight')

    pass

