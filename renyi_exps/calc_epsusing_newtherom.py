import os
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from kneed import  KneeLocator
from sklearn.metrics import roc_auc_score

def get_eps(rho_temp,delta=0.00008):
    sqrt_term = rho_temp * math.log(1 / delta)  # 2nd page first prep from the paper
    epsilon = rho_temp + 2 * math.sqrt(sqrt_term)
    return epsilon

def get_line_eqn(x1,x2,y1,y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope,intercept
    # rho_plot = [[x1, y1], [x2, y2]]
def get_line_eqn_nointercept(x1,y1):
    # slope = (y2 - y1) / (x2 - x1)
    # intercept = y1 - slope * x1
    slope = y1/x1
    intercept = 0
    return slope,intercept

def get_line_plots(sorted_points,delta_diff):
    x2 = sorted_points[1][0]; y2 = sorted_points[1][1]
    x1 = sorted_points[0][0]; y1 = sorted_points[0][1]
    slope, intercept = get_line_eqn(x1,x2,y1,y2)
    sqrt_term = slope * math.log(1 / delta_diff)
    eps = intercept + slope + 2 * math.sqrt(sqrt_term)
    plot_lines = []
    for x,y in sorted_points:
        plot_lines.append([x,slope*x+intercept])
    return np.asarray(plot_lines),slope,intercept,eps

def get_line_plots_using_kneedle(sorted_points,x,y,delta_diff,delta=0.1):
    counter = 0
    for x_i,y_i,_ in sorted_points:
        if x_i == x and y_i==y:
            break
        counter +=1

    x2 = sorted_points[counter+1][0];y2 = sorted_points[counter+1][1]+delta

    slope, intercept = get_line_eqn(x, x2, y, y2)
    sqrt_term = slope * math.log(1/delta_diff)
    eps = intercept + slope + 2*math.sqrt(sqrt_term)
    plot_lines = []
    for x, y,_ in sorted_points:
        plot_lines.append([x, slope * x + intercept])
    return np.asarray(plot_lines), slope, intercept,eps
def get_line_plots_using_kneedle_nointercept(sorted_points,x,y):
    # x2 = sorted_points[counter+1][0];y2 = sorted_points[counter+1][1]+0.1
    slope, intercept = get_line_eqn_nointercept(x, y)
    plot_lines = []
    for x, y in sorted_points:
        plot_lines.append([x, slope * x + intercept])
    return np.asarray(plot_lines), slope, intercept

def getInpfeatures(featuresPath):
    features = np.load(featuresPath, allow_pickle=True)
    return features

def getRandomWithReplacement(inp_feat):
    import random
    inp_ret = []
    numSamplesretinp = []
    for i in range(inp_feat.shape[0]):
        numSamplesretinp.append(i)
    idxsSampleToUse = random.choices(numSamplesretinp,k=inp_feat.shape[0])
    for idx_inp in idxsSampleToUse:
        inp_ret.append(inp_feat[idx_inp])
    return inp_ret,idxsSampleToUse

def getVectoriserAndModel(inpDir):

    import pickle
    with open('{}/vectorizer.pickle'.format(inpDir), 'rb') as f:
        vectorizer = pickle.load(f)
    with open('{}/model.pickle'.format(inpDir), 'rb') as f:
        clf = pickle.load(f)
    return clf, vectorizer

if __name__ == '__main__':
    main_data_dir = "data/emebeddings_sent"
    alpha = 2
    map_vector_dirname = {"smart_amazon_utility_redaction_embs_not_norm": "smart_amazon_utility_redaction",
                          "smart_masking_redit_suicide_sent_embs": "smart_masking_redit_suicide",
                          "smartMaskingPoliticalDataset_sent_embs": "smartMaskingPoliticalDataset",
                          "smartMaskingValidationMedalSepTrain_diff_length_sent_embs": "smartMaskingValidationMedalSepTrain_diff_length"}

    json_dir = "data/divergence_alpha_vals"
    delta = 0.00008
    for dirName in map_vector_dirname:
        dataToPlot = {}
        model_dir = map_vector_dirname[dirName]
        json_path = json_dir+"/{}.json".format(dirName)

        with open(json_path) as json_file:
            data_json = json.load(json_file)

        mask_data = {}
        rhos = []
        for alpha in data_json:
            alpha_data = data_json[alpha]
            for data in alpha_data:
                redaction_percent = float(data[0]); divergence = float(data[1]); divergence_std = float(data[2])
                if redaction_percent not in mask_data:
                    mask_data[redaction_percent] = []
                mask_data[redaction_percent].append([float(alpha),divergence,divergence_std])
        data_plot_redaction = []
        if "amazon" in dirName:
            clf, vectoriser = getVectoriserAndModel("{}".format(model_dir))
        else:
            clf, vectoriser = getVectoriserAndModel("data/sentences/{}".format(model_dir))
        # for maskPerc in [90]:
        print(dirName)
        for key in mask_data:
            if key == 100:
                data_plot_redaction.append([key, 0,0.0,0.5,0.0])
                continue
            p_text = getInpfeatures(main_data_dir + "/" + dirName + "/{}/p_sent.npy".format(int(key)))
            q_text = getInpfeatures(main_data_dir + "/" + dirName + "/{}/q_sent.npy".format(int(key)))


            data_c = np.asarray(mask_data[key])
            kneedle = KneeLocator(data_c[:,0],data_c[:,1],curve="concave", direction="increasing")
            x1 = kneedle.knee
            y1 = kneedle.knee_y
            data_c_sorted  = sorted(data_c, key=lambda x: x[1])
            plot_lines, slope, intercept, eps = get_line_plots_using_kneedle(data_c_sorted, x1, y1, delta_diff=delta,
                                                                             delta=0.1)
            ypts = []
            for bootstap_idx in range(25):

                p_text_sampled, btstrap_p_idx = getRandomWithReplacement(p_text)
                q_text_sampled, btstrap_q_idx = getRandomWithReplacement(q_text)
                gt_n = [0 for _ in p_text_sampled]
                for temp_ in q_text_sampled:
                    gt_n.append(1)

                test_data = []
                test_data.extend(p_text_sampled)
                test_data.extend(q_text_sampled)
                test_data_transformed = vectoriser.transform(test_data)
                prediction_n = clf.predict(test_data_transformed)
                y_pt = roc_auc_score(gt_n, prediction_n)
                ypts.append(y_pt)
            data_plot_redaction.append([key, eps,0,np.mean(ypts),np.std(ypts)])

        data_to_plot = np.asarray(data_plot_redaction)
        plt.rcParams.update({'font.size': 15})
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.errorbar(data_to_plot[:, 0], data_to_plot[:, 1], data_to_plot[:,2],
                    label="EPS",color="b")
        ax2.errorbar(data_to_plot[:, 0], data_to_plot[:, 3], data_to_plot[:,4],
                    label="Attack Accuracy",color="r")
        ax.set_ylabel('EPS')
        ax.set_xlabel('WordsMasked %')
        # ax.set_xlabel('WordsMasked %')
        # ax.set_ylabel('Divergence')
        ax2.set_ylabel('Attack Accuracy')
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        lines, labels = ax.get_legend_handles_labels()
        plt.legend()
        plt.tight_layout()
        plt.show()
        # plt.savefig("new_eps_guarantee/{}_eps_values.eps".format(dirName), format="eps", bbox_inches='tight')
    pass
