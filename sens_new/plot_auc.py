import math
import matplotlib.pyplot as plt
import numpy as np
import json

def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)

def convert_divergence_to_diff_priv(renyi_divergence,delta,alpha=2):

    rho_temp = renyi_divergence/alpha
    sqrt_term = rho_temp * math.log(1/delta)
    epsilon = rho_temp + 2 * math.sqrt(sqrt_term)
    return epsilon

def get_spread(list_of_values,original_mean=None):
    arrayofvalues = np.asarray(list_of_values)

    # arrayofvalues = (arrayofvalues - arrayofvalues.min()) / (arrayofvalues.max() - arrayofvalues.min())
    # arrayofvalues = arrayofvalues.round(decimals=5)
    if original_mean is None:
        mean = np.mean(arrayofvalues)
    else:
        mean = original_mean
    ret_list = [data-mean for data in arrayofvalues]
    return ret_list,mean

if __name__ == '__main__':
    redaction_to_eps_amazon = {
        0:2.11,10:1.29,20:1.15,30:0.93,40:0.84,50:0.78,60:0.72,70:0.72,80:0.69,90:0.51,100:0.0
    }
    json_filename = "auc_log_reg.json"
    # metric_keys = ["Procrustes"]
    data_to_read = js_r(json_filename)
    layer_data = {}
    for redaction in range(0,101,10):
        for layer_name in data_to_read[str(redaction)]:
            auc_layer = data_to_read[str(redaction)][layer_name]
            if layer_name not in layer_data:
                layer_data[layer_name] = []
            layer_data[layer_name].append([redaction,auc_layer])
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    for layer_name in layer_data:
        layer_data_plot = np.asarray(layer_data[layer_name])
        ax.plot(layer_data_plot[:, 0], layer_data_plot[:, 1],
                    label=layer_name)

    plt.title("AUC layerwise v redaction")
    ax.set_ylabel('AUC')
    ax.set_xlabel('redaction')
    lines, labels = ax.get_legend_handles_labels()
    plt.legend()
    plt.tight_layout()
    plt.show()
