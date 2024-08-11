
import argparse
import math
from statistics import mean
import os
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model_dp import RNNModel
from data_Dp import DatasetCSV_redcated,DatasetCSV
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='/home/vaibhav/ML/bartexps/smart_amazon_utility_redaction',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', "--learning_rate",type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument(
    "-c",
    "--max-per-sample-grad-norm",
    type=float,
    default=1.5,
    metavar="C",
    help="Clip per-sample gradients to this norm",
)
parser.add_argument(
    "--disable-dp",
    action="store_true",
    default=False,
    help="Disable privacy training and just train with vanilla SGD",
)
parser.add_argument(
    "--secure-rng",
    action="store_true",
    default=False,
    help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
)
parser.add_argument(
    "--delta",
    type=float,
    default=8e-5,
    metavar="D",
    help="Target delta",
)
parser.add_argument(
    "--print-every",
    type=int,
    default=5,
    help="Print the evaluation accuracy every these many iterations",
)

args = parser.parse_args()
# torch.manual_seed(args.seed)

###############################################################################
# Training code

# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def train(
    model,
    criterion,
    optimizer,
    train_loader,
    device="cuda:1",
):
    model.train()
    lr = args.lr
    losses = []
    for x, y in tqdm(train_loader):
        # print(x.size())
        model.zero_grad()
        x = torch.t(x)
        y = y.view(-1)
        x = x.to(device)
        y = y.to(device)
        output, hidden = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss))
    return model,mean(losses)


def evaluate(
    model,
    criterion,
    val_loader,
    device="cpu",
):
    #
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            x = torch.t(x)
            y = y.view(-1)
            x = x.to(device)
            y = y.to(device)
            output, hidden = model(x)
            loss = criterion(output, y)
            total_loss += len(x)*loss.item()

    return total_loss/ (len(val_loader.dataset)-1)


# def run_sgd_Redaction():



def main():

    args = parser.parse_args()
    device = torch.device("cuda:1")
    eval_batch_size = 10
    redact_dict = {}
    for mask_perc in range(0,101,10):
        loss_vals = []
        for seed in range(0, 5):
            torch.manual_seed(seed)
            print("*****mask_perc******* {}".format(mask_perc))
            train_dataset = DatasetCSV_redcated( args.data,batch_size=args.batch_size,device=device,bptt=args.bptt,train=True,maskperc=mask_perc)
            val_dataset = DatasetCSV_redcated( args.data,batch_size=eval_batch_size,device=device,bptt=args.bptt,train=False,maskperc=mask_perc)

            ntokens = len(train_dataset.dictionary)
            criterion = nn.NLLLoss()

            # model = model.to(device)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.bptt,
                num_workers=0,
                pin_memory=False,
                shuffle=True,
            )

            test_loader = DataLoader(
                val_dataset,
                batch_size=args.bptt,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            # torch.manual_seed(args.seed)

            model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.batch_size, args.dropout, args.tied).to(device)
            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            print("*" * 86)
            print(f"Train stats ({args.model}): \n")
            min_eval_loss = []
            for epoch in tqdm(range(args.epochs)):
                model, avg_train_loss = train(
                    model,
                    criterion,
                    optimizer,
                    train_loader,
                )

                test_loss = evaluate(
                    model,
                    criterion,
                    test_loader,
                    device="cuda:1",
                )
                min_eval_loss.append(test_loss)
                print(
                    "Epoch {} | Train loss {:.6f} | Validation loss {:.6f} | Validation ppl {:.6f} | Orignal data".format(
                        epoch, avg_train_loss, test_loss, math.exp(test_loss),
                    ))
            # redact_dict[mask_perc] = min(min_eval_loss)
            loss_vals.append(min(min_eval_loss))
            test_loss = evaluate(
                model,
                criterion,
                test_loader,
                device="cuda:1",
            )
            print(
                "Accuracy of trained model = {:.6f}| Orignal data".format(test_loss))
            print("*" *86)
        redact_dict[mask_perc] = [np.mean(loss_vals),np.std(loss_vals)]



    print("#####DP SGD #########")
    train_dataset = DatasetCSV(args.data, batch_size=args.batch_size, device=device, bptt=args.bptt, train=True)
    val_dataset = DatasetCSV(args.data, batch_size=eval_batch_size, device=device, bptt=args.bptt, train=False)

    ntokens = len(train_dataset.dictionary)
    # model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
    #                            args.tied).to(device)
    dp_sgd = {}
    for eps in [0.1,0.5,1.0,2.0,3.0,4.0,5.0,6.0,7.0,7.5,8.0,8.5]:
        loss_vals = []
        for seed in range(0, 5):
            torch.manual_seed(seed)
            criterion = nn.NLLLoss()

            # model = model.to(device)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.bptt,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
            )

            test_loader = DataLoader(
                val_dataset,
                batch_size=args.bptt,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            # torch.manual_seed(args.seed)
            model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.batch_size, args.dropout,
                             args.tied).to(device)
            model = model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            if eps != 0:
                privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    epochs=args.epochs,
                    target_epsilon=eps,
                    target_delta=args.delta,
                    max_grad_norm=args.max_per_sample_grad_norm,
                    batch_first=False
                )
            else:
                privacy_engine = None

            prev_loss = None
            print("*" * 86)
            print(f"Train stats ({args.model}): \n")
            for epoch in tqdm(range(args.epochs)):
                model_new, avg_train_loss = train(
                    model,
                    criterion,
                    optimizer,
                    train_loader,
                )

                test_loss = evaluate(
                    model,
                    criterion,
                    test_loader,
                    device="cuda:1",
                )
                if prev_loss is None:
                    prev_loss = test_loss
                    model = model_new
                else:
                    if prev_loss > test_loss:
                        prev_loss = test_loss
                        model = model_new
                if privacy_engine is not None:
                    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                    print(
                        "Epoch {} | Train loss {:.6f} | Validation loss {:.6f} | Validation ppl {:.6f} | Epsilon {:.6f} | Delta {:.6f}".format(
                            epoch, avg_train_loss, test_loss, test_loss, epsilon, args.delta
                        ))
                else:
                    # epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                    print(
                        "Epoch {} | Train loss {:.6f} | Validation loss {:.6f} | Validation ppl {:.6f} | Orignal data".format(
                            epoch, avg_train_loss, test_loss, test_loss,
                        ))
            test_loss = evaluate(
                model,
                criterion,
                test_loader,
                device="cuda:1",
            )
            loss_vals.append(test_loss)

            if privacy_engine is not None:
                epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
                print("Accuracy of trained model = {:.6f} with epsilon {:.6f}".format(test_loss, epsilon))
            else:
                print(
                    "Accuracy of trained model = {:.6f}| Orignal data".format(test_loss))
            print("*" * 86)
        dp_sgd[eps] = [np.mean(loss_vals),np.std(loss_vals)]

    print("**********Final data ********")
    print("Redaction ")
    print(redact_dict)
    print("DPSGD")
    print(dp_sgd)
    plot_fig(redact_dict,dp_sgd,args.data.split("/")[-1])

def plot_fig(redaction_dict,dp_sgd_dict,dataset_name):
    # print(key)
    if "amazon" not in dataset_name:
        loaded_divergence_value = np.load("data/npy_eps_data/{}_sent_embs.npy".format(dataset_name),allow_pickle=True)

    else:
        loaded_divergence_value = np.load("data/npy_eps_data/smart_amazon_utility_redaction_embs_not_norm.npy",
                                          allow_pickle=True)
    data_to_plot_redaction = []
    data_to_plot_dp_sgd = []
    original_data = []
    for data_row in loaded_divergence_value:
        mask_perc = data_row[0]
        mean_divergence = data_row[1]
        loss_value_for_mask_perc = redaction_dict[mask_perc][0]
        loss_std = redaction_dict[mask_perc][1]
        # eps_updated = convert_divergence_to_diff_priv(mean_divergence, delta=0.00008, alpha=2)
        data_to_plot_redaction.append([mean_divergence,loss_value_for_mask_perc,loss_std])
    data_to_plot_redaction.sort(key=lambda x: x[0])
    for eps in dp_sgd_dict:
        data_to_plot_dp_sgd.append([eps,dp_sgd_dict[eps][0],dp_sgd_dict[eps][1]])
    data_to_plot_dp_sgd = np.asarray(data_to_plot_dp_sgd)
    data_to_plot_redaction = np.asarray(data_to_plot_redaction)
    original_data = np.asarray(original_data)

    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.text(0.005, 0.5, r"log(ppl)", va="center", rotation="vertical")
    fig.subplots_adjust(hspace=0.01)  # adjust space between axes
    print(data_to_plot_dp_sgd)
    # plot the same data on both axes
    ax1.errorbar(data_to_plot_dp_sgd[:,0],data_to_plot_dp_sgd[:,1],data_to_plot_dp_sgd[:,2],label="DP-SGD finetuned",color="blue")
    ax2.errorbar(data_to_plot_redaction[:,0],data_to_plot_redaction[:,1],data_to_plot_redaction[:,2],label="Redaction finetuned",color="red")
    ax2.errorbar(original_data[:,0],original_data[:,1],original_data[:,2],label="Original-loss finetuned",color="green")
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.errorbar(data_to_plot_dp_sgd[1:,0],data_to_plot_dp_sgd[1:,1], data_to_plot_dp_sgd[1:,2],transform=ax1.transAxes, **kwargs)
    ax2.errorbar(data_to_plot_redaction[:,0],data_to_plot_redaction[:,1],data_to_plot_redaction[:,2], transform=ax2.transAxes, **kwargs)
    ax2.errorbar(original_data[:,0],original_data[:,1], original_data[:,2],transform=ax2.transAxes, **kwargs)
    # ax1.set_title("EPS V log(PPL) for {} data (seed = 1111)".format(key))
    # ax1.set_ylabel("log(ppl)")
    # ax2.set_ylabel("log(ppl)")
    ax2.set_xlabel("EPS (Differential Privacy)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.5, 1.25), loc='upper left')
    # lines = [p1, p2,p3]

    # ax1.legend(lines, [l.get_label() for l in lines])
    plt.tight_layout()
    # plt.legend()
    plt.show()


def collate_fn(data):
    x = torch.tensor(data[0]); y = torch.tensor(data[1])
    x = torch.t(x); y = torch.t(y)
    y = y.contiguous().view(-1)
    return x,y



if __name__ == "__main__":
    main()
