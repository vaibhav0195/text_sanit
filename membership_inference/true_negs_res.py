# coding: utf-8
import scipy
import math
import random
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import data
import model
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='/home/vaibhav/ML/bartexps/smart_amazon_utility_redaction',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=60,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=2,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
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
args = parser.parse_args()

def get_sents_targets_batchified(examples_sen_batched,masked_idx,segerate_mask=True):
    data_new = [];target_new = []
    for batch, i in enumerate(range(0, examples_sen_batched.size(0) - 1, args.bptt)):

        data, targets = get_batch(examples_sen_batched, i)
        if data.size()[0] != args.bptt:
            continue
        for idx_t, d_t in enumerate(range(data.size()[1])):
            tensor_inp = data[:, idx_t]
            tensor_op = data[:,idx_t]
            if not segerate_mask:
                data_new.append(data[:, idx_t])
                target_new.append(targets[:, idx_t])
            else:
                if masked_idx in tensor_inp or masked_idx in tensor_op:
                    data_new.append(data[:, idx_t])
                    target_new.append(targets[:, idx_t])
    return data_new,target_new


def get_sents_targets_batchified_original(examples_sen_batched,check_sents,masked_idx,segerate_mask=True):
    data_new = [];target_new = []
    for batch, i in enumerate(range(0, examples_sen_batched.size(0) - 1, args.bptt)):
        data, targets = get_batch(examples_sen_batched, i)
        data_c, targets_c = get_batch(check_sents, i)
        if data.size()[0] != args.bptt:
            continue
        for idx_t, d_t in enumerate(range(data.size()[1])):
            tensor_inp = data_c[:, idx_t]
            tensor_op = targets_c[:,idx_t]
            if not segerate_mask:
                data_new.append(data[:, idx_t])
                target_new.append(targets[:, idx_t])
            else:
                if masked_idx in tensor_inp or masked_idx in tensor_op:
                    data_new.append(data[:, idx_t])
                    target_new.append(targets[:, idx_t])
    return data_new,target_new

def p_value_DP_audit(m, r, v, eps, delta):
    # m = number of examples, each included independently with probability 0.5
    # r = number of guesses (i.e. excluding abstentions)
    # v = number of correct guesses by auditor
    # eps,delta = DP guarantee of null hypothesis
    # output: p-value = probability of >=v correct guesses under null hypothesis
    assert 0 <= v <= r <= m # trhos error at 60 redaction
    assert eps >= 0
    assert 0 <= delta <= 1
    q = 1/(1+math.exp(-eps)) # accuracy of eps-DP randomized response
    beta = scipy.stats.binom.sf(v-1, r, q) # = P[Binomial(r, q) >= v]
    alpha = 0
    sum = 0 # = P[v > Binomial(r, q) >= v - i]
    for i in range(1, v + 1):
        sum = sum + scipy.stats.binom.pmf(v - i, r, q)
        if sum > i * alpha:
            alpha = sum / i
    p = beta + alpha * delta * 2 * m
    return min(p, 1)

def get_eps_audit(m, r, v, delta, p):
    # m = number of examples, each included independently with probability 0.5
    # r = number of guesses (i.e. excluding abstentions)
    # v = number of correct guesses by auditor
    # p = 1-confidence e.g. p=0.05 corresponds to 95%
    # output: lower bound on eps i.e. algorithm is not (eps,delta)-DP
    assert 0 <= v <= r <= m
    assert 0 <= delta <= 1
    assert 0 < p < 1
    eps_min = 0 # maintain p_value_DP(eps_min) < p
    eps_max = 10 # maintain p_value_DP(eps_max) >= p
    while p_value_DP_audit(m, r, v, eps_max, delta) < p: eps_max = eps_max + 1
    # print(eps_max)
    for _ in range(30): # binary search
        eps = (eps_min + eps_max) / 2
        # print(p_value_DP_audit(m, r, v, eps, delta))
        if p_value_DP_audit(m, r, v, eps, delta) < p:
            # print(p_value_DP_audit(m, r, v, eps, delta))
            eps_min = eps
        else:
            eps_max = eps
    # print(p_value_DP_audit(m, r, v, eps_min, delta))
    print(eps_max)
    return eps_min

def audit_black_box_thresh(modelInst,examples_sen,criterion,examples_original_val,masked_idx,segerate_mask):
    """
    params are taken for algorithm2 from https://proceedings.neurips.cc/paper_files/paper/2023/file/9a6f6e0d6781d1cb8689192408946d73-Paper-Conference.pdf
    :param model: model to perform blackbox attack against
    :param num_pos: k+
    :param num_neg: k_
    :param num_examples: number of examples to randomise
    :param examples: actual examples having m canaries and n auditing examples
    :param loss_for_black_box: actual examples having m canaries and n auditing examples
    :return:
    """
    # some data processing
    examples_sen_batched = batchify(examples_sen, args.batch_size)
    # examples_safe_batched = batchify(examples_safe, args.batch_size)
    examples_original_sensitive_batched_masked = batchify(examples_original_val.valid, args.batch_size)
    examples_original_sensitive_batched = batchify(examples_original_val.train, args.batch_size)

    data_new = [];target_new = []
    data_new_val = [];target_new_val = []; gtinp = [] # gtinp == whether the validation set is same or not.

    data_new_sen, target_new_sen = get_sents_targets_batchified(examples_sen_batched,masked_idx,segerate_mask=segerate_mask)
    data_new.extend(data_new_sen)
    target_new.extend(target_new_sen)

    idxs_used = [i for i in range(len(data_new))]
    random.shuffle(idxs_used)
    for idxtemp in idxs_used[:500]:
        data_new_val.append(data_new[idxtemp])
        target_new_val.append(target_new[idxtemp])
        gtinp.append(1)
    # if seg_mask:
    data_original, target_new_original = get_sents_targets_batchified_original(examples_original_sensitive_batched,
                                                                               check_sents=examples_original_sensitive_batched_masked,
                                                                               masked_idx=masked_idx,
                                                                               segerate_mask=False)
    # else:
        # examples_original_sensitive_batched =
        # data_original, target_new_original = get_sents_targets_batchified(examples_sen_batched_val,masked_idx,segerate_mask=False)
    # print(len(data_original))
    idx_data_original = [_ for _ in range(len(data_original))]
    # print(len(idx_data_original))
    slicing_tn = len(idx_data_original)
    random.shuffle(idx_data_original)
    for idx in idx_data_original[:500]:
        data_new_val.append(data_original[idx])
        target_new_val.append(target_new_original[idx])
        gtinp.append(-1)

    #train model
    X_in = torch.stack(data_new,dim=0)
    target_in = torch.stack(target_new,dim=0)
    modelInst,min_tr_loss,max_tr_loss = train_for_epochs(model_i=modelInst,train_data=X_in,target_data=target_in)
    #trained model.
    modelInst.eval()
    scores = []
    data_new_val_tensor = torch.stack(data_new_val,dim=0)
    target_test_inp_tensor = torch.stack(target_new_val,dim=0)
    eval_loss = evaluate(modelInst, data_new_val_tensor, target_test_inp_tensor, criterion)

    for i in range(len(data_new_val)):
        modelInst.zero_grad()
        x_inp_test = data_new_val[i]
        x_inp_test = x_inp_test.view(args.bptt,1)
        target_test_inp = target_new_val[i]
        output, hidden = modelInst(x_inp_test, None)
        loss = criterion(output, target_test_inp.contiguous().view(-1))
        if loss.item()<= eval_loss :
            scores.append(1)
        else:
            scores.append(-1)

    print(len(scores[-slicing_tn:]),len(gtinp[-slicing_tn:]))
    fp=0;fn=0;tp=0;tn=0;correct_prediction=0;m=0;n=0;num_pos=0
    for idx,gt_val in enumerate(gtinp[-slicing_tn:]):
        # print(idx,500+idx)
        pred_val = scores[-slicing_tn:][idx]
        # print(gt_val,pred_val)
        if pred_val == -1 and gt_val==1:
            fn +=1
        elif pred_val == 1 and gt_val ==-1:
            fp+=1
        elif pred_val ==1 and gt_val==1:
            tp+=1
        elif pred_val == -1 and gt_val ==-1:
            tn +=1
    return fp,fn,tp,tn,slicing_tn

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def batchify_sensitive_original(data,data_sen, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target

def evaluate(modelInst,train_data_sensitive,target_inp,criterion):
    # Turn on evaluation mode which disables dropout.
    modelInst.eval()
    total_loss = 0.
    ntokens = len(orginal_dict)
    tot = 0
    with torch.no_grad():
        for i in range(0, train_data_sensitive.size(0) - args.batch_size, args.batch_size):
            # print(i)
            data = train_data_sensitive[i:i + args.batch_size, :]
            target = target_inp[i:i + args.batch_size, :]
            data = torch.t(data)
            targets = target.contiguous().view(-1)
            output, hidden = modelInst(data, None)
            # hidden = repackage_hidden(hidden)
            tot += data.size()[0]
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (tot - 1)


def train_for_epochs(model_i,train_data,target_data):
    lr = args.lr
    criterion = nn.NLLLoss()
    losses = []
    for epoch in range(1, args.epochs + 1):
        model_i,avg_tr_loss = train(model_i, train_data,target_data,lr,criterion)
        losses.extend(avg_tr_loss)
    return model_i,min(losses),max(losses)

def train(modelInst_sen,train_data_sensitive,target_inp,lr,criterion):
    # Turn on training mode which enables dropout.
    modelInst_sen.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(orginal_dict)
    avg_loss = []
    # print(train_data_sensitive.size())
    # print(target_inp.size())
    for i in range(0,train_data_sensitive.size(0)-args.batch_size,args.batch_size):
        # print(i)
        data = train_data_sensitive[i:i+args.batch_size,:]
        target = target_inp[i:i+args.batch_size,:]
        data = torch.t(data)
        target = target.contiguous().view(-1)
        # print(data.size())
        modelInst_sen.zero_grad()
        output, hidden = modelInst_sen(data, None)
        loss = criterion(output, target)
        loss.backward()
        for p in modelInst_sen.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        avg_loss.append(loss.item())
    return modelInst_sen,avg_loss

def convert_divergence_to_diff_priv(renyi_divergence,delta,alpha=2):

    rho_temp = renyi_divergence/alpha
    sqrt_term = rho_temp * math.log(1/delta)
    epsilon = rho_temp + 2 * math.sqrt(sqrt_term)
    return epsilon
# Set the random seed manually for reproducibility.

device = torch.device("cuda:1" if args.cuda else "cpu")

###############################################################################
# Load data and create dictionary
###############################################################################

datadir = args.data+"/{}".format(0)
taskName = args.data.split("/")[-1]
corpus_temp = data.CorpusRedacted(datadir,args.data)
orginal_dict = corpus_temp.dictionary
ntokens = len(orginal_dict)
redaction_to_eps_amazon = {
        0:2.11,10:1.29,20:1.15,30:0.93,40:0.84,50:0.78,60:0.72,70:0.72,80:0.69,90:0.51,100:0.0
    }
mask_idx = ntokens-1 # mask idx is the last one double checked
json_data = {}
# for the mask percentage
for maskPerc in range(0,91,10):
    # json_data_util = {}
    eval_batch_size =10
    datadir = args.data+"/{}".format(maskPerc)
    corpus_sensitive = data.CorpusRedacted(datadir,args.data,original_dict=orginal_dict,sensitive=True)
    # datadir_orignal = args.data+"/{}".format(0)
    corpus_sensitive_original = data.CorpusRedacted_original(datadir,args.data,original_dict=orginal_dict,sensitive=True)
    corpus_safe = data.CorpusRedacted(datadir,args.data,original_dict=orginal_dict,sensitive=False)
    redact_renyi_eps = redaction_to_eps_amazon[maskPerc]
    # eps_redact = convert_divergence_to_diff_priv(redact_renyi_eps,delta=8e-5,alpha=2)
    eps_redact = 100
    # num_pos = 500
    # num_neg = 500
    eps_mins = []
    if maskPerc == 0:
        seg_mask = False
    else:
        seg_mask = True
    for seed in range(0,10):
        # json_data_util = {'sen':{},'safe':{}}
        torch.manual_seed(seed)

        print("***** mask {}".format(maskPerc))
        model_i = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
                                   args.tied).to(device)

        fp,fn,tp,tn,slicing_tn = audit_black_box_thresh(modelInst=model_i, examples_sen=corpus_sensitive.train, criterion=nn.NLLLoss()
                               , examples_original_val=corpus_sensitive_original,masked_idx=mask_idx,segerate_mask=seg_mask)
        delta = 8e-5
        # eps = max([math.log((1-delta-fp)/fn),math.log((1-delta-fn)/fp)])
        print(["fp,fn,tp,tn,slicing_tn"])
        print([fp,fn,tp,tn,slicing_tn])
        # print("result of test")
        # print("canaries included {}, TP = {}, canaries not included = {},TN = {}".format(canaries_included_1,tp,canaries_included_0,tn))
        eps_mins.append([fp,fn,tp,tn,slicing_tn])
    json_data[maskPerc] = eps_mins

import json

with open('json_data/{}.json'.format(taskName), 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
