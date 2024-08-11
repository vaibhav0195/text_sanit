# coding: utf-8
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
from tqdm import tqdm
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='/home/vaibhav/ML/bartexps/smart_amazon_utility_redaction',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
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

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
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


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    modelInst.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = modelInst.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = modelInst(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = modelInst(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(modelInst):
    # Turn on training mode which enables dropout.
    modelInst.train()
    numParam = sum(p.numel() for p in modelInst.parameters() if p.requires_grad)
    # print("total parameters {}".format(numParam))
    hidden = modelInst.init_hidden(args.batch_size)
    data_new = []
    target_new = []
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        for row in range(data.size()[0]):
            data_new.append(data[row,:])
            target_new.append(targets[row,:])
    data_new = torch.stack(data_new)
    target_new = torch.stack(target_new).contiguous().view(-1)
    modelInst.zero_grad()

    hidden = repackage_hidden(hidden)
    output, hidden = modelInst(data_new, hidden)
    loss = criterion(output, target_new)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(modelInst.parameters(), args.clip)
    grad_norm = np.sqrt(sum([torch.norm(p.grad).cpu() ** 2 for p in modelInst.parameters()]))
    return grad_norm.item()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    modelInst.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = modelInst.init_hidden(batch_size)
    torch.onnx.export(modelInst, (dummy_input, hidden), path)


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

# dataToSave = []

datadir = args.data+"/{}".format(0)
taskName = args.data.split("/")[-1]
saveDir = taskName+"/{}".format(0)
os.makedirs(saveDir,exist_ok=True)
# savePath = saveDir+"/{}".forargs.save
# print(datadir,args.data)
corpus = data.CorpusRedacted(datadir,args.data)
orginal_doct = corpus.dictionary
eval_batch_size = 10
ntokens = len(corpus.dictionary)
modelInst = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
                           args.tied).to(device)
# torch.save(model.state_dict(), "model.pt")
with open('model.pt', 'wb') as f:
    torch.save(modelInst, f)
json_data = {}

for maskPerc in range(0,101,10):
    datadir = args.data+"/{}".format(maskPerc)
    taskName = args.data.split("/")[-1]
    saveDir = taskName+"/{}".format(maskPerc)
    os.makedirs(saveDir,exist_ok=True)
    # savePath = saveDir+"/{}".forargs.save
    print("***** mask {}".format(maskPerc))
    corpus = data.CorpusRedacted(datadir,args.data,original_dict=orginal_doct)

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    lr = args.lr
    best_val_loss = None
    grad_norms = []
    for _ in tqdm(range(500)):
        # modelInst = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
        #                            args.tied).to(device)
        # model.load_state_dict(torch.load("model.pt"))
        with open('model.pt', 'rb') as f:
            modelInst = torch.load(f)
        modelInst = modelInst.to(device)
        criterion = nn.NLLLoss()
        epoch_start_time = time.time()
        grad_norms.append(train(modelInst))
    json_data[maskPerc] = grad_norms
import json
with open('smart_amazon_utility_redaction_old.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)

