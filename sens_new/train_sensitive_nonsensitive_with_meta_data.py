# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import data
import model
import json

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

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate_sen(data_source):
    # Turn on evaluation mode which disables dropout.
    modelInst_sen.eval()
    total_loss = 0.
    ntokens = len(orginal_dict)
    if args.model != 'Transformer':
        hidden = modelInst_sen.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = modelInst_sen(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = modelInst_sen(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion_sen(output, targets).item()
    return total_loss / (len(data_source) - 1)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    modelInst_safe.eval()
    total_loss = 0.
    ntokens = len(orginal_dict)
    if args.model != 'Transformer':
        hidden = modelInst_safe.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = modelInst_safe(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = modelInst_safe(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion_safe(output, targets).item()
    return total_loss / (len(data_source) - 1)

def train_sensitive():
    # Turn on training mode which enables dropout.
    modelInst_sen.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(orginal_dict)
    # if args.model != 'Transformer':
    #     hidden = modelInst_sen.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data_sensitive.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data_sensitive, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # print(data.size())
        # print(targets.size())
        modelInst_sen.zero_grad()
        if args.model == 'Transformer':
            output = modelInst_sen(data)
            output = output.view(-1, ntokens)
        else:
            # hidden = repackage_hidden(hidden)
            output, hidden = modelInst_sen(data, None)
        loss = criterion_sen(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(modelInst_sen.parameters(), args.clip)
        for p in modelInst_sen.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data_sensitive) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

def train_safe():
    # Turn on training mode which enables dropout.
    modelInst_safe.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(orginal_dict)
    # if args.model != 'Transformer':
        # hidden = modelInst_safe.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data_safe.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data_safe, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        modelInst_safe.zero_grad()
        if args.model == 'Transformer':
            output = modelInst_safe(data)
            output = output.view(-1, ntokens)
        else:
            # hidden = repackage_hidden(hidden)
            output, hidden = modelInst_safe(data, None)
        loss = criterion_safe(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(modelInst_safe.parameters(), args.clip)

        for p in modelInst_safe.parameters():
            print(p,p.grad)

        for p in modelInst_safe.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data_safe) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

# Set the random seed manually for reproducibility.

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data and create dictionary
###############################################################################

datadir = args.data+"/{}".format(0)
taskName = args.data.split("/")[-1]
corpus_temp = data.CorpusRedacted(datadir,args.data)
orginal_dict = corpus_temp.dictionary
ntokens = len(orginal_dict)

# for the mask percentage
for maskPerc in range(0,101,10):
    # json_data_util = {}
    eval_batch_size =10
    datadir = args.data+"/{}".format(maskPerc)
    corpus_sensitive = data.CorpusRedacted(datadir,args.data,original_dict=orginal_dict,sensitive=True)
    corpus_safe = data.CorpusRedacted(datadir,args.data,original_dict=orginal_dict,sensitive=False)
    # print(corpus_sensitive.trainsent)
    # print('##########################')
    # print(print(corpus_safe.trainsent))
    train_data_sensitive = batchify(corpus_sensitive.train, args.batch_size)
    val_data_sensitive = batchify(corpus_sensitive.valid, eval_batch_size)
    train_data_safe = batchify(corpus_safe.train, args.batch_size)
    val_data_safe = batchify(corpus_safe.valid, eval_batch_size)

    for seed in range(0,50):
        # json_data_util = {'sen':{},'safe':{}}
        torch.manual_seed(seed)
        # save_dir_model = "trained_model_no_clip/{}_{}/{}".format( taskName,maskPerc,seed)
        # os.makedirs(save_dir_model, exist_ok=True)
        # savePath_sen = save_dir_model + "/model_sen.pt"
        # savePath_safe = save_dir_model + "/model_safe.pt"

        print("***** mask {}".format(maskPerc))
        model_i = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout,
                                   args.tied).to(device)
        with open('model_pttem.pt', 'wb') as f:
            torch.save(model_i, f)

        with open('model_pttem.pt', 'rb') as f:
            modelInst_sen = torch.load(f)
        with open('model_pttem.pt', 'rb') as f:
            modelInst_safe = torch.load(f)

        lr = args.lr
        criterion_sen = nn.NLLLoss()
        criterion_safe = nn.NLLLoss()

        best_val_loss_sen = None
        best_val_loss_safe= None
        # json_path = "trained_model_no_clip/{}_{}/{}/seed_epoch_result.json".format(taskName, maskPerc, seed)
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()
                train_sensitive()
                train_safe()
                val_loss_sen = evaluate_sen(val_data_sensitive)
                val_loss_safe = evaluate(val_data_safe)
                print('-' * 89)
                print('sensitive True| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss_sen, math.exp(val_loss_sen)))
                print('sensitive False| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss_safe, math.exp(val_loss_safe)))
                print('-' * 89)
                # json_data_util['sen'][epoch] = val_loss_sen
                # json_data_util['safe'][epoch] = val_loss_safe
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss_sen or val_loss_sen < best_val_loss_sen:
                    # with open(savePath_sen, 'wb') as f:
                    #     torch.save(modelInst_sen, f)
                    best_val_loss_sen = val_loss_sen
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss_safe or val_loss_safe < best_val_loss_safe:
                    # with open(savePath_safe, 'wb') as f:
                    #     torch.save(modelInst_safe, f)
                    best_val_loss_safe = val_loss_safe
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
            # with open('{}.json'.format(json_path), 'w', encoding='utf-8') as f:
            #     json.dump(json_data_util, f, ensure_ascii=False, indent=4)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

