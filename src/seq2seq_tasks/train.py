import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy

import random
import math
import time

import models
from models import Attention, Seq2Seq, Encoder, Decoder, DecoderNoAttn, DecoderUniform
import utils
from utils import Language
from gen_utils import *
import numpy as np

import argparse
from tqdm import tqdm
import log
from data_utils import compute_frequencies, unkify_lines

# --------------- parse the flags etc ----------------- #
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--task', dest='task', default='copy',
        choices=('copy', 'rev', 'binary-flip', 'en-hi', 'en-de'),
        help = 'select the task you want to run on')

parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--loss-coef', dest='loss_coeff', type=float, default=0.0)
parser.add_argument('--epochs', dest='epochs', type=int, default=5)
parser.add_argument('--seed', dest='seed', type=int, default=1234)
parser.add_argument('--uniform', dest='uniform', action='store_true')
parser.add_argument('--no-attn', dest='no_attn', action='store_true')
parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
parser.add_argument('--num-train', dest='num_train', type=int, default=1000000)
parser.add_argument('--decode-with-no-attn', dest='no_attn_inference', action='store_true')



params = vars(parser.parse_args())
TASK = params['task']
DEBUG = params['debug']
COEFF = params['loss_coeff']
NUM_EPOCHS = params['epochs']
UNIFORM = params['uniform']
NO_ATTN = params['no_attn']
NUM_TRAIN = params['num_train']
DECODE_WITH_NO_ATTN = params['no_attn_inference']

INPUT_VOCAB = 10000
OUTPUT_VOCAB = 10000

long_type = torch.LongTensor
float_type = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    long_type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor



# The following function is not being used right now, and is deprecated.
def generate_mask(attn_shape, list_src_lens=None):
    trg_len, batch_size, src_len = attn_shape

    mask = torch.zeros(attn_shape).type(float_type)
    min_seq_len = min(trg_len, src_len)

    if TASK == 'copy':
        diag_items = torch.arange(min_seq_len)
        mask[diag_items, :, diag_items] = 1.0
    elif TASK == 'rev':
        assert list_src_lens is not None
        for b in range(batch_size):
            i = torch.arange(min_seq_len)
            j = torch.tensor([max(0, list_src_lens[b]- i - 1) for i in range(min_seq_len)])
            mask[i, b, j] = 1.0
    elif TASK == 'binary-flip':
        last = min_seq_len if min_seq_len % 2 == 1 else  min_seq_len - 1
        i = torch.tensor([i for i in range(1, last)])
        j = torch.tensor([i - 1 if i%2 == 0 else i + 1 for i in range(1, last)])
        mask[i, :, j] = 1.0
    elif TASK == 'en-hi':
        # english hindi, nothing as of now... will have a billingual dict later.
        pass
    else:
        raise ValueError("TASK can be one of copy, rev, binary-flip")        

    # make sure there are no impermissible tokens for first target
    mask[0, :, :] = 0.0 # the first target is free...
    mask[:, :, 0] = 0.0 # attention to sos is permissible
    mask[:, :, -1] = 0.0 # attention to eos is permissible

    return mask


def train(model, data, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    total_trg = 0.0
    # total_src = 0.0
    total_correct = 0.0
    total_attn_mass_imp = 0.0
    
    for src, src_len, trg, trg_len, alignment in tqdm(data):

        # create tensors here... 
        src = torch.tensor(src).type(long_type).permute(1, 0)
        trg = torch.tensor(trg).type(long_type).permute(1, 0)
        alignment = torch.tensor(alignment).type(float_type).permute(1, 0, 2)
        # algiment is not trg_len x batch_size x src_len

        optimizer.zero_grad()

        # print (f"source shape {src.shape}") 
        # print (f"source lens {src_len}")
        output, attention = model(src, src_len, trg)
        # attention is 

        mask = alignment # generate_mask(attention.shape, src_len)
        # mask shape trg_len x batch_size x src_len


        attn_mass_imp = torch.einsum('ijk,ijk->', attention, mask) 
        total_attn_mass_imp += attn_mass_imp

        # print (output.shape)
        
        #trg = [trg sent len, batch size]
        #output = [trg sent len, batch size, output dim]
        
        output = output[1:].contiguous().view(-1, output.shape[-1])
        # print (output.shape)
        preds = torch.argmax(output, dim=1) # long tensor 
        trg = trg[1:].contiguous().view(-1)
        
        #trg = [(trg sent len - 1) * batch size]
        #output = [(trg sent len - 1) * batch size, output dim]

        trg_non_pad_indices = (trg != utils.PAD_token)
        non_pad_tokens_trg = torch.sum(trg_non_pad_indices).item()
        # non_pad_tokens_src = torch.sum((src != utils.PAD_token)).item()

        total_trg += non_pad_tokens_trg # non pad tokens trg
        # total_src += non_pad_tokens_src # non pad tokens src
        total_correct += torch.sum((trg == preds) * trg_non_pad_indices).item()
        
        loss = criterion(output, trg) - COEFF * torch.log(1 - attn_mass_imp/non_pad_tokens_trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(data), 100. * total_correct/total_trg, \
        100. * total_attn_mass_imp/total_trg


def evaluate(model, data, criterion):
    
    model.eval()
    
    epoch_loss = 0
    total_correct = 0.0
    total_trg = 0.0
    # total_src = 0.0
    total_attn_mass_imp = 0.0
    
    with torch.no_grad():
    
        for src, src_len, trg, trg_len, alignment in tqdm(data):

            # create tensors here... 
            src = torch.tensor(src).type(long_type).permute(1, 0)
            trg = torch.tensor(trg).type(long_type).permute(1, 0)
            alignment = torch.tensor(alignment).type(float_type).permute(1, 0, 2)
            # algiment is not trg_len x batch_size x src_len

            # output, attention = model(src, src_len, None, 0) #turn off teacher forcing
            output, attention = model(src, src_len, trg, 0) #turn off teacher forcing
            # NOTE: it is not a bug to not count extra produce from the model, beyond target len
            
            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]            


            mask = alignment # generate_mask(attention.shape, src_len)
            # print ("Mask shape ", mask.shape)
            # mask shape trg_len x batch_size x src_len

            attn_mass_imp = torch.einsum('ijk,ijk->', attention, mask) 
            total_attn_mass_imp += attn_mass_imp


            output = output[1:].contiguous().view(-1, output.shape[-1])            
            trg = trg[1:].contiguous().view(-1)

            #trg = [(trg sent len - 1) * batch size]
            #output = [(trg sent len - 1) * batch size, output dim]

            preds = torch.argmax(output, dim=1) # long tensor 

            trg_non_pad_indices = (trg != utils.PAD_token)
            non_pad_tokens_trg = torch.sum(trg_non_pad_indices).item()
            # non_pad_tokens_src = torch.sum((src != utils.PAD_token)).item()

            total_trg += non_pad_tokens_trg # non pad tokens trg
            # total_src += non_pad_tokens_src # non pad tokens src

            total_correct += torch.sum((trg == preds) * trg_non_pad_indices).item()

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(data), 100. * total_correct/total_trg, \
        100. * total_attn_mass_imp/total_trg


def generate(model, data):
    
    # NOTE this assumes batch size 1
    model.eval()
    
    epoch_loss = 0
    total_correct = 0.0
    total_trg = 0.0
    # total_src = 0.0
    total_attn_mass_imp = 0.0
    generated_lines = []
    
    with torch.no_grad():
    
        for src, src_len, _, _, _ in tqdm(data):

            # create tensors here... 
            src = torch.tensor(src).type(long_type).permute(1, 0)
            # trg = torch.tensor(trg).type(long_type).permute(1, 0)

            # output, attention = model(src, src_len, None, 0) #turn off teacher forcing
            output, attention = model(src, src_len, None, 0) #turn off teacher forcg

            output = output[1:].squeeze(dim=1)
            #output = [(trg sent len - 1), output dim]

            preds = torch.argmax(output, dim=1) # long tensor 
            #shape [trg len - 1]
            generated_tokens = [trg_lang.get_word(w) for w in preds.cpu().numpy()]

            generated_lines.append(" ".join(generated_tokens))

    return generated_lines


SEED = params['seed']
BATCH_SIZE = params['batch_size']

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
models.set_seed(SEED)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



src_lang = Language('src')
trg_lang = Language('trg')


splits = ['train', 'dev', 'test']
sents = []


for sp in splits:
    src_filename = "./data/" + sp +  "." + TASK + ".src"
    trg_filename = "./data/" + sp +  "." + TASK + ".trg"

    src_sents = open(src_filename).readlines()
    trg_sents = open(trg_filename).readlines()

    alignment_filename = "./data/" + sp +  "." + TASK + ".align"

    alignment_sents = open(alignment_filename).readlines()

    if DEBUG: # small scale
        src_sents = src_sents[:int(1e5)]
        trg_sents = trg_sents[:int(1e5)]
        alignment_sents = alignment_sents[: int(1e5)]

    if sp == 'train':
        src_sents = src_sents[:NUM_TRAIN]
        trg_sents = trg_sents[:NUM_TRAIN]
        alignment_sents = alignment_sents[:NUM_TRAIN]

    sents.append([src_sents, trg_sents, alignment_sents])

train_sents = sents[0]


'''
train_src_sents = train_sents[0]
train_trg_sents = train_sents[1]
train_alignments = train_sents[2]
top_src_words = compute_frequencies(train_src_sents, INPUT_VOCAB)
top_trg_words = compute_frequencies(train_trg_sents, OUTPUT_VOCAB)

train_src_sents = unkify_lines(train_src_sents, top_src_words)
train_trg_sents = unkify_lines(train_trg_sents, top_trg_words)
train_sents = train_src_sents, train_trg_sents
'''

dev_sents = sents[1]
test_sents = sents[2]

def get_batches(src_sents, trg_sents, alignments, batch_size):

    # parallel should be at least equal len
    assert (len(src_sents) == len(trg_sents)) 

    for b_idx in range(0, len(src_sents), batch_size):

        # get the slice
        src_sample = src_sents[b_idx: b_idx + batch_size]
        trg_sample = trg_sents[b_idx: b_idx + batch_size]
        align_sample = alignments[b_idx: b_idx + batch_size]


        # represent them 
        src_sample = [src_lang.get_sent_rep(s) for s in src_sample] 
        trg_sample = [trg_lang.get_sent_rep(s) for s in trg_sample] 

        # sort by decreasing source len
        sorted_ids = sorted(enumerate(src_sample), reverse=True, key=lambda x: len(x[1]))
        src_sample = [src_sample[i] for i, v in sorted_ids]
        trg_sample = [trg_sample[i] for i, v in sorted_ids]
        align_sample = [align_sample[i] for i, v in sorted_ids]


        src_len = [len(s) for s in src_sample]
        trg_len = [len(t) for t in trg_sample]

        # largeset seq len 
        max_src_len = max(src_len)
        max_trg_len = max(trg_len)

        # pad the extra indices 
        src_sample = src_lang.pad_sequences(src_sample, max_src_len)
        trg_sample = trg_lang.pad_sequences(trg_sample, max_trg_len)

        # generata masks 
        aligned_outputs = []

        for alignment in align_sample:
            # print (alignment)
            current_alignment = np.zeros([max_trg_len, max_src_len])

            for pair in alignment.strip().split():
                src_i, trg_j = pair.split("-")
                src_i = min(int(src_i) + 1, max_src_len-1)
                trg_j = min(int(trg_j) + 1, max_trg_len-1)
                current_alignment[trg_j][src_i] = 1

            aligned_outputs.append(current_alignment)


        # numpy them 
        src_sample = np.array(src_sample, dtype=np.int64)
        trg_sample = np.array(trg_sample, dtype=np.int64)
        aligned_outputs = np.array(aligned_outputs)
        # align output is batch_size x max target_len x max_src_len

        assert (src_sample.shape[1] == max_src_len)

        yield src_sample, src_len, trg_sample, trg_len, aligned_outputs




train_batches = list(get_batches(train_sents[0], train_sents[1], train_sents[2], BATCH_SIZE))
src_lang.stop_accepting_new_words()
trg_lang.stop_accepting_new_words()
dev_batches = list(get_batches(dev_sents[0], dev_sents[1], dev_sents[2], BATCH_SIZE))
test_batches = list(get_batches(test_sents[0], test_sents[1], test_sents[2], BATCH_SIZE))


# --------------------------------------------------------#
# ------------------- define the model -------------------#
# --------------------------------------------------------#
INPUT_DIM = src_lang.get_vocab_size()
OUTPUT_DIM = trg_lang.get_vocab_size()
print (f"Input vocab {INPUT_DIM} and output vocab {OUTPUT_DIM}")
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
PAD_IDX = utils.PAD_token
SOS_IDX = utils.SOS_token
EOS_IDX = utils.EOS_token
SUFFIX = ""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

if UNIFORM: 
    dec = DecoderUniform(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    SUFFIX = "_uniform"
elif NO_ATTN or DECODE_WITH_NO_ATTN:
    dec = DecoderNoAttn(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    if NO_ATTN: 
        SUFFIX = "_no-attn"
else:
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)


model = Seq2Seq(enc, dec, PAD_IDX, SOS_IDX, EOS_IDX, device).to(device)
# init weights 
model.apply(init_weights)
# count the params 
print(f'The model has {count_parameters(model):,} trainable parameters')
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
# --------- end of model definiition --------- #


# NUM_EPOCHS = 5

CLIP = 1

best_valid_loss = float('inf')
convergence_time = 0.0
epochs_taken_to_converge = 0

no_improvement_last_time = False 

for epoch in range(NUM_EPOCHS):
    
    start_time = time.time()
    
    train_loss, train_acc, train_attn_mass = train(model, train_batches, 
                                                optimizer, criterion, CLIP)
    valid_loss, val_acc, val_attn_mass  = evaluate(model, dev_batches, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), \
            'data/models/model_' + TASK + SUFFIX + '_seed=' + str(SEED) + '_coeff=' \
            + str(COEFF) + '_num-train=' + str(NUM_TRAIN) + '.pt') 
        epochs_taken_to_converge = epoch + 1
        convergence_time += (end_time - start_time)
        no_improvement_last_time = False
    else:
        # no improvement this time
        if no_improvement_last_time:
            break
        no_improvement_last_time = True 

    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc:0.2f} \
        | Train Attn Mass: {train_attn_mass:0.2f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |   Val Acc: {val_acc:0.2f} \
        |  Val. Attn Mass: {val_attn_mass:0.2f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

# load the best model and print stats:
model.load_state_dict(torch.load('data/models/model_' + TASK + SUFFIX + \
    '_seed=' + str(SEED) + '_coeff=' + str(COEFF) + '_num-train=' + str(NUM_TRAIN) + '.pt'))


test_loss, test_acc, test_attn_mass  = evaluate(model, test_batches, criterion)
print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc:0.2f} \
        |  Test Attn Mass: {test_attn_mass:0.2f} |  Test PPL: {math.exp(test_loss):7.3f}')

log.pr_green (f"Final Test Accuracy ..........\t{test_acc:0.2f}")
log.pr_green (f"Final Test Attention Mass ....\t{test_attn_mass:0.2f}")
log.pr_green (f"Convergence time in seconds ..\t{convergence_time:0.2f}")
log.pr_green (f"Sample efficiency in epochs ..\t{epochs_taken_to_converge}")


src_lang.save_vocab("data/vocab/" + TASK + SUFFIX + '_seed=' + str(SEED) \
     + '_coeff=' + str(COEFF) +  '_num-train=' + str(NUM_TRAIN) + ".src.vocab")
trg_lang.save_vocab("data/vocab/" + TASK + SUFFIX + '_seed=' + str(SEED) \
     + '_coeff=' + str(COEFF) +  '_num-train=' + str(NUM_TRAIN) + ".trg.vocab")


if TASK in ['en-hi', 'en-de']:
    # generate the output to copmute bleu scores as well...
    log.pr_green("generating the output translations from the model")
    test_batches_single = list(get_batches(test_sents[0], test_sents[1], test_sents[2], 1))
    output_lines = generate(model, test_batches_single)
    log.pr_green("[done] .... now dumping the translations")
    outfile = "data/" + TASK + SUFFIX + "_seed" + str(SEED) + \
          '_coeff=' + str(COEFF) +  '_num-train=' + str(NUM_TRAIN) + ".test.out"
    fw = open(outfile, 'w')
    for line in output_lines:
        fw.write(line.strip() + "\n")
    fw.close()
