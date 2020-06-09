import numpy as np
from numpy import linalg as LA
import argparse
from tqdm import tqdm
from collections import defaultdict
import random
import time
from time import sleep
from tabulate import tabulate

import torch
import torch.nn as nn
from models import EmbAttModel, BiLSTMAttModel, BiLSTMModel
import pickle

import log
import util


# parsing stuff from the command line
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--emb-size', dest='emb_size', type=int, default=128,
        help = 'number of dimensions for the embedding layer')

parser.add_argument('--hid-size', dest='hid_size', type=int, default=64,
        help = 'size of the hidden dimension')

parser.add_argument('--model', dest='model', default='emb-att',
        choices=('emb-att', 'emb-lstm-att', 'no-att-only-lstm'),
        help = 'select the model you want to run')

parser.add_argument('--task', dest='task', default='pronoun',
        choices=('pronoun', 'sst', 'sst-wiki', 'sst-wiki-unshuff', 'reco', 'reco-rank', 'de-pronoun', 'de-refs', 'de-sst-wiki', 'occupation-classification', 'de-occupation-classification', 'occupation-classification_all'),
        help = 'select the task you want to run on')

parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=5,
        help = 'number of epochs')

parser.add_argument('--num-visualize', dest='num_vis', type=int, default=5,
        help = 'number of examples to visualize')

parser.add_argument('--loss-entropy', dest='loss_entropy', type=float, default=0.,
        help = 'strength for entropy loss on attention weights')

parser.add_argument('--loss-hammer', dest='loss_hammer', type=float, default=0.,
        help = 'strength for hammer loss on attention weights')

parser.add_argument('--loss-kld', dest='loss_kld', type=float, default=0.,
        help = 'strength for KL Divergence Loss on attention weights')

parser.add_argument('--top', dest='top', type=int, default=3,
        help = 'how many of the most attended words to ignore (default is 3)')

parser.add_argument('--seed', dest='seed', type=int, default=1,
        help = 'set random seed, defualt = 1')

# flags specifying whether to use the block and attn file or not
parser.add_argument('--use-attn-file', dest='use_attn_file', action='store_true')

parser.add_argument('--use-block-file', dest='use_block_file', action='store_true')

parser.add_argument('--block-words', dest='block_words', nargs='+', default=None,
        help = 'list of words you wish to block (default is None)')

parser.add_argument('--dump-attn', dest='dump_attn', action='store_true')

parser.add_argument('--use-loss', dest='use_loss', action='store_true')

parser.add_argument('--anon', dest='anon', action='store_true')

parser.add_argument('--debug', dest='debug', action='store_true')

parser.add_argument('--understand', dest='understand', action='store_true')

parser.add_argument('--flow', dest='flow', action='store_true')

parser.add_argument('--clip-vocab', dest='clip_vocab', action='store_true')

parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=20000,
        help='in case you clip vocab, specify the vocab size')

params = vars(parser.parse_args())

# useful constants
SEED = params['seed']

# user specified constants
C_ENTROPY = params['loss_entropy']
C_HAMMER = params['loss_hammer']
C_KLD = params['loss_kld']
NUM_VIS = params['num_vis']
NUM_EPOCHS = params['num_epochs']
EMB_SIZE = params['emb_size']
HID_SIZE = params['hid_size']
EPSILON = 1e-12
TO_ANON = params['anon']
TO_DUMP_ATTN = params['dump_attn']
BLOCK_TOP = params['top']
BLOCK_WORDS = params['block_words']
USE_ATTN_FILE = params['use_attn_file']
USE_BLOCK_FILE = params['use_block_file']

MODEL_TYPE = params['model']
TASK_NAME = params['task']
USE_LOSS = params['use_loss']
DEBUG = params['debug']
UNDERSTAND = params['understand']
FLOW = params['flow']
CLIP_VOCAB = params['clip_vocab']
VOCAB_SIZE = params['vocab_size']

# print useful info
log.pr_blue("Task: %s" %(TASK_NAME))
log.pr_blue("Model: %s" %(MODEL_TYPE))
log.pr_blue("Coef (hammer): %0.2f" %(C_HAMMER))
log.pr_blue("Coef (random-entropy): %0.2f" %(C_ENTROPY))

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

w2i = defaultdict(lambda: len(w2i))
w2c = defaultdict(lambda: 0.0) # word to count
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]

# gender_tokens = ["he", "she", "her", "his", "him"]

def read_dataset(data_file, block_words=None, block_file=None, attn_file=None, clip_vocab=False):

    data_lines = open(data_file).readlines()
    global w2i

    if clip_vocab:
        for line in data_lines:
            tag, words = line.strip().lower().split("\t")

            for word in words.split():
                w2c[word] += 1.0
            
        # take only top VOCAB_SIZE words
        word_freq_list = sorted(w2c.items(), key=lambda x: x[1], reverse=True)[:VOCAB_SIZE - len(w2i)]

        for idx, (word, freq) in enumerate(word_freq_list):
            temp = w2i[word] # assign the next available idx
    
        w2i = defaultdict(lambda: UNK, w2i)

    if block_file is not None:
        block_lines = open(block_file).readlines()
        if len(data_lines) != len(block_lines):
            raise ValueError("num lines in data file does not match w/ block file")

    if attn_file is not None:
        attn_lines = open(attn_file).readlines()
        if len(data_lines) != len(attn_lines):
            raise ValueError("num lines in data file does not match w/ attn file")

    for idx, data_line in enumerate(data_lines):
        tag, words = data_line.strip().lower().split("\t")
        if TO_ANON:
            words = util.anonymize(words)

        # populate block ids
        words = words.strip().split()
        block_ids = [0 for _ in words]
        attn_wts = None
        if block_words is not None:
            block_ids = [1 if i in block_words else 0 for i in words]
        elif block_file is not None:
            block_ids = [int(i) for i in block_lines[idx].strip().split()]

        if attn_file is not None:
            attn_wts = [float(i) for i in attn_lines[idx].strip().split()]
            # neglect_top = max(0, min(BLOCK_TOP, len(words) - 1))
            # top_ids = np.argsort(neg_attn_wts)[: neglect_top]
            # for i in top_ids:
            #     block_ids[i] = 1

        # check for the right len
        if len(block_ids) != len(words):
            raise ValueError("num of block words not equal to words")
        # done populating
        yield (idx, [w2i[x] for x in words], block_ids, attn_wts, t2i[tag])

def quantify_attention(ix, p, block_ids):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    base_prop = len(sent_keyword_idxs) / len(ix)
    att_prop = sum([p[i] for i in sent_keyword_idxs])
    return base_prop, att_prop

def quantify_norms(ix, word_embeddings, block_ids):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    base_ratio = len(sent_keyword_idxs) / len(ix)
    attn_ratio = sum([LA.norm(word_embeddings[i]) for i in sent_keyword_idxs])
    # normalize the attn_ratio
    attn_ratio /= sum([LA.norm(emb) for emb in word_embeddings])
    return base_ratio, attn_ratio

def calc_hammer_loss(ix, attention, block_ids, coef=0.0):
    sent_keyword_idxs = [idx for idx, val in enumerate(block_ids) if val == 1]
    if len(sent_keyword_idxs) == 0:
        return torch.zeros([1]).type(float_type)
    loss = -1 * coef * torch.log(1 - torch.sum(attention[sent_keyword_idxs]))
    return loss

def calc_kld_loss(p, q, coef=0.0):
    if p is None or q is None:
        return torch.tensor([0.0]).type(float_type)
    return -1 * coef * torch.dot(p, torch.log(p/q))

def entropy(p):
    return torch.distributions.Categorical(probs=p).entropy()

def calc_entropy_loss(p, beta):
    return -1 * beta * entropy(p)

def evaluate(dataset, iter, name='test', attn_stats=False, num_vis=0):
    print ("evaluating on %s set" %(name))
    # Perform testing
    test_correct = 0.0
    test_base_prop = 0.0
    test_attn_prop = 0.0
    test_base_emb_norm = 0.0
    test_attn_emb_norm = 0.0
    test_base_h_norm = 0.0
    test_attn_h_norm = 0.0

    example_data = []

    total_loss = 0.0
    if num_vis > 0 and UNDERSTAND:
        wts, bias = model.get_linear_wts()
        print ("Weights below")
        print (wts.detach().cpu().numpy())
        print ("bias below")
        print (bias.detach().cpu().numpy())
    for idx, words, block_ids, attn_orig , tag in dataset:
        words_t = torch.tensor([words]).type(type)
        tag_t = torch.tensor([tag]).type(type)
        if attn_orig is not None:
            attn_orig = torch.tensor(attn_orig).type(float_type)

        block_ids_t = torch.tensor([block_ids]).type(float_type)

        if name == 'test' and FLOW:
            pred, attn = model(words_t, block_ids_t)
        else:
            pred, attn = model(words_t)
        attention = attn[0]

        if not FLOW or (name != 'test'):
            assert 0.99 < torch.sum(attention).item() < 1.01

        ce_loss = calc_ce_loss(pred, tag_t)
        entropy_loss = calc_entropy_loss(attention, C_ENTROPY)
        hammer_loss = calc_hammer_loss(words, attention,
                                        block_ids, C_HAMMER)
        kld_loss = calc_kld_loss(attention, attn_orig, C_KLD)

        assert hammer_loss.item() >= 0.0
        assert ce_loss.item() >= 0.0

        loss = ce_loss + entropy_loss + hammer_loss
        total_loss += loss.item()

        word_embeddings = model.get_embeddings(words_t)
        word_embeddings = word_embeddings[0].detach().cpu().numpy()
        assert len(words) == len(word_embeddings)

        final_states = model.get_final_states(words_t)
        final_states = final_states[0].detach().cpu().numpy()
        assert len(words) == len(final_states)

        predict = pred[0].argmax().item()
        if predict == tag:
            test_correct += 1


        if idx < num_vis:

            attn_scores = attn[0].detach().cpu().numpy()
            # util.pretty_importance_scores_vertical([i2w[w] \
            #     for w in words], attn_scores)

            example_data.append([[i2w[w] for w in words], attn_scores, i2t[predict], i2t[tag]])


            if UNDERSTAND:
                headers = ['words', 'attn'] + ['e' + str(i + 1) for i in range(EMB_SIZE)]
                tabulated_list = []
                for j in range(len(words)):
                    temp_list =  [i2w[words[j]], attn_scores[j]]
                    for emb in word_embeddings[j]:
                        temp_list.append(emb)
                    tabulated_list.append(temp_list)
                print (tabulate(tabulated_list, headers=headers))


        base_prop, attn_prop = quantify_attention(words, attention.detach().cpu().numpy(), block_ids)
        base_emb_norm, attn_emb_norm = quantify_norms(words, word_embeddings, block_ids)
        base_h_norm, attn_h_norm = quantify_norms(words, final_states, block_ids)

        test_base_prop += base_prop
        test_attn_prop += attn_prop

        test_base_emb_norm += base_emb_norm
        test_attn_emb_norm += attn_emb_norm

        test_base_h_norm += base_h_norm
        test_attn_h_norm += attn_h_norm

    print("iter %r: %s acc = %.2f" % (iter, name, 100.*test_correct/len(dataset)))
    print("iter %r: %s loss = %.8f" % (iter, name, total_loss/len(dataset)))

    '''
    outfile_name = "examples/" + TASK_NAME + "_" + MODEL_TYPE + "_hammer=" + str(C_HAMMER) \
         +"_kld=" + str(C_KLD) + "_seed=" + str(SEED) + "_iter=" +  str(iter) + ".pickle"
        
    pickle.dump(example_data, open(outfile_name, 'wb'))
    '''

    if attn_stats:
        print("iter %r: in %s set base_ratio = %.8f, attention_ratio = %.14f" % (
            iter,
			name,
            test_base_prop/len(dataset),
            test_attn_prop/len(dataset)))

        print("iter %r: in %s set base_emb_norm = %.4f, attn_emb_norm = %.4f" % (
            iter,
			name,
            test_base_emb_norm/len(dataset),
            test_attn_emb_norm/len(dataset)))

        print("iter %r: in %s set base_h_norm = %.4f, attn_h_norm = %.4f" % (
            iter,
			name,
            test_base_h_norm/len(dataset),
            test_attn_h_norm/len(dataset)))

    return test_correct/len(dataset), total_loss/len(dataset)


def dump_attention_maps(dataset, filename):

    fw = open(filename, 'w')

    dataset = sorted(dataset, key=lambda x:x[0])
    for _ , words, _ , _, _ in dataset:
        words_t = torch.tensor([words]).type(type)
        _ , attn = model(words_t)
        attention = attn[0].detach().cpu().numpy()

        for att in attention:
            fw.write(str(att) + " ")
        fw.write("\n")
    fw.close()
    return


"""" Reading the data """
prefix = "data/" + TASK_NAME + "/"

if USE_BLOCK_FILE:
    log.pr_blue("Using block file")
    train = list(read_dataset(prefix+"train.txt",
                block_file=prefix + "train.txt.block", clip_vocab=CLIP_VOCAB))
    w2i = defaultdict(lambda: UNK, w2i)
    nwords = len(w2i) if not CLIP_VOCAB else VOCAB_SIZE
    t2i = defaultdict(lambda: UNK, t2i)

    dev = list(read_dataset(prefix+"dev.txt",
                block_file=prefix + "dev.txt.block"))
    test = list(read_dataset(prefix+"test.txt",
                block_file=prefix + "test.txt.block"))
elif USE_ATTN_FILE:
    log.pr_blue("Using attn file")
    train = list(read_dataset(prefix+"train.txt", block_words=BLOCK_WORDS,
                attn_file=prefix + "train.txt.attn." + MODEL_TYPE, clip_vocab=CLIP_VOCAB))
    w2i = defaultdict(lambda: UNK, w2i)
    nwords = len(w2i) if not CLIP_VOCAB else VOCAB_SIZE
    t2i = defaultdict(lambda: UNK, t2i)

    dev = list(read_dataset(prefix+"dev.txt", block_words=BLOCK_WORDS,
                attn_file=prefix + "dev.txt.attn." + MODEL_TYPE))
    test = list(read_dataset(prefix+"test.txt", block_words=BLOCK_WORDS,
                attn_file=prefix + "test.txt.attn." + MODEL_TYPE))
else:
    if BLOCK_WORDS is None:
        log.pr_blue("Vanilla case: no attention manipulation")
    else:
        log.pr_blue("Using block words")

    train = list(read_dataset(prefix+"train.txt", block_words=BLOCK_WORDS, clip_vocab=CLIP_VOCAB))
    nwords = len(w2i) if not CLIP_VOCAB else VOCAB_SIZE
    w2i = defaultdict(lambda: UNK, w2i)
    t2i = defaultdict(lambda: UNK, t2i)

    dev = list(read_dataset(prefix+"dev.txt", block_words=BLOCK_WORDS))
    test = list(read_dataset(prefix+"test.txt", block_words=BLOCK_WORDS))


if DEBUG:
    train = train[:100]
    dev = dev[:100]
    test = test[:100]

# Create reverse dicts
i2w = {v: k for k, v in w2i.items()}
i2w[UNK] = "<unk>"
i2t = {v: k for k, v in t2i.items()}


ntags = len(t2i)

log.pr_cyan("The vocabulary size is %d" %(nwords))

if MODEL_TYPE == 'emb-att':
    model = EmbAttModel(nwords, EMB_SIZE, ntags)
elif MODEL_TYPE == 'emb-lstm-att':
    model = BiLSTMAttModel(nwords, EMB_SIZE, HID_SIZE, ntags)
elif MODEL_TYPE == 'no-att-only-lstm':
    model = BiLSTMModel(nwords, EMB_SIZE, HID_SIZE, ntags)
else:
    raise ValueError("model type not compatible")

calc_ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
type = torch.LongTensor
float_type = torch.FloatTensor
use_cuda = torch.cuda.is_available()

if use_cuda:
    type = torch.cuda.LongTensor
    float_type = torch.cuda.FloatTensor
    model.cuda()

print ("evaluating without any training ...")
_, _ = evaluate(test, 0, name='test', attn_stats=True,
                        num_vis=0)


print ("starting to train")


best_dev_accuracy  = 0.
best_dev_loss = np.inf
best_test_accuracy = 0.
best_epoch = 0

for ITER in range(1, NUM_EPOCHS+1):
    random.shuffle(train)
    train_loss = 0.0
    train_ce_loss = 0.0
    train_entropy_loss = 0.0
    train_hammer_loss = 0.0
    train_kld_loss = 0.0

    start = time.time()
    for num, (idx, words_orig, block_ids, attn_orig, tag) in enumerate(train):

        words = torch.tensor([words_orig]).type(type)
        tag = torch.tensor([tag]).type(type)
        if attn_orig is not None:
            attn_orig = torch.tensor(attn_orig).type(float_type)

        # forward pass
        out, attns = model(words)
        attention = attns[0]

        ce_loss = calc_ce_loss(out, tag)
        entropy_loss = calc_entropy_loss(attention, C_ENTROPY)
        hammer_loss = calc_hammer_loss(words_orig, attention,
                                        block_ids, C_HAMMER)

        kld_loss = calc_kld_loss(attention, attn_orig, C_KLD)

        loss = ce_loss + entropy_loss + hammer_loss + kld_loss
        train_loss += loss.item()

        train_ce_loss += ce_loss.item()
        train_entropy_loss += entropy_loss.item()
        train_hammer_loss += hammer_loss.item()
        train_kld_loss += kld_loss.item()

        print ("ID: %4d\t CE: %0.4f\t ENTROPY: %0.4f\t HAMMER: %0.4f\t KLD: %.4f\t TOTAL: %0.4f" %(
            num,
            ce_loss.item(),
            entropy_loss.item(),
            hammer_loss.item(),
            kld_loss.item(),
            loss.item()
        ), end='\r')

        # update the params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("iter %r: train loss=%.4f, ce_loss=%.4f, entropy_loss=%.4f,"
                "hammer_loss=%.4f, kld_loss==%.4f, time=%.2fs" % (
                ITER,
                train_loss/len(train),
                train_ce_loss/len(train),
                train_entropy_loss/len(train),
                train_hammer_loss/len(train),
                train_kld_loss/len(train),
                time.time()-start))

    _, _  = evaluate(train, ITER, name='train')
    dev_acc, dev_loss  = evaluate(dev, ITER, name='dev', attn_stats=True)
    test_acc, test_loss = evaluate(test, ITER, name='test', attn_stats=True,
                        num_vis=NUM_VIS)

    if ((not USE_LOSS) and dev_acc > best_dev_accuracy) or (USE_LOSS and dev_loss < best_dev_loss):

        if USE_LOSS:
            best_dev_loss = dev_loss
        else:
            best_dev_accuracy = dev_acc
        best_test_accuracy = test_acc
        best_epoch = ITER

        if TO_DUMP_ATTN:
            log.pr_bmagenta("dumping attention maps")
            dump_attention_maps(train, prefix + "train.txt.attn." + MODEL_TYPE)
            dump_attention_maps(dev, prefix + "dev.txt.attn." + MODEL_TYPE)
            dump_attention_maps(test, prefix + "test.txt.attn." + MODEL_TYPE)


    print ("iter %r: best test accuracy = %.4f attained after epoch = %d" %(
        ITER, best_test_accuracy, best_epoch))

