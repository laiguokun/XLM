# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import copy
import time
import json
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from ..optim import get_optimizer
from ..utils import truncate, to_cuda
from ..utils import concat_batches as concat_batches_xnli
from ..data.dataset import ParallelDataset, Dataset
from ..data.loader import load_binarized, set_dico_parameters


XNLI_LANGS = ['en', 'de']


logger = getLogger()

def create_batches(
        x, lens, lang_id, pad_idx, eos_idx,
        reset_positions):
    langs = x.new(x.size()).fill_(lang_id)
    lengths = lens[None, :]
    positions = torch.arange(x.size(0))[:, None].repeat(1, x.size(1)).to(x.device)
    return x, lengths, positions, langs

def concat_batches(
        x1, len1, lang1_id, x2, len2, lang2_id, pad_idx, eos_idx,
        reset_positions):
    slen = max(x1.size(0), x2.size(0))
    bsz1 = x1.size(1)
    bsz = bsz1 * 2
    langs = x1.new(slen, bsz).fill_(0)
    lengths = torch.cat([len1, len2], 0)[None, :]
    x = x1.new(slen, bsz).fill_(pad_idx)
    x[:x1.size(0), :bsz1] = x1
    x[:x2.size(0), bsz1:] = x2
    langs[:x1.size(0), :bsz1] = lang1_id
    langs[:x2.size(0), bsz1:] = lang2_id
    positions = torch.arange(slen)[:, None].repeat(1, bsz).to(x1.device)
    return x, lengths, positions, langs

def concat_batches_0(
            x1, len1, lang1_id, x2, len2, lang2_id, pad_idx, eos_idx,
            reset_positions):
    """
    Concat batches with different languages.
    """
    bsz = x1.size(1)
    x1 = x1[:, :, None].repeat(1, 1, bsz).view(-1, bsz*bsz)
    len1 = len1[:, None].repeat(1, bsz).view(bsz*bsz,)
    x2 = x2[:, None, :].repeat(1, bsz, 1).view(-1, bsz*bsz)
    len2 = len2[None, :].repeat(bsz, 1).view(bsz*bsz,)
    x, lengths, positions, langs = concat_batches_xnli(
            x1, len1, lang1_id, x2, len2, lang2_id, pad_idx, eos_idx,
            reset_positions)
    lengths = lengths[None, :]
    return x, lengths, positions, langs

def loss_func_0(x, params, proj=None):
    bsz = int(x.size(0) ** 0.5)
    y = torch.zeros(bsz, bsz, device=x.device).long()
    mask = torch.eye(bsz, bsz).byte().to(x.device)
    y.masked_fill_(mask, 1)
    y = y.view(-1,)
    output = proj(x)
    loss = F.cross_entropy(output, y)
    pred = torch.max(output, 1)[1]
    mask = mask.view(bsz * bsz,).float()
    acc = torch.sum(mask * pred.eq(y).float()) / bsz
    return loss, acc

#loss softmax([x,y]) classification [0, 1]
def loss_func_1(x, params, proj=None):
    print(x.size())
    bsz = x.size(0) // 2
    x1 = x[:bsz, None, :]
    x1 = x1.repeat(1, bsz, 1).view(bsz*bsz, -1)
    x2 = x[None, bsz:, :]
    x2 = x2.repeat(bsz, 1, 1).view(bsz*bsz, -1)
    output = torch.cat([x1, x2], -1)
    output = proj(output)
    y = torch.zeros(bsz, bsz, device=x.device).long()
    mask = torch.eye(bsz, bsz).bool().to(x.device)
    y.masked_fill_(mask, 1)
    y = y.view(-1,)
    loss = F.cross_entropy(output, y)
    pred = torch.max(output, 1)[1]
    acc = torch.mean(pred.eq(y).float())
    return loss, acc

#loss softmax(xWy)
def loss_func_2(x, params, proj=None):
    bsz = x.size(0) // 2
    x1 = x[:bsz, None, :]
    x2 = x[None, bsz:, :]
    x1 = proj(x1)
    output = (x1 * x2).mean(-1).view(bsz, bsz)
    y = torch.arange(bsz).long().to(x.device)
    loss = F.cross_entropy(output, y)
    pred = torch.max(output, 1)[1]
    acc = torch.mean((pred.eq(y)).float()).item()
    return loss, acc

#loss softmax((x-y)^2)
def loss_func(x, params, proj=None):
    bsz = x.size(0) // 2
    x1 = x[:bsz, None, :]
    x2 = x[None, bsz:, :]
    dist = ((x1 - x2) * (x1 - x2)).mean(-1)
    output = dist.view(bsz, bsz)
    y = torch.arange(bsz).long().to(x.device)
    loss = F.cross_entropy(output, y)
    pred = torch.max(output, 1)[1]
    acc = torch.mean((pred.eq(y)).float()).item()
    return loss, acc


#loss hinge_loss([x,y])
def loss_func_hinge(x, params, proj=None):
    bsz = x.size(0) // 2
    x1 = x[:bsz, None, :]
    x1 = x1.repeat(1, bsz, 1).view(bsz*bsz, -1)
    x2 = x[None, bsz:, :]
    x2 = x2.repeat(bsz, 1, 1).view(bsz*bsz, -1)
    y = torch.zeros(bsz, bsz, device=x.device).float() - 1
    mask = torch.eye(bsz, bsz).bool().to(x.device)
    y.masked_fill_(mask, 1)
    y = y.view(-1, 1)
    output = torch.cat([x1, x2], -1)
    output = proj(output)
    loss = torch.max(1 - (output * y), torch.zeros(bsz*bsz, device=x.device))
    loss = torch.mean(loss)
    pred = ((output > 0).float() * 2 - 1).view(-1, 1)
    acc = torch.mean((pred.eq(y)).float()).item()
    return loss, acc


class TRANS:

    def __init__(self, embedder, scores, params):
        """
        Initialize XNLI trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        self.params = params
        self.scores = scores
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_iterator(self, splt, lang):
        """
        Get a monolingual data iterator.
        """
        assert splt in ['valid', 'train']
        return self.data[lang][splt]['x'].get_iterator(
            shuffle=(splt == 'train'),
            group_by_size=self.params.group_by_size,
            return_indices=True
        )

    def run(self):
        """
        Run XNLI training / evaluation.
        """
        params = self.params

        # load data
        self.data = self.load_data()
        if not self.data['dico'] == self._embedder.dico:
            raise Exception(("Dictionary in evaluation data (%i words) seems different than the one " +
                             "in the pretrained model (%i words). Please verify you used the same dictionary, " +
                             "and the same values for max_vocab and min_count.") % (len(self.data['dico']), len(self._embedder.dico)))

        # embedder
        self.embedder = copy.deepcopy(self._embedder)
        if torch.cuda.device_count() > 1:
            print('using', torch.cuda.device_count(), 'GPUs')
            self.embedder.set_para()

        self.embedder.to(self.device)

        # projection layer
        self.proj = nn.Sequential(*[
           nn.Dropout(params.dropout),
           nn.Linear(self.embedder.out_dim, self.embedder.out_dim)
        ]).cuda()

        # self.proj = nn.Sequential(*[
        #     nn.Dropout(params.dropout),
        #     nn.Linear(self.embedder.out_dim*2, 64, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(64, 1, bias=False)
        # ]).cuda()

        # optimizers
        self.optimizer_e = get_optimizer(list(self.embedder.get_parameters(params.finetune_layers)), params.optimizer_e)
        self.optimizer_p = get_optimizer(self.proj.parameters(), params.optimizer_p)

        # train and evaluate the model
        best_score = 0
        for epoch in range(params.n_epochs):

            # update epoch
            self.epoch = epoch

            # training
            logger.info("XNLI - Training epoch %i ..." % epoch)
            self.train()

            # evaluation
            logger.info("XNLI - Evaluating epoch %i ..." % epoch)
            with torch.no_grad():
                scores = self.eval()
                if scores['res'] > best_score:
                    best_score = scores['res']
                    self.save_checkpoint('checkpoint_best')
                self.save_checkpoint('epoch_{}'.format(epoch))
                self.scores.update(scores)

    def train(self):
        """
        Finetune for one epoch on the XNLI English training set.
        """
        params = self.params
        self.embedder.train()

        # training variables
        losses = []
        ns = 0  # number of sentences
        nw = 0  # number of words
        nacc = 0
        t = time.time()

        iterator = self.get_iterator('train', 'en')
        lang_id_1 = params.lang2id['en']
        lang_id_2 = params.lang2id['de']

        while True:

            # batch
            try:
                batch = next(iterator)
            except StopIteration:
                break

            (sent1, len1), (sent2, len2), idx = batch
            sent1, len1 = truncate(sent1, len1, params.max_len, params.eos_index)
            sent2, len2 = truncate(sent2, len2, params.max_len, params.eos_index)

            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang_id_1,
                sent2, len2, lang_id_2,
                params.pad_index,
                params.eos_index,
                reset_positions=False
            )

            bs = len(len1)

            # cuda
            x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)
            # loss
            output = self.embedder.get_embeddings(x, lengths, positions, langs)
            loss, acc = loss_func(output, self.params, self.proj)
            # backward / optimization
            self.optimizer_e.zero_grad()
            self.optimizer_p.zero_grad()
            loss.backward()
            self.optimizer_e.step()
            self.optimizer_p.step()

            # update statistics
            ns += bs
            nw += lengths.sum().item()
            losses.append(loss.item())
            nacc += (acc * bs)
            bns += bs

            # log
            if ns % (100 * bs) < bs:
                logger.info("XNLI - Epoch %i - Train sample %7i - %.1f \
                            words/s - Loss: %.4f - ACC: %.4f" % \
                            (self.epoch, ns, nw / (time.time() - t),
                            sum(losses) / len(losses), nacc / bns))
                nw, t = 0, time.time()
                bns = 0
                nacc = 0
                losses = []

            # epoch size
            if params.epoch_size != -1 and ns >= params.epoch_size:
                break

    def eval(self):
        """
        Evaluate on XNLI validation and test sets, for all languages.
        """
        params = self.params
        self.embedder.eval()

        scores = OrderedDict({'epoch': self.epoch})
        lang_id_1 = params.lang2id['en']
        lang_id_2 = params.lang2id['de']
        splt = 'valid'
        valid = 0
        total = 0

        for batch in self.get_iterator(splt, 'en'):

            # batch
            (sent1, len1), (sent2, len2), idx = batch
            x, lengths, positions, langs = concat_batches(
                sent1, len1, lang_id_1,
                sent2, len2, lang_id_2,
                params.pad_index,
                params.eos_index,
                reset_positions=False
            )

            # cuda
            x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)

            # forward
            output = self.embedder.get_embeddings(x, lengths, positions, langs)
            loss, acc = loss_func(output, self.params, self.proj)

            # update statistics
            valid += acc * len(len1)
            total += len(len1)

        # compute accuracy
        acc = 100.0 * valid / total
        scores['res'] = acc
        logger.info("TRANS - %s - Epoch %i - Acc: %.1f%%" % (splt, self.epoch, acc))
        return scores

    def load_data(self):
        """
        Load XNLI cross-lingual classification data.
        """
        params = self.params
        data = {}
        lang = 'en'
        data[lang] = {splt: {} for splt in ['train', 'valid']}
        dpath = os.path.join(params.data_path, 'ende')

        for splt in ['train', 'valid']:

            # load data and dictionary
            # data1 = load_binarized(os.path.join(dpath, '%s.%s.pth' % (splt, lang)), params)
            # data2 = load_binarized(os.path.join(dpath, '%s.%s.pth' % (splt, lang)), params)
            data1 = load_binarized(os.path.join(dpath, '%s.%s.pth' % (splt, 'en')), params)
            data2 = load_binarized(os.path.join(dpath, '%s.%s.pth' % (splt, 'de')), params)
            data['dico'] = data.get('dico', data1['dico'])

            # set dictionary parameters
            set_dico_parameters(params, data, data1['dico'])
            set_dico_parameters(params, data, data2['dico'])

            # create dataset
            data[lang][splt]['x'] = ParallelDataset(
                data1['sentences'], data1['positions'],
                data2['sentences'], data2['positions'],
                params
            )

        return data

    def save_checkpoint(self, name, include_optimizers=False):
        """
        Save the model / checkpoints.
        """
        path = os.path.join(self.params.dump_path, '%s.pth' % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {}

        logger.warning(f"Saving model parameters ...")
        data['model'] = self.embedder.model.state_dict()
        data['proj'] = self.proj.state_dict()

        data['dico_id2word'] = self.data['dico'].id2word
        data['dico_word2id'] = self.data['dico'].word2id
        data['dico_counts'] = self.data['dico'].counts
        data['params'] = {k: v for k, v in self.params.__dict__.items()}

        torch.save(data, path)


class TRANS_h:

    def __init__(self, embedder, proj, scores, params):
        """
        Initialize XNLI trainer / evaluator.
        Initial `embedder` should be on CPU to save memory.
        """
        self._embedder = embedder
        self.params = params
        self.scores = scores
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # embedder
        self.embedder = copy.deepcopy(self._embedder)
        if torch.cuda.device_count() > 1:
            print('using', torch.cuda.device_count(), 'GPUs')
            self.embedder.set_para()

        self.embedder.to(self.device)
        self.proj = proj
        if proj is not None:
            self.proj = self.proj.to(self.device)

    def get_iterator(self, lang):
        """
        Get a monolingual data iterator.
        """
        return self.data[lang]['x'].get_iterator(
            shuffle=False,
            group_by_size=self.params.group_by_size,
            return_indices=True
        )

    def get_hidden(self, file_name, lang):
        params = self.params
        self.data = self.load_data(file_name, lang)
        self.embedder.eval()
        lang_id = params.lang2id[lang]
        hidden = []
        with torch.no_grad():
            for batch in self.get_iterator(lang):

                # batch
                (sents, lens), idx = batch
                x, lengths, positions, langs = create_batches(
                    sents, lens, lang_id,
                    params.pad_index,
                    params.eos_index,
                    reset_positions=False
                )

                # cuda
                x, lengths, positions, langs = to_cuda(x, lengths, positions, langs)

                # forward
                output = self.embedder.get_embeddings(x, lengths, positions, langs)
                if lang == 'en' and self.proj is not None:
                    output = self.proj(output)

                hidden.append(output.cpu())
        hidden = torch.cat(hidden, 0)
        save_path = os.path.join(params.save_path, file_name.split('/')[-1])
        torch.save(hidden, save_path)

    def load_data(self, file_name, lang):
        """
        Load XNLI cross-lingual classification data.
        """
        params = self.params
        data = {}
        data[lang] = {}
        # load data and dictionary
        data1 = load_binarized(file_name, params)
        # set dictionary parameters
        data['dico'] = data.get('dico', data1['dico'])
        set_dico_parameters(params, data, data1['dico'])
        # create dataset
        data[lang]['x'] = Dataset(
                data1['sentences'], data1['positions'],
                params)
        return data