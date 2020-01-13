# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import subprocess
from collections import OrderedDict
import numpy as np
import torch

from ..utils import to_cuda, restore_segmentation, concat_batches
from ..model.memory import HashingMemory
from .evaluator import EncDecEvaluator, convert_to_text
from ..model.transformer import BeamHypotheses


BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)


logger = getLogger()

class EncDecGenerator(EncDecEvaluator):

    def __init__(self, trainer, data, params):
        super().__init__(trainer, data, params)

    def generate(self, trainer):
        params = self.params
        with torch.no_grad():
            for data_set in ['test']:
                # machine translation task (evaluate perplexity and accuracy)
                for lang1, lang2 in set(params.mt_steps):
                    self.generate_mt(data_set, lang1, lang2)

    def generate_mt(self, data_set, lang1, lang2):
        logger.info("begining generating")
        params = self.params
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        hypothesis = []
        hyp_name = 'hyp.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set)
        ref_name = 'src.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set)
        hyp_path = os.path.join(params.data_path, hyp_name)
        ref_path = os.path.join(params.data_path, ref_name)
        f_hyp = open(hyp_path, 'w', encoding='utf-8')
        f_ref = open(ref_path, 'w', encoding='utf-8')
        for batch in self.get_iterator(data_set, lang1, lang2):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)

            # cuda
            x1, len1, langs1 = to_cuda(x1, len1, langs1)

            # encode source sentence
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1

            # generate translation - translate / convert to text
            max_len = int(1.5 * len1.max().item() + 10)
            generated, lengths = decoder.generate_beam(
                enc1, len1, lang2_id, beam_size=params.beam_size,
                length_penalty=params.length_penalty,
                early_stopping=params.early_stopping,
                max_len=max_len
            )
            hypothesis = convert_to_text(generated, lengths, self.dico, params)
            ref = convert_to_text(x1, len1, self.dico, params)
            f_hyp.write('\n'.join(hypothesis) + '\n')
            f_hyp.flush()
            f_ref.write('\n'.join(ref) + '\n')
            f_ref.flush()
        f_hyp.close()
        f_ref.close()
        restore_segmentation(hyp_path)
