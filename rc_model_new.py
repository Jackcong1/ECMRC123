# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""
import sys

sys.path.append("..")
import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder
import math
import tensorflow.contrib as tc


class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self , vocab , args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # the vocab
        self.vocab = vocab

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # self.writer = tf.summary.FileWriter("/home/congyao/Dureader_congyao2/data/summary/test" , self.sess.graph)

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()

        self._hybrid_encoder()
        #self._self_attention()

        self._match()
        self._fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32 , [None , None])
        self.q = tf.placeholder(tf.int32 , [None , None])
        self.p_length = tf.placeholder(tf.int32 , [None])
        self.q_length = tf.placeholder(tf.int32 , [None])
        self.start_label = tf.placeholder(tf.int32 , [None])
        self.end_label = tf.placeholder(tf.int32 , [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0') , tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings' ,
                shape=(self.vocab.size() , self.vocab.embed_dim) ,
                initializer=tf.constant_initializer(self.vocab.embeddings) ,
                trainable=False
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings , self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings , self.q)
            print(self.p_emb , self.q_emb)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        # hidden_size        
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes , _ = rnn('bi-oncell' , self.p_emb , self.p_length , self.hidden_size)
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes , _ = rnn('bi-oncell' , self.q_emb , self.q_length , self.hidden_size)

        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes , self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes , self.dropout_keep_prob)
        print(self.sep_p_encodes , tf.shape(self.sep_p_encodes) , self.sep_q_encodes , tf.shape(self.sep_q_encodes))

    def _hybrid_encoder(self):
        with tf.variable_scope('shared_paramater_encoder' , reuse=True):
            self.shared_p_encodes , _ = rnn('bi-lstm' , self.sep_p_encodes , self.p_length , self.hidden_size)
            self.shared_q_encodes , _ = rnn('bi-lstm' , self.sep_q_encodes , self.q_length , self.hidden_size)

        with tf.variable_scope('independent_paramater_encoder' ):
            self.indepent_q_encodes , _ = rnn('bi-lstm' , self.sep_q_encodes , self.q_length , self.hidden_size)
        with tf.variable_scope('Hybrid_encoding' ):
            self.temperature = tf.Variable(tf.convert_to_tensor(1 / math.sqrt(self.hidden_size)) , trainable=False)
            sim_matrix1 = tf.matmul(self.shared_p_encodes , self.shared_q_encodes , transpose_b=True) * self.temperature
            print(tf.shape(sim_matrix1))
            self.hybrid_H1 = tf.matmul(tf.nn.softmax(sim_matrix1 , -1) , self.indepent_q_encodes)

            sim_matrix2 = tf.matmul(self.sep_p_encodes , self.shared_q_encodes , transpose_b=True) * self.temperature
            print(tf.shape(sim_matrix2))
            self.hybrid_H2 = tf.matmul(tf.nn.softmax(sim_matrix2 , -1) , self.indepent_q_encodes)

            self.hybrid_encoding = self.hybrid_H1 + self.hybrid_H2
            print(tf.shape(self.hybrid_encoding))

    def _self_attention(self):
        with tf.variable_scope('attention_encoding'):
            self.temperature = tf.Variable(tf.convert_to_tensor(1 / math.sqrt(self.hidden_size)) , trainable=False)
            sim_matrix1 = tf.matmul(self.p_emb , self.p_emb , transpose_b=True) * self.temperature
            print(tf.shape(sim_matrix1))
            self.attention_p_encoding = tf.matmul(tf.nn.softmax(sim_matrix1 , -1) , self.indepent_q_encodes)

            sim_matrix2 = tf.matmul(self.q_emb , self.q_emb , transpose_b=True) * self.temperature
            print(tf.shape(sim_matrix2))
            self.attention_q_encoding = tf.matmul(tf.nn.softmax(sim_matrix2 , -1) , self.indepent_q_encodes)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes , _ = match_layer.match(self.sep_p_encodes , self.sep_q_encodes ,
                                                     self.p_length , self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes , self.dropout_keep_prob)

        print(self.match_p_encodes , tf.shape(self.match_p_encodes))

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion' , reuse=tf.AUTO_REUSE):
            self.match_p_encodes = tf.concat([self.hybrid_encoding , self.match_p_encodes ] ,
                                             -1)
            # self.match_p_encodes = tc.layers.fully_connected(self.match_p_encodes , num_outputs=self.hidden_size * 2 ,
            #                                                  activation_fn=tf.nn.elu)
            self.fuse_p_encodes , _ = rnn('bi-lstm' , self.match_p_encodes , self.p_length ,
                                          self.hidden_size , layer_num=2)

            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes , self.dropout_keep_prob)
            print(self.fuse_p_encodes , tf.shape(self.fuse_p_encodes))

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.fuse_p_encodes ,
                [batch_size , -1 , 2 * self.hidden_size]
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes ,
                [batch_size , -1 , tf.shape(self.sep_q_encodes)[1] , 2 * self.hidden_size]
            )[0: , 0 , 0: , 0:]
        decoder = PointerNetDecoder(self.hidden_size)
        print(concat_passage_encodes , tf.shape(concat_passage_encodes))
        self.start_probs , self.end_probs = decoder.decode(concat_passage_encodes ,
                                                           no_dup_question_encodes)
        print(tf.shape(self.start_probs) , tf.shape(self.end_probs))

    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs , labels , epsilon=1e-9 , scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope , "log_loss"):
                labels = tf.one_hot(labels , tf.shape(probs)[1] , axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon) , 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs , labels=self.start_label)
        print(tf.shape(self.start_loss))
        self.end_loss = sparse_nll_loss(probs=self.end_probs , labels=self.end_label)
        print(tf.shape(self.end_loss))
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss , self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
                l1 = tf.add_n([tf.nn.l1_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss +self.weight_decay * l1_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self , train_batches , dropout_keep_prob):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num , total_loss = 0 , 0
        log_every_n_batch , n_batch_loss = 50 , 0
        for bitx , batch in enumerate(train_batches , 1):
            feed_dict = {self.p: batch['passage_token_ids'] ,
                         self.q: batch['question_token_ids'] ,
                         self.p_length: batch['passage_length'] ,
                         self.q_length: batch['question_length'] ,
                         self.start_label: batch['start_id'] ,
                         self.end_label: batch['end_id'] ,
                         self.dropout_keep_prob: dropout_keep_prob}
            _ , loss = self.sess.run([self.train_op , self.loss] , feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1 , bitx , n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self , data , epochs , batch_size , save_dir , save_prefix ,
              dropout_keep_prob=1.0 , evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0
        for epoch in range(1 , epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train' , batch_size , pad_id , shuffle=True)
            train_loss = self._train_epoch(train_batches , dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch , train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev' , batch_size , pad_id , shuffle=False)
                    eval_loss , bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir , save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir , save_prefix + '_' + str(epoch))

    def evaluate(self , eval_batches , result_dir=None , result_prefix=None , save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers , ref_answers = [] , []
        total_loss , total_num = 0 , 0
        for b_itx , batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'] ,
                         self.q: batch['question_token_ids'] ,
                         self.p_length: batch['passage_length'] ,
                         self.q_length: batch['question_length'] ,
                         self.start_label: batch['start_id'] ,
                         self.end_label: batch['end_id'] ,
                         self.dropout_keep_prob: 1.0}
            start_probs , end_probs , loss = self.sess.run([self.start_probs ,
                                                            self.end_probs , self.loss] , feed_dict)
            if b_itx % 10 == 0:
                # counting when the model predicting answers
                self.logger.info('predict results {}'.format(b_itx))

                # self.writer.add_summary(summary_str, b_itx)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample , start_prob , end_prob in zip(batch['raw_data'] , start_probs , end_probs):

                best_answer = self.find_best_answer(sample , start_prob , end_prob , padded_p_len)
                # print(best_answer)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'] ,
                                         'question_type': sample['question_type'] ,
                                         'question_tokens': sample['question_tokens'] ,
                                         'answers': [best_answer] ,
                                         'entity_answers': [[]] ,
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'] ,
                                        'question_type': sample['question_type'] ,
                                        'answers': sample['answers'] ,
                                        'entity_answers': [[]] ,
                                        'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir , result_prefix + '.json')
            with open(result_file , 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer , ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix , result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict , ref_dict = {} , {}
            for pred , ref in zip(pred_answers , ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict , ref_dict)
        else:
            bleu_rouge = None
        return ave_loss , bleu_rouge

    def find_best_answer(self , sample , start_prob , end_prob , padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx , best_span , best_score = None , None , 0
        pp_scores = (0.43 , 0.23 , 0.16 , 0.10 , 0.09)
        pb_scores = (0.9 , 0.05 , 0.01 , 0.0001 , 0.0001)
        pp_score = [0.45302071830347496 , 0.2338820395134873 , 0.1498311318839278 , 0.09622094011093592 ,
                    0.06704517018817402]
        for p_idx , passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len , len(passage['passage_tokens']))
            answer_span , score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len] ,
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len] ,
                passage_len)
            # score *= pp_scores[p_idx]
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        # print(sample['passages'][best_p_idx]['passage_tokens'], best_span)
        return best_answer

    def find_best_answer_for_passage(self , start_probs , end_probs , passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        """
        ADD word_level answer prior 12.10
        """
        wp_score_start = [8.255388704977174e-05 , 4.1276943524885865e-06 , 0.0002889740481059785 ,
                          0.0014777930990243903 , 0.007842866283183626 , 0.019569932757759315 , 0.03159417414078938 ,
                          0.03750944969347897 , 0.04051493029204531 , 0.04100230412971938 , 0.039161627816710204 ,
                          0.034365188633776925 , 0.030278614728429833 , 0.02655930637884525 , 0.022757468679971363 ,
                          0.019706975502411468 , 0.017403547562981778 , 0.015649163498908873 , 0.01450573163679373 ,
                          0.013060963364290641 , 0.011612036316271817 , 0.011240486569773434 , 0.01001444738281907 ,
                          0.009642938532588123 , 0.008932904762380097 , 0.008231160188458129 , 0.007789453270056585 ,
                          0.007991716291447749 , 0.0069762413184090545 , 0.006980351018403871 , 0.006352938750407778 ,
                          0.0061919412215866165 , 0.006253882810485104 , 0.005803928137348505 , 0.005684211369037191 ,
                          0.0054241317266821975 , 0.005420023117254513 , 0.005184731452357084 , 0.004837977317286187 ,
                          0.004771951656720475 , 0.004891630255182183 , 0.004643970775167131 , 0.004260069026774534 ,
                          0.004466470102905937 , 0.004342636000629885 , 0.004152757698146883 , 0.004288976519331099 ,
                          0.004276583075885883 , 0.003991767803295644 , 0.004086701501701487 , 0.003966995093777923 ,
                          0.003909199193589595 , 0.0038679364274374203 , 0.003686313513659396 , 0.0038142654951837524 ,
                          0.0037193492458520157 , 0.003760627279944033 , 0.00377712606111554 , 0.0034840428582983106 ,
                          0.0037853945366260965 , 0.0035377170622533727 , 0.003624398643655633 , 0.0036533050456450666 ,
                          0.003409745995804213 , 0.0035335910037515815 , 0.0034964330300421313 , 0.003327200833291494 ,
                          0.0034551539053829824 , 0.003281787470877067 , 0.003137321440241361 , 0.003162077791252108 ,
                          0.0030423626587914913 , 0.002811186692008103 , 0.0032074780668609557 , 0.003046491988994677 ,
                          0.0028359534034066007 , 0.0031042889797501362 , 0.002926775765966929 , 0.002922643164062348 ,
                          0.0029350295188212087 , 0.002922658432002191 , 0.002646077827341428 , 0.0027534000616403946 ,
                          0.00274514303708472 , 0.0028772270752300917 , 0.0025181018533401763 , 0.0028566038714074914 ,
                          0.0027368941917825327 , 0.0024107801643247755 , 0.0025015752627068136 ,
                          0.0024603037720175858 , 0.002386011271792015 , 0.0023901340585924112 , 0.00246032013052456 ,
                          0.002344722877312247 , 0.0023653657113432165 , 0.0023447277848643393 , 0.0022662950487642667 ,
                          0.002324082769699107 , 0.002249785361921444 , 0.002121809933203758 , 0.002088786197249586 ,
                          0.0019443130779275247 , 0.002026873508380086 , 0.002225002837299538 , 0.002101167099172788 ,
                          0.0021713542616720685 , 0.0020887851066824542 , 0.0019649542761077964 ,
                          0.0019814574195478295 , 0.0018782519739300356 , 0.001964941734585783 , 0.002101162191620696 ,
                          0.0018865024550829205 , 0.0019360582345061134 , 0.001845221149289508 , 0.0018493722908353265 ,
                          0.001865873798424662 , 0.001684241614826019 , 0.0017585461112900375 , 0.0018617351984008572 ,
                          0.001762677077343921 , 0.0017461624829490058 , 0.0015067284010428117 , 0.0015975442202003504 ,
                          0.0015480173408061453 , 0.0016181892353655828 , 0.0017502879961672313 ,
                          0.0015686574484192859 , 0.0015149996029711972 , 0.0016388255259937627 ,
                          0.0015769057484379073 , 0.001535618444525271 , 0.001572788959756735 , 0.0014200430026555908 ,
                          0.001395269202570738 , 0.001395274655406396 , 0.0015067376708634303 , 0.0015026028878245864 ,
                          0.001349871108096153 , 0.001292080115459918 , 0.0015562776370632147 , 0.0014984735576214004 ,
                          0.001242536877558739 , 0.0013333510608655801 , 0.0014035273176935441 , 0.001341610266555518 ,
                          0.0012755557059608187 , 0.001428298391360568 , 0.0014778263613219045 , 0.0013044621079502522 ,
                          0.0012508015360843349 , 0.0012177712567273732 , 0.001250795537965111 , 0.0011475900923473168 ,
                          0.0012342787624359326 , 0.0012631797115897083 , 0.0013044664702187786 , 0.001213639200106358 ,
                          0.0012136413812406214 , 0.0010815469827074994 , 0.0012425314247230808 ,
                          0.0010567671845034229 , 0.001089801280845345 , 0.0011269499847341765 , 0.001073292139286088 ,
                          0.0011186967771634627 , 0.0011269434413313868 , 0.0011847415226539774 ,
                          0.0011558482074701232 , 0.0010361298033081114 , 0.0010113745428644962 ,
                          0.0010237510825191723 , 0.0010237412674149876 , 0.0010691502675608886 ,
                          0.0009989821899864116 , 0.0009783393559554421 , 0.0009288048425913159 ,
                          0.0010278798674387924 , 0.0010732877770175616 , 0.001023745629683514 , 0.0009824659597407993 ,
                          0.001040260224078429 , 0.0009659530011965817 , 0.001052654212807211 , 0.00102374617496708 ,
                          0.0009411939237680056 , 0.0010320097429255442 , 0.0009618400295003699 ,
                          0.0008668817933340659 , 0.000866884519751895 , 0.0009907311635499608 , 0.0008710138499550808 ,
                          0.000970089965369689 , 0.0008503688347898484 , 0.0008751366367554772 , 0.0008875202650965087 ,
                          0.0008710122141043835 , 0.0008999011670197114 , 0.0009122935198977958 ,
                          0.0008049647421960397 , 0.0008462449574223204 , 0.0008049691044645661 ,
                          0.0009535650105870236 , 0.0008503693800734141 , 0.0008421281687411479 , 0.000788455055353217 ,
                          0.0008545008914108633 , 0.0007554373175182688 , 0.0008462455027058862 ,
                          0.0008710149405222126 , 0.0008627562801158406 , 0.0008710100329701203 ,
                          0.0008049691044645661 , 0.0008792719650778872 , 0.0008338673272005126 ,
                          0.0008214787913073888 , 0.0008214749743224283 , 0.000829736906430195 , 0.00077194591379396 ,
                          0.0007925832949892712 , 0.0008008375931271169 , 0.0008008430459627749 ,
                          0.0007224081287284388 , 0.0007471688420077123 , 0.0007430444193566187 ,
                          0.0007719393703911703 , 0.0007224026758927808 , 0.000710022319253144 , 0.0007760719722957512 ,
                          0.0007389183608548274 , 0.000664613864390809 , 0.0007389243589740513 , 0.0006893822116400038 ,
                          0.0007141538305905933 , 0.0007843224534486361 , 0.0006398373378881271 ,
                          0.0006522280549155141 , 0.000784325725150031 , 0.0006687393776090343 , 0.0007265347325137959 ,
                          0.0006480981794287624 , 0.0005737969546661388 , 0.000689375668237214 , 0.000619204864244908 ,
                          0.00074718083824616 , 0.0007430433287894869 , 0.0006398427907237851 , 0.0007347906665023388 ,
                          0.0006315863114516765 , 0.0005738013169346652 , 0.0006604823530533598 ,
                          0.0006026919057006905 , 0.0006728714342300494 , 0.0005366444337923466 ,
                          0.0005531557564858668 , 0.0005903050056582642 , 0.0006109516566741942 ,
                          0.0005614067829223174 , 0.0005242591696006176 , 0.000619204864244908 , 0.0006274635246512801 ,
                          0.0005077462110564001 , 0.0005201292941138659 , 0.0004416971032973588 ,
                          0.0005985658471988993 , 0.0005242537167649596 , 0.0005696627169108605 ,
                          0.0005036234242560037 , 0.0005366427979416492 , 0.0005944321547271868 ,
                          0.0004664605429944613 , 0.0005077418487878737 , 0.000466467086397251 , 0.0005572747263013025 ,
                          0.00046646490526298777 , 0.0005407743092790985 , 0.00044169328631239816 ,
                          0.00047472302038579394 , 0.0006026913604171247 , 0.0005077516638920582 ,
                          0.00048298168079216597 , 0.0004293162013741563 , 0.0004334389881745526 ,
                          0.0004994930034856861 , 0.0005160010544778115 , 0.00046646490526298777 ,
                          0.00044582534293341323 , 0.0004251835994695754 , 0.0004623366656269334 ,
                          0.00041692821076459827 , 0.0004747235656693598 , 0.00048710773929395713 ,
                          0.00045408073163839043 , 0.0004829838619264292 , 0.0004334422598759475 ,
                          0.0004623383014776308 , 0.000449952492002336 , 0.0004251835994695754 , 0.0004953604015811053 ,
                          0.0003591366728447974 , 0.0005077402129371762 , 0.00040867500319388434 ,
                          0.0003302406312431141 , 0.0004871028317418649 , 0.0004458269787841106 ,
                          0.0004375650466763438 , 0.0003591366728447974 , 0.00041280106169567544 ,
                          0.0004210591768184817 , 0.0003962881031514579 , 0.0003880359861478756 ,
                          0.0004293151108070247 , 0.0004458237070827158 , 0.0004004168880710781 ,
                          0.0003302406312431141 , 0.0003921604087989693 , 0.0003261145727413229 ,
                          0.00037152030118582896 , 0.00034675304450376584 , 0.0003756512672397124 ,
                          0.00035913503699409996 , 0.0004045462182742641 , 0.0002930881103693219 ,
                          0.00034262535015127727 , 0.0003715164842008683 , 0.00033436832559560265 ,
                          0.0004210591768184817 , 0.0003962875578678921 , 0.0003054771915460116 , 0.000379778961592201 ,
                          0.00035500352565665075 , 0.00038802671632725687 , 0.00030960106891353954 ,
                          0.0003384954746645254 , 0.0003921631352167983 , 0.00035088455584121505 ,
                          0.0003343715972969975 , 0.0003261091199056648 , 0.0003384954746645254 ,
                          0.0002600698272508081 , 0.000276576242392236 , 0.00033849711051522284 ,
                          0.00026006655554941326 , 0.0002518068045759096 , 0.0003632638219137202 ,
                          0.00033436887087916846 , 0.0003013451349249966 , 0.00026832248953795627 ,
                          0.0002807066631625536 , 0.0003632654577644176 , 0.0002435557781394589 ,
                          0.00026006110271375516 , 0.00027244527633835255 , 0.00025181062156087025 ,
                          0.0002394193592499174 , 0.0002930870198021903 , 0.000255932317794135 , 0.0003426264407184089 ,
                          0.00029721416887111313 , 0.00025181171212800186 , 0.00025181062156087025 ,
                          0.00022291130825779208 , 0.0002972196217067712 , 0.0002807066631625536 ,
                          0.000367393697400472 , 0.00033023899539241666 , 0.0003219846972545711 ,
                          0.0002518073498594754 , 0.0003178580934692141 , 0.0002683219442543904 ,
                          0.0002889636877182282 , 0.00022704063846097812 , 0.0002683213989708246 ,
                          0.00021878634032313252 , 0.0002930892009364535 , 0.00022703845732671485 ,
                          0.0002765756971086702 , 0.00019814459685929477 , 0.00024767801965628944 ,
                          0.00021878579503956672 , 0.0002187836139053035 , 0.0001816310930315114 ,
                          0.00023116615167920345 , 0.0002435535970051957 , 0.00025180844042660703 ,
                          0.00024768074607411846 , 0.00020227392706248076 , 0.00024355141587093244 ,
                          0.00017750667038041762 , 0.00017337843074436324 , 0.00020227338177891495 ,
                          0.00020640271198210094 , 0.00024768074607411846 , 0.00020640271198210094 ,
                          0.000272449638606879 , 0.00016924964582474303 , 0.0001857620590853948 ,
                          0.0002270411837445439 , 0.00025181062156087025 , 0.00020227229121178334 ,
                          0.00017750667038041762 , 0.0002683192178365614 , 0.00016924855525761142 ,
                          0.00018163381944934042 , 0.0001940196289246352 , 0.00018163381944934042 ,
                          0.00015686274578231662 , 0.0002229145799591869 , 0.00018988757230362015 ,
                          0.00016099480240333168 , 0.00018576096851826319 , 0.00021465646483638073 ,
                          0.00018988866287075176 , 0.00017338006659506065 , 0.000198146777993558 ,
                          0.00021465919125420975 , 0.0001898902987214492 , 0.00016512031562155704 ,
                          0.00016099044013480522 , 0.00015686601748371147 , 0.0001816327288822088 ,
                          0.00017337461375940258 , 0.00021052822520032632 , 0.0001816332741657746 ,
                          0.00015273777784765707 , 0.0001733762496101 , 0.00021052931576745793 ,
                          0.00013622372873630788 , 0.00016099207598550263 , 0.00019814296100859735 ,
                          0.00021465646483638073 , 0.00016099207598550263 , 0.0001362253645870053 ,
                          0.00014860572122664205 , 0.00016512086090512284 , 0.0001279683400313307 ,
                          0.00014860844764447108 , 0.00016099044013480522 , 0.0001320954891002535 ,
                          0.00011145756262137637 , 0.00012796943059846234 , 0.00014448347970981153 ,
                          0.0001403530589394939 , 0.0001362264551541369 , 0.0001403536042230597 ,
                          0.00014447911744128506 , 0.0001403525136559281 , 0.00011971295132635354 ,
                          0.00012384064567884213 , 0.00016099098541837102 , 0.0001362264551541369 ,
                          0.00012797106644915976 , 0.00016099262126906844 , 0.0001155819852724701 ,
                          0.0001403536042230597 , 0.00015686492691657986 , 0.00010732877770175617 ,
                          0.00015686656276727728 , 9.49451493607246e-05 , 0.00011558471169029915 ,
                          7.017625682796405e-05 , 0.00011145701733781056 , 0.00012796888531489654 ,
                          0.00011971240604278774 , 9.49451493607246e-05 , 0.000107332049403151 ,
                          0.00011145429091998153 , 0.00012796888531489654 , 9.494351351002717e-05 ,
                          0.00011558253055603593 , 0.00011971022490852451 , 8.256097573612723e-05 ,
                          0.00011145429091998153 , 4.95366944983895e-05 , 9.081909085893343e-05 ,
                          0.00011145483620354733 , 7.43017700461894e-05 , 7.430340589688683e-05 ,
                          7.017571154439823e-05 , 9.90739342803448e-05 , 6.604747190834385e-05 , 9.494460407715878e-05 ,
                          4.540954542946671e-05 , 9.08158191575386e-05 , 7.430449646401844e-05 , 6.192032283942107e-05 ,
                          6.192032283942107e-05 , 5.779317377049828e-05 , 7.430340589688683e-05 ,
                          5.3664388850878085e-05 , 4.95366944983895e-05 , 7.017789267866146e-05 ,
                          4.9534513364126264e-05 , 5.3665479418009695e-05 , 3.302428123773773e-05 ,
                          7.017625682796405e-05 , 5.366384356731228e-05 , 2.8895496318117525e-05 ,
                          2.064065289670616e-05 , 6.191705113802622e-05 , 4.1280760509846516e-05 ,
                          1.2383628341031566e-05 , 2.8897132168814946e-05 , 4.1276943524885865e-06 ,
                          2.8895496318117525e-05 , 8.255933988542981e-06 , 2.063956232957455e-05 ,
                          1.2383628341031566e-05 , 8.255933988542981e-06 , 1.6512958544217575e-05 ,
                          4.128239636054394e-06]
        wp_score_end = [0.0 , 0.0 , 0.0 , 3.3023735954171925e-05 , 0.0001692474646904798 , 0.00046645290902454 ,
                        0.0009411443029635172 , 0.0017089281695403429 , 0.0023611267791433035 , 0.0032898885443329202 ,
                        0.004185646203776735 , 0.0050607686631595015 , 0.006167049094967984 , 0.006542697635789867 ,
                        0.006778002387492876 , 0.007170148073635568 , 0.007855407498475176 , 0.007706818681039073 ,
                        0.008136107072951374 , 0.008082445410518325 , 0.008210418112818182 , 0.008392041026596206 ,
                        0.008594305683838068 , 0.008226957790257124 , 0.008643863099111958 , 0.008643849467022813 ,
                        0.008775977673137016 , 0.00883374140159496 , 0.008577843981949036 , 0.008639740857595127 ,
                        0.008548931036556812 , 0.008326040449074522 , 0.008272362973418063 , 0.007925595206258022 ,
                        0.008136146878651676 , 0.00807420910673815 , 0.007826558896543717 , 0.0075251864974404305 ,
                        0.007471572274677606 , 0.007805907337975695 , 0.007186736281311867 , 0.007256920172109753 ,
                        0.006988608042959546 , 0.007236264796556769 , 0.006707897562812032 , 0.0071165622056181665 ,
                        0.006509758964071961 , 0.006761565223364305 , 0.006712036162835837 , 0.006336393620133177 ,
                        0.006059844096635666 , 0.006402424188250981 , 0.006010285590794645 , 0.005874055318655546 ,
                        0.005791505793874301 , 0.006460223360140703 , 0.005944232120767076 , 0.005948395258551342 ,
                        0.005675923808601831 , 0.0057130790558934524 , 0.005386975388823446 , 0.005275527641306253 ,
                        0.005213618224138149 , 0.005411743190789074 , 0.005246624511018215 , 0.004866866270201515 ,
                        0.005143439240892356 , 0.005151694629597333 , 0.004941150591173598 , 0.005019588234825763 ,
                        0.0049700559025959 , 0.004763651009479536 , 0.004623310492062056 , 0.004515994801165879 ,
                        0.004747168586815004 , 0.0046439696845999995 , 0.004400414451744106 , 0.004408679655553268 ,
                        0.004532510486127925 , 0.004165113517026059 , 0.004433432189579054 , 0.004429309402778657 ,
                        0.004189899313349359 , 0.004239429464444959 , 0.004243541345574039 , 0.004103197556455164 ,
                        0.004008264948616453 , 0.003929842572904131 , 0.003715170294844341 , 0.0038224985272625315 ,
                        0.0038637961915629184 , 0.0038059839328676167 , 0.003739955545884076 , 0.0038225028895310584 ,
                        0.00370690890802014 , 0.003673907528692166 , 0.003533557741454067 , 0.0035624477849365266 ,
                        0.0035500690641475877 , 0.003471621605391238 , 0.003368431972996852 , 0.003290015050120188 ,
                        0.0034633722148054843 , 0.003372570573020657 , 0.0033849476579588984 , 0.0032033083856739 ,
                        0.0031785454912603636 , 0.0032900035991653056 , 0.0031991861441570697 , 0.003223964851794015 ,
                        0.0030423392115981615 , 0.003211573589483062 , 0.002835928320362574 , 0.0029845449472605317 ,
                        0.0029473842471332526 , 0.0030464630889656895 , 0.0031455282987089815 , 0.0028895905280791885 ,
                        0.002906105667757669 , 0.0029391174074733933 , 0.002910243722497908 , 0.0027822584786760373 ,
                        0.002579995457284873 , 0.0026666895802091467 , 0.0028565662468414505 , 0.0027244691218904996 ,
                        0.002666675948120001 , 0.002654305951868115 , 0.0026377924480403317 , 0.0025263316137175604 ,
                        0.002579986187464254 , 0.0026501798933663235 , 0.0027162137331855226 , 0.002542845117545344 ,
                        0.0026543010443160226 , 0.0025263397929710476 , 0.0027368669276042427 , 0.0023983605472684007 ,
                        0.002613030098910361 , 0.0023281864715746997 , 0.0024231327115025562 , 0.0025923878101629574 ,
                        0.0021837171692375996 , 0.0023488227622028793 , 0.0022415043448888736 , 0.0023694694132188098 ,
                        0.002509827379710396 , 0.002348817854650787 , 0.00221673327122185 , 0.0023736063773919167 ,
                        0.002220876778797747 , 0.0023859807359123296 , 0.002229113082577921 , 0.002224994658046051 ,
                        0.0020763785764316587 , 0.0022208506051865883 , 0.002262138999666356 , 0.0021424320464592263 ,
                        0.0022043632749699634 , 0.0020928969878115343 , 0.0021465630125131097 , 0.00200209752716097 ,
                        0.002059882521677981 , 0.0022043469164629895 , 0.00213830925965883 , 0.0019360260627757308 ,
                        0.0020846503236436097 , 0.002063998765075588 , 0.002055745012221308 , 0.001981442696891553 ,
                        0.0019773188195240246 , 0.002113527825604056 , 0.002167197667290592 , 0.0020392429593484064 ,
                        0.0018699878606880055 , 0.0017585281169323658 , 0.0019360396948648758 , 0.001750274909361652 ,
                        0.0018699878606880055 , 0.001717248446989651 , 0.0019979562007193365 , 0.0017874296113697071 ,
                        0.0018782410682587195 , 0.001874115555040494 , 0.0017709221056611477 , 0.0017296315300471166 ,
                        0.0018328326133963844 , 0.0017296353470320773 , 0.0017833073698528769 , 0.0018617237474459754 ,
                        0.0019236615193595021 , 0.001865867800305438 , 0.001783304098151482 , 0.0018121968680517705 ,
                        0.001634684199552129 , 0.001803944751048188 , 0.0015769019314529468 , 0.001704865909215751 ,
                        0.0016512053373498336 , 0.001614056633461002 , 0.00170073985071396 , 0.0017502705470931255 ,
                        0.00162643426368281 , 0.0016512015203648729 , 0.0016099240315564213 , 0.0015562552804370168 ,
                        0.0017048675450664482 , 0.0017461504867105583 , 0.0015810361692082248 , 0.0016553270335830984 ,
                        0.0015562705483768593 , 0.0016016828202241552 , 0.001651203701499136 , 0.0017709291943475032 ,
                        0.0015438727426631167 , 0.0014737052103722056 , 0.0016388217090088021 , 0.0016635922373922601 ,
                        0.0015108522784103397 , 0.0015232484482733846 , 0.0017007513016688417 , 0.0015191147558016723 ,
                        0.0015397521369969836 , 0.0016016719145528389 , 0.0016181963240519386 , 0.0015686547220014567 ,
                        0.0016966263337341821 , 0.0014654470952493994 , 0.0014902138066478967 , 0.0014282967555098705 ,
                        0.0016099425711976587 , 0.001428307115897621 , 0.0014448086234869566 , 0.0014406787480002047 ,
                        0.0013746274591069003 , 0.0014241734234259084 , 0.0013457286910873879 , 0.0013663737062526203 ,
                        0.0012961947230068272 , 0.0014117854328163505 , 0.0013250918551756421 , 0.0013209723400766408 ,
                        0.0013168375570377967 , 0.0012838094588150982 , 0.001287942606003245 , 0.001271434009727554 ,
                        0.0013498580212905737 , 0.001172345898074498 , 0.001209507143485343 , 0.0011682318358111547 ,
                        0.0011971306038306671 , 0.0013292255476473548 , 0.0012218880454085458 , 0.0012177527170861358 ,
                        0.0012342793077194986 , 0.0010650242090590974 , 0.001126951620584874 , 0.0011310760432359677 ,
                        0.001180610556600094 , 0.001114575080930198 , 0.0010650209373577026 , 0.0009659562728979765 ,
                        0.0011682285641097597 , 0.0010443868278637862 , 0.0009907229842964738 , 0.0010732883223011276 ,
                        0.0010196206617488547 , 0.000994846316380436 , 0.0010113630919096142 , 0.0010113598202082194 ,
                        0.0010526443977030265 , 0.0011104293922200377 , 0.0009122913387635326 , 0.000858631312181181 ,
                        0.0008710078518358571 , 0.0009907180767443815 , 0.0010072250371693753 , 0.0008008310497243272 ,
                        0.0009370558690277666 , 0.0009329314463766729 , 0.0011352048281555878 , 0.0009989843711206748 ,
                        0.0008916463235982999 , 0.0008586258593455229 , 0.0008751382726061747 , 0.0008710073065522912 ,
                        0.0009205429104835491 , 0.0008214733384717308 , 0.0008049587440768157 , 0.0008008294138736297 ,
                        0.0008379841158816851 , 0.0008957821972042756 , 0.0008462482291237152 , 0.0008999060745718036 ,
                        0.0008421123555177395 , 0.0008792697839436239 , 0.0007925767515864817 , 0.000821475519605994 ,
                        0.0008710138499550808 , 0.0007967082629239308 , 0.0007595475627966517 , 0.0007719382798240387 ,
                        0.0007801936685290159 , 0.0007925811138550081 , 0.0008957821972042756 , 0.0007595486533637833 ,
                        0.0007182722551224631 , 0.0006811262776514607 , 0.0007182717098388974 , 0.0008627535536980114 ,
                        0.0007471682967241465 , 0.0007719393703911703 , 0.0006604736285163069 , 0.0006811224606665001 ,
                        0.0007678127666058133 , 0.0006976337833600203 , 0.0006563541134173053 , 0.0006893778493714774 ,
                        0.0007430471457744476 , 0.0007554236854291237 , 0.0007636763477162718 , 0.0006563502964323447 ,
                        0.0006687328342062447 , 0.0007636872533875879 , 0.0007141483777549352 , 0.0007100163211339202 ,
                        0.0006563562945515685 , 0.0006728659813943912 , 0.0006522318719004749 , 0.0007182777079581212 ,
                        0.0006191977755585526 , 0.0006563590209693977 , 0.0007471748401269362 , 0.0007017631135632064 ,
                        0.0006728659813943912 , 0.0006480987247123281 , 0.0005531530300680377 , 0.0006852512455861202 ,
                        0.0005655361131255035 , 0.0005696670791793869 , 0.0005861745848879465 , 0.0006439710303598396 ,
                        0.0005614073282058834 , 0.0005779224678843641 , 0.00063158194918315 , 0.0006315868567352422 ,
                        0.0005820474358190237 , 0.0006398400643059562 , 0.0006192021378270791 , 0.0005160075978806011 ,
                        0.0005861805830071703 , 0.0005614182338771995 , 0.0006439753926283661 , 0.0005242548073320912 ,
                        0.000610943477420707 , 0.0005572807244205263 , 0.0005696583546423341 , 0.0005201352922330897 ,
                        0.0005077445752057026 , 0.0005283830469681456 , 0.0006233303774631334 , 0.0005407704922941378 ,
                        0.0005655410206775958 , 0.0004953598562975395 , 0.0005696654433286895 , 0.0005366422526580834 ,
                        0.0005283857733859747 , 0.0004912332525121825 , 0.00041692384849607183 ,
                        0.00047885344115611165 , 0.0005861800377236046 , 0.0005448970960794948 ,
                        0.00043757049951200183 , 0.00047059259961547634 , 0.0005283825016845798 ,
                        0.0005118771771102835 , 0.00046646817696438265 , 0.0005242542620485254 , 0.0005820490716697211 ,
                        0.0005448960055123631 , 0.00047472029396796497 , 0.0004747241109529256 ,
                        0.00047059259961547634 , 0.0004994913676349887 , 0.00041692821076459827 ,
                        0.0004912332525121825 , 0.00048297677324007373 , 0.000421052633415692 , 0.00038390283895972893 ,
                        0.00047471920340083336 , 0.00044169110517813494 , 0.0005036136091518192 , 0.000511872814841757 ,
                        0.00047059205433191054 , 0.0004705887826305157 , 0.00047059150904834473 ,
                        0.0004499514014352044 , 0.00044169383159596397 , 0.0004210531786992578 ,
                        0.00045407473351916647 , 0.0004499535825694676 , 0.0004417003749987537 ,
                        0.00034262262373344825 , 0.0004086657333732656 , 0.0004045473088413957 , 0.0003550149766115327 ,
                        0.0003591383086954948 , 0.00037152084646939476 , 0.0003921598635154035 , 0.0004293162013741563 ,
                        0.0004045434918564351 , 0.00033436559917777363 , 0.0003962881031514579 , 0.0003673958785347352 ,
                        0.000363264367197286 , 0.00042105426926638943 , 0.00038390392952686054 , 0.0003673926068333404 ,
                        0.00033436668974490524 , 0.00031372876326602806 , 0.0003302417218102457 ,
                        0.0003508796482891228 , 0.00034675304450376584 , 0.0002806990291926323 , 0.0003839006578254657 ,
                        0.00032611348217419125 , 0.0004210537239828236 , 0.00036325945964519376 ,
                        0.00034262207844988244 , 0.000408667369223963 , 0.0002683186725529956 , 0.0003508774671548596 ,
                        0.0003508801935726886 , 0.00030959725192857885 , 0.000264188251782678 , 0.00043344007874168423 ,
                        0.0002972130783039815 , 0.00031786081988704313 , 0.00024767529323846036 ,
                        0.00035913340114340254 , 0.00032198415197100526 , 0.0003219825161203079 ,
                        0.00030133968208933843 , 0.0002930886556528878 , 0.00027244527633835255 ,
                        0.00035500952377587466 , 0.00035500570679091397 , 0.00031373203496742294 ,
                        0.0003013440443578649 , 0.0003095988877792763 , 0.0003095977972121447 , 0.00026006601026584746 ,
                        0.0002518073498594754 , 0.0002518046234416464 , 0.0002394193592499174 , 0.00033436341804351035 ,
                        0.0003591383086954948 , 0.00029308320281722967 , 0.0003219852425381369 , 0.000284834902798608 ,
                        0.0002724485480397474 , 0.0003467514086530684 , 0.00023942535736914126 ,
                        0.00024768129135768427 , 0.0002807028461775929 , 0.0002807072084461194 , 0.0003054711934267877 ,
                        0.0002476774743727236 , 0.00022703791204314904 , 0.0002807023008940271 , 0.0002807066631625536 ,
                        0.0003013434990742991 , 0.00027657351597440693 , 0.00024767911022342105 ,
                        0.00021878197805460609 , 0.0002930859292350587 , 0.00023942263095131223 ,
                        0.00026831540085160075 , 0.0002311677875299009 , 0.0002889582348825701 , 0.0002765784235264992 ,
                        0.00023942426680200965 , 0.00027244691218904996 , 0.0002518068045759096 ,
                        0.00027244527633835255 , 0.0002600649196987158 , 0.0002518013517402516 ,
                        0.00023942154038418062 , 0.0002187847044724351 , 0.00027244800275618157 ,
                        0.0002641926140512044 , 0.00022703845732671485 , 0.0002930870198021903 ,
                        0.00023942208566774643 , 0.000264188251782678 , 0.00021465428370211748 ,
                        0.00023116506111207186 , 0.00023116451582850603 , 0.00023116724224633508 ,
                        0.00020227174592821754 , 0.00018988648173648854 , 0.00023942044981704901 ,
                        0.0001981407798743341 , 0.00022703682147601746 , 0.00022290749127283144 ,
                        0.00021465646483638073 , 0.00022703791204314904 , 0.0002518051687252122 ,
                        0.00018575824210043416 , 0.0002559339536448324 , 0.00018988702702005435 ,
                        0.00025180844042660703]
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs) , passage_len)
        best_start , best_end , max_prob = -1 , -1 , 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx] * wp_score_start[start_idx] * wp_score_end[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        # print(best_start, best_end)
        return (best_start , best_end) , max_prob

    def save(self , model_dir , model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess , os.path.join(model_dir , model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir , model_prefix))

    def restore(self , model_dir , model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess , os.path.join(model_dir , model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir , model_prefix))
