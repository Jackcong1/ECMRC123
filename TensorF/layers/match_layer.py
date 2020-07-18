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
This module implements the core layer of Match-LSTM and BiDAF
"""

import tensorflow as tf
import tensorflow.contrib as tc



class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context, inputs * attended_context],
                                   -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)


class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state


class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        with tf.variable_scope('bidaf',reuse=tf.AUTO_REUSE):
            scale = 8

            passage_encodes2 = tc.layers.fully_connected(passage_encodes, num_outputs=self.hidden_size * 2, activation_fn=tf.nn.elu)
            question_encodes2 = tc.layers.fully_connected(question_encodes, num_outputs=self.hidden_size * 2, activation_fn=tf.nn.elu)

            sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            #sim_matrix = tc.layers.fully_connected(sim_matrix, num_outputs=self.hidden_size * 2, activation_fn=None)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix , -1), question_encodes2)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix , 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes2),
                                            [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn, passage_encodes2,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)
            return concat_outputs, None


class SelfMatchingLayer(object):
    """
    Implements the self-matching layer.
    """

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def getSelfMatchingCell(self, hidden_size, in_keep_prob=1):
        cell = tc.rnn.LSTMCell(num_units=hidden_size , state_is_tuple=True, reuse=tf.AUTO_REUSE)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=in_keep_prob)
        return cell

    def match(self, passage_encodes, whole_passage_encodes, p_length, q_length):
        with tf.variable_scope('self-matching'):

            # 创建cell
            # whole_passage_encodes 作为整体匹配信息

            cell_fw = self.getSelfMatchingCell(self.hidden_size)
            cell_bw = self.getSelfMatchingCell(self.hidden_size)

            # function:
            # self.context_to_attend = whole_passage_encodes
            # fc_context = W * context_to_attend
            #print(whole_passage_encodes.get_shape())
            #
            # random_attn_vector = tf.Variable(tf.random_normal([1 , self.hidden_size]),
            #                                 trainable=True , name="random_attn_vector")

            U = tf.tanh(tc.layers.fully_connected(passage_encodes , num_outputs=self.hidden_size ,
                                                  activation_fn=tf.nn.elu , biases_initializer=None)
                        + tc.layers.fully_connected(tf.expand_dims(whole_passage_encodes , 1) ,
                                                    num_outputs=self.hidden_size ,
                                                    activation_fn=tf.nn.elu))
            logits = tc.layers.fully_connected(U , num_outputs=1, activation_fn=tf.nn.elu)
            scores = tf.nn.softmax(logits , 1)
            pooled_vector = tf.reduce_sum(passage_encodes * scores , axis=1)
            #s = whole_passage_encodes * scores
            # pooled_vector = tf.reduce_sum(whole_passage_encodes * scores , axis=1)

            input_encodes = tf.concat([passage_encodes , pooled_vector], -1)
            I = tf.sigmoid(tc.layers.fully_connected(input_encodes , num_outputs=self.hidden_size * 4,
                                                  activation_fn=tf.nn.elu , biases_initializer=None))
            input_encodes = tf.matmul(input_encodes , I , transpose_b=True)

            # self.fc_context = tc.layers.fully_connected(whole_passage_encodes , num_outputs=self.hidden_size ,
            #                                             activation_fn=tf.nn.tanh)
            # # print(self.fc_context)
            # ref_vector = passage_encodes
            # batch_size = tf.shape(ref_vector)[0]
            # d = tf.shape(ref_vector)[1]
            # # 求St的tanh部分
            # G = tf.tanh(self.fc_context + tf.expand_dims(
            #     tc.layers.fully_connected(ref_vector , num_outputs=self.hidden_size , activation_fn=None) , 1))
            # # tanh部分乘以bias
            # logits = tc.layers.fully_connected(G , num_outputs=1 , activation_fn=None)
            # # 求a
            # scores = tf.nn.softmax(logits, 1)
            #
            # scores = tf.squeeze(input=scores)
            # # 求c
            # # s = tf.matmul(scores, fc_context)
            # # s= scores * fc_context
            # w_2d = tf.reshape(self.fc_context , [-1 , self.fc_context.get_shape()[-1]])
            # s_2d = tf.reshape(scores , [self.fc_context.get_shape()[-1] , -1])
            #
            # r_2d = tf.matmul(w_2d , s_2d)
            #
            # r_4d = tf.reshape(r_2d , [-1 , self.fc_context.get_shape()[1], scores.get_shape()[1], scores.get_shape()[2]])
            #
            # attended_context = tf.reduce_sum(r_4d, axis=2)

            # birnn inputs

            # self.fc_context = tf.layers.dense(whole_passage_encodes, units=self.hidden_size, activation=tf.nn.tanh, kernel_regularizer=tc.layers.l2_regularizer(0.03))
            #
            # ref_vector = passage_encodes
            # sim_matrix = tf.matmul(passage_encodes , whole_passage_encodes , transpose_b=True)
            # context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix , -1) , whole_passage_encodes)
            #
            # self.context2question_attn = tf.layers.dense(context2question_attn, units=self.hidden_size, activation=tf.nn.tanh, kernel_regularizer=tc.layers.l2_regularizer(0.03))
            # # 求St的tanh部分
            # G = tf.tanh(self.fc_context + self.context2question_attn)
            # # tanh部分乘以bias
            #
            # logits = tf.layers.dense(G, units=1, activation=tf.nn.tanh)
            # # 求a
            # scores = tf.nn.softmax(logits, 1)
            # # 求c
            # attended_context = tf.expand_dims(tf.reduce_sum(passage_encodes * scores, axis=1), axis=1)
            # # birnn inputs
            # input_encodes = tf.concat([ref_vector, attended])

            # fc_mean , fc_var = tf.nn.moments(
            #     input_encodes ,
            #     axes=[2]
            # )
            # scale = tf.Variable(tf.ones([input_encodes.get_shape()[-1]]))
            # shift = tf.Variable(tf.zeros([input_encodes.get_shape()[-1]]))
            #
            # epsilon = 0.001
            #
            # ema = tf.train.ExponentialMovingAverage(decay=0.5)
            #
            # def mean_var_with_upadte():
            #     ema_apply_op = ema.apply([fc_mean , fc_var])
            #     with tf.control_dependencies([ema_apply_op]):
            #         return tf.identity(fc_mean) , tf.identity(fc_var)
            #
            # mean , var = mean_var_with_upadte()
            # input_encodes = tf.nn.batch_normalization(input_encodes , mean , var , shift , scale ,
            #                                           epsilon)

            """
            gated
            g_t = tf.sigmoid( tc.layers.fully_connected(whole_passage_encodes,num_outputs=self.hidden_size,activation_fn = None) )
            v_tP_c_t_star = tf.squeeze(tf.multiply(input_encodes , g_t))
            input_encodes = v_tP_c_t_star
            """

            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=input_encodes,
                                                             dtype=tf.float32)

            match_outputs = tf.concat(outputs, 2)
            match_state = tf.concat([state, state], 1)

            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
        return match_outputs, match_state


