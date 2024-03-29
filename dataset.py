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
This module implements data process strategies.
"""

import os
import sys
import json
import logging
import numpy as np
from collections import Counter
sys.path.append('..')
from paragraph_extraction import paragraph_selection, compute_paragraph_score


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,
                 train_files=[], dev_files=[], test_files=[], dataset = "Dureader"):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if dataset == 'Dureader':
            if train_files:
                for train_file in train_files:
                    if train_file.endswith("search.train.json"):

                        self.train_set += self._load_dataset(train_file, train=True, source="search")
                        self.logger.info('Search Train set size: {} questions.'.format(len(self.train_set)))
                    else:
                        self.train_set += self._load_dataset(train_file, train=True, source="zhidao")
                self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

            if dev_files:
                for dev_file in dev_files:
                    if dev_file.endswith("search.dev.json"):

                        self.dev_set += self._load_dataset(dev_file, source="search")
                        self.logger.info('Search Dev set size: {} questions.'.format(len(self.dev_set)))
                    else:
                        self.dev_set += self._load_dataset(dev_file, source="zhidao")

                # with open('/home/congyao/preprocessed/congyao_precessed/dev.json' , 'w') as fout:
                #     for pred_answer in self.dev_set:
                #         fout.write(json.dumps(pred_answer , ensure_ascii=False) + '\n')
                #
                self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

            if test_files:
                for test_file in test_files:
                    if test_file.endswith("search.test.json"):
                        self.test_set += self._load_dataset(test_file, source="search")
                        self.logger.info('Search Test set size: {} questions.'.format(len(self.test_set)))
                    else:
                        self.test_set += self._load_dataset(test_file, source="zhidao")

                # with open('/home/congyao/preprocessed/congyao_precessed/test.json', 'w') as fout:
                #     for pred_answer in self.test_set:
                #         fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

                self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))
        else:
            if train_files:
                for train_file in train_files:
                    self.train_set += self._load_mrc_dataset(train_file , train=True)
                self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

            if dev_files:
                for dev_file in dev_files:
                    self.dev_set += self._load_mrc_dataset(dev_file)
                self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

            if test_files:
                for test_file in test_files:
                    self.test_set += self._load_mrc_dataset(test_file)
                self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False, source =None):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path, encoding='utf-8') as fin:
            data_set = []
            for lidx, line in enumerate(fin):
                """
                if train:
                    if lidx > 70000:
                        continue
                else:
                    if lidx > 2000:
                        continue
                """
                sample = json.loads(line.strip())
                # if sample['question_type'] != 'DESCRIPTION':
                #     continue

                if train:
                    #compute_paragraph_score(sample, mode="train")
                    #paragraph_selection(sample, mode="train", source=source)
                    if len(sample['answer_spans']) == 0:
                        continue
                    # if sample['answer_spans'][0][1] >= self.max_p_len:
                    #     continue
                else:
                    
                    #compute_paragraph_score(sample, mode="test")
                    #paragraph_selection(sample, mode="test", source=source)
                    pass

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['segmented_question']

                sample['passages'] = []
                para_infos = []
                for d_idx, doc in enumerate(sample['documents']):
                    if train:
                        most_related_para = doc['most_related_para']

                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']}
                        )

                    else:

                        if type(sample['question_id']) != int:
                            print(sample['question_id'])
                            continue
                        for para_tokens in doc['segmented_paragraphs']:
                            title_tokens = doc["segmented_title"]
                            question_tokens = sample["segmented_question"]
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            common_with_title = Counter(title_tokens) & Counter(question_tokens)

                            correct_preds = sum(common_with_title.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)

                            para_infos.append((para_tokens, recall_wrt_question, d_idx))

                        # choose k sentence which has the the better recall_wrt_question score to test
                        # this strategy be used to filter sentence

                        # sort sentence by the values =  common with other sentence
                        # counter_infos = []
                        # for k, para_info in enumerate(para_infos):
                        #     sentence = para_info[0]
                        #     values = 0
                        #     for j, para_info in enumerate(para_infos):
                        #         if k != j:
                        #             Counter_with_others = Counter(para_info[0]) & Counter(sentence)
                        #             values += sum(Counter_with_others.values())
                        #     counter_infos.append((sentence, values, len(sentence)))
                        #
                        # counter_infos.sort(key=lambda x: (-x[1], x[2]))
                        # for k, sent_info in enumerate(counter_infos):
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        for k, para_info in enumerate(para_infos[:5]):
                            sample['passages'].append({'passage_tokens': para_info[0]})
                #print(sample)
                data_set.append(sample)
        return data_set

    def _load_mrc_dataset(self , data_path , train=False):
        count =0
        with open(data_path , encoding='utf-8') as fin:
            data_set = []

            for lidx , line in enumerate(fin):
                sample = json.loads(line.strip())
                # if sample['query_type'] != "entity":
                #     continue
                question_tokens = []
                for i in sample['query'].split(" "):
                    #print(i)
                    if i.endswith("."):
                        i.replace(".","")

                    if i.endswith(","):
                        i.replace(",","")

                    if i.endswith("?"):
                        i.replace("?","")
                    i.lower()
                    question_tokens.append(i)

                id = sample['query_id']
                if type(id) != int:
                    continue
                sample['question_tokens'] = question_tokens


                if train:

                    for d_idx , doc in enumerate(sample['passages']):
                        if 'is_selected' in doc:
                            if doc['is_selected'] == 0:
                                continue
                            else:
                                if doc['passage_text'].find(sample['answers'][0]) == -1:
                                    count += 1
                                    print(count)
                                    continue
                                fake_answer = []
                                fake_answer = doc['passage_text'].split()
                                sample['passages'] = []
                                sample['passages'].append({'passage_tokens': fake_answer})
                                con = False
                                if len(sample["answers"]) >0:
                                    try:
                                        sample['answer_passages'] = [
                                            fake_answer.index(sample['answers'][0].split()[2]) - 2 ,
                                            fake_answer.index(sample['answers'][0].split()[2]) + len(
                                                sample['answers'][0].split())]
                                        con = True
                                        sample['answer_passages'] = [fake_answer.index(sample['answers'][0].split()[3]) -3, fake_answer.index(sample['answers'][0].split()[3])+ len(sample['answers'][0].split())]
                                        con = True
                                        sample['answer_passages'] = [
                                            fake_answer.index(sample['answers'][0].split()[2]) - 2 ,
                                            fake_answer.index(sample['answers'][0].split()[2]) + len(
                                                sample['answers'][0].split())]
                                        con = True
                                    except Exception as e:
                                        if con ==False:
                                            sample['answer_passages'] = [0, int(len(fake_answer) /2)]

                            data_set.append(sample)
                        else:
                            continue

                else:

                    for d_idx , doc in enumerate(sample['passages']):
                        answer_noexist = True
                        if d_idx ==0:
                            fake_answer = doc['passage_text'].split()

                        if 'is_selected' in doc:
                            if doc['is_selected'] == 0:
                                continue
                            else:

                                answer_noexist = False
                                fake_answer = doc['passage_text'].split()
                                sample['passages'] = []
                                sample['passages'].append({'passage_tokens': fake_answer})
                    if answer_noexist:
                        sample['passages'] = []
                        sample['passages'].append({'passage_tokens': fake_answer})
                    data_set.append(sample)

        print(len(sample))
        return data_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],
                      'question_token_ids': [],
                      'question_length': [],
                      'passage_token_ids': [],
                      'passage_length': [],
                      'start_id': [],
                      'end_id': []}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids) , self.max_p_len))

                else:
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)

        """
        Dureader
        
        
        
                for sample in batch_data['raw_data']:   #mrc
            if 'answer_passages' in sample:
                batch_data['start_id'].append(sample['answer_passages'][0])
                batch_data['end_id'].append(sample['answer_passages'][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
        """
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_tokens']:
                        yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                try:
                    sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                    for passage in sample['passages']:
                        passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])
                except Exception as e:
                    print(sample)

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
