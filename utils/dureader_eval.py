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
This module computes evaluation metrics for DuReader dataset.
"""


import argparse
import json
import sys
import zipfile

from collections import Counter
from .bleu_metric.bleu import Bleu
from .rouge_metric.rouge import Rouge

import collections
import re
import six
import tensorflow as tf
import numpy as np

EMPTY = ''
YESNO_LABELS = set(['Yes', 'No', 'Depends'])

def gelu(x):
    """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x , 3)))))
    return x * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

    # We assume that anything that"s not a string is already an activation
    # function, so we just return it.
    if not isinstance(activation_string , six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars , init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$" , name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name , var) = (x[0] , x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map , initialized_variable_names)


def dropout(input_tensor , dropout_prob):
    """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor , 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor , name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor , begin_norm_axis=-1 , begin_params_axis=-1 , scope=name)


def layer_norm_and_dropout(input_tensor , dropout_prob , name=None):
    """Runs layer normalization followed by dropout."""
    output_tensor = layer_norm(input_tensor , name)
    output_tensor = dropout(output_tensor , dropout_prob)
    return output_tensor


def embedding_postprocessor(input_tensor ,
                            use_token_type=False ,
                            token_type_ids=None ,
                            token_type_vocab_size=16 ,
                            token_type_embedding_name="token_type_embeddings" ,
                            use_position_embeddings=True ,
                            position_embedding_name="position_embeddings" ,
                            initializer_range=0.02 ,
                            max_position_embeddings=512 ,
                            dropout_prob=0.1):
    """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
    input_shape = get_shape_list(input_tensor , expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        token_type_table = tf.get_variable(
            name=token_type_embedding_name ,
            shape=[token_type_vocab_size , width] ,
            initializer=create_initializer(initializer_range))
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids , [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids , depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids , token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings ,
                                           [batch_size , seq_length , width])
        output += token_type_embeddings

    if use_position_embeddings:

        assert_op = tf.assert_less_equal(seq_length , max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            full_position_embeddings = tf.get_variable(
                name=position_embedding_name ,
                shape=[max_position_embeddings , width] ,
                initializer=create_initializer(initializer_range))
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(full_position_embeddings , [0 , 0] ,
                                           [seq_length , -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length , width])
            position_embeddings = tf.reshape(position_embeddings ,
                                             position_broadcast_shape)
            output += position_embeddings

    output = layer_norm_and_dropout(output , dropout_prob)
    return output


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.truncated_normal_initializer(stddev=initializer_range)


def get_shape_list(tensor , expected_rank=None , name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor , expected_rank , name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index , dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def assert_rank(tensor , expected_rank , name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank , six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    # if actual_rank not in expected_rank_dict:
    #     scope_name = tf.get_variable_scope().name
    #     raise ValueError(
    #         "For the tensor `%s` in scope `%s`, the actual rank "
    #         "`%d` (shape = %s) is not equal to the expected rank `%s`" %
    #         (name , scope_name , actual_rank , str(tensor.shape) , str(expected_rank)))

def normalize(s):
    """
    Normalize strings to space joined chars.

    Args:
        s: a list of strings.

    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(' '.join(tokens))
    return normalized


def data_check(obj, task):
    """
    Check data.

    Raises:
        Raises AssertionError when data is not legal.
    """
    assert 'question_id' in obj, "Missing 'question_id' field."
    assert 'question_type' in obj, \
            "Missing 'question_type' field. question_id: {}".format(obj['question_type'])

    assert 'yesno_answers' in obj, \
            "Missing 'yesno_answers' field. question_id: {}".format(obj['question_id'])
    assert isinstance(obj['yesno_answers'], list), \
            r"""'yesno_answers' field must be a list, if the 'question_type' is not
            'YES_NO', then this field should be an empty list.
            question_id: {}""".format(obj['question_id'])

    assert 'entity_answers' in obj, \
            "Missing 'entity_answers' field. question_id: {}".format(obj['question_id'])
    assert isinstance(obj['entity_answers'], list) \
            and len(obj['entity_answers']) > 0, \
            r"""'entity_answers' field must be a list, and has at least one element,
            which can be a empty list. question_id: {}""".format(obj['question_id'])


def read_file(file_name, task, is_ref=False):
    """
    Read predict answers or reference answers from file.

    Args:
        file_name: the name of the file containing predict result or reference
                   result.

    Returns:
        A dictionary mapping question_id to the result information. The result
        information itself is also a dictionary with has four keys:
        - question_type: type of the query.
        - yesno_answers: A list of yesno answers corresponding to 'answers'.
        - answers: A list of predicted answers.
        - entity_answers: A list, each element is also a list containing the entities
                    tagged out from the corresponding answer string.
    """
    def _open(file_name, mode, zip_obj=None):
        if zip_obj is not None:
            return zip_obj.open(file_name, mode)
        return open(file_name, mode)

    results = {}
    keys = ['answers', 'yesno_answers', 'entity_answers', 'question_type']
    if is_ref:
        keys += ['source']

    zf = zipfile.ZipFile(file_name, 'r') if file_name.endswith('.zip') else None
    file_list = [file_name] if zf is None else zf.namelist()

    for fn in file_list:
        for line in _open(fn, 'r', zip_obj=zf):
            try:
                obj = json.loads(line.strip())
            except ValueError:
                raise ValueError("Every line of data should be legal json")
            data_check(obj, task)
            qid = obj['question_id']
            assert qid not in results, "Duplicate question_id: {}".format(qid)
            results[qid] = {}
            for k in keys:
                results[qid][k] = obj[k]
    return results


def compute_bleu_rouge(pred_dict, ref_dict, bleu_order=4):
    """
    Compute bleu and rouge scores.
    """
    assert set(pred_dict.keys()) == set(ref_dict.keys()), \
            "missing keys: {}".format(set(ref_dict.keys()) - set(pred_dict.keys()))
    scores = {}
    bleu_scores, _ = Bleu(bleu_order).compute_score(ref_dict, pred_dict)
    for i, bleu_score in enumerate(bleu_scores):
        scores['Bleu-%d' % (i + 1)] = bleu_score
    rouge_score, _ = Rouge().compute_score(ref_dict, pred_dict)
    scores['Rouge-L'] = rouge_score
    return scores


def local_prf(pred_list, ref_list):
    """
    Compute local precision recall and f1-score,
    given only one prediction list and one reference list
    """
    common = Counter(pred_list) & Counter(ref_list)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    p = 1.0 * num_same / len(pred_list)
    r = 1.0 * num_same / len(ref_list)
    f1 = (2 * p * r) / (p + r)
    return f1


def compute_prf(pred_dict, ref_dict):
    """
    Compute precision recall and f1-score.
    """
    pred_question_ids = set(pred_dict.keys())
    ref_question_ids = set(ref_dict.keys())
    correct_preds, total_correct, total_preds = 0, 0, 0
    for question_id in ref_question_ids:
        pred_entity_list = pred_dict.get(question_id, [[]])
        assert len(pred_entity_list) == 1, \
            'the number of entity list for question_id {} is not 1.'.format(question_id)
        pred_entity_list = pred_entity_list[0]
        all_ref_entity_lists = ref_dict[question_id]
        best_local_f1 = 0
        best_ref_entity_list = None
        for ref_entity_list in all_ref_entity_lists:
            local_f1 = local_prf(pred_entity_list, ref_entity_list)[2]
            if local_f1 > best_local_f1:
                best_ref_entity_list = ref_entity_list
                best_local_f1 = local_f1
        if best_ref_entity_list is None:
            if len(all_ref_entity_lists) > 0:
                best_ref_entity_list = sorted(all_ref_entity_lists,
                        key=lambda x: len(x))[0]
            else:
                best_ref_entity_list = []
        gold_entities = set(best_ref_entity_list)
        pred_entities = set(pred_entity_list)
        correct_preds += len(gold_entities & pred_entities)
        total_preds += len(pred_entities)
        total_correct += len(gold_entities)
    p = float(correct_preds) / total_preds if correct_preds > 0 else 0
    r = float(correct_preds) / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return {'Precision': p, 'Recall': r, 'F1': f1}


def prepare_prf(pred_dict, ref_dict):
    """
    Prepares data for calculation of prf scores.
    """
    preds = {k: v['entity_answers'] for k, v in pred_dict.items()}
    refs = {k: v['entity_answers'] for k, v in ref_dict.items()}
    return preds, refs


def filter_dict(result_dict, key_tag):
    """
    Filter a subset of the result_dict, where keys ends with 'key_tag'.
    """
    filtered = {}
    for k, v in result_dict.items():
        if k.endswith(key_tag):
            filtered[k] = v
    return filtered


def get_metrics(pred_result, ref_result, task, source):
    """
    Computes metrics.
    """
    metrics = {}

    ref_result_filtered = {}
    pred_result_filtered = {}
    if source == 'both':
        ref_result_filtered = ref_result
        pred_result_filtered = pred_result
    else:
        for question_id, info in ref_result.items():
            if info['source'] == source:
                ref_result_filtered[question_id] = info
                if question_id in pred_result:
                    pred_result_filtered[question_id] = pred_result[question_id]

    if task == 'main' or task == 'all' \
            or task == 'description':
        pred_dict, ref_dict = prepare_bleu(pred_result_filtered,
                ref_result_filtered,
                task)
        metrics = compute_bleu_rouge(pred_dict, ref_dict)
    elif task == 'yesno':
        pred_dict, ref_dict = prepare_bleu(pred_result_filtered,
                ref_result_filtered,
                task)
        keys = ['Yes', 'No', 'Depends']
        preds = [filter_dict(pred_dict, k) for k in keys]
        refs = [filter_dict(ref_dict, k) for k in keys]

        metrics = compute_bleu_rouge(pred_dict, ref_dict)

        for k, pred, ref in zip(keys, preds, refs):
            m = compute_bleu_rouge(pred, ref)
            k_metric = [(k + '|' + key, v) for key, v in m.items()]
            metrics.update(k_metric)

    elif task == 'entity':
        pred_dict, ref_dict = prepare_prf(pred_result_filtered,
                ref_result_filtered)
        pred_dict_bleu, ref_dict_bleu = prepare_bleu(pred_result_filtered,
                ref_result_filtered,
                task)
        metrics = compute_prf(pred_dict, ref_dict)
        metrics.update(compute_bleu_rouge(pred_dict_bleu, ref_dict_bleu))
    else:
        raise ValueError("Illegal task name: {}".format(task))

    return metrics


def prepare_bleu(pred_result, ref_result, task):
    """
    Prepares data for calculation of bleu and rouge scores.
    """
    pred_list, ref_list = [], []
    qids = ref_result.keys()
    for qid in qids:
        if task == 'main':
            pred, ref = get_main_result(qid, pred_result, ref_result)
        elif task == 'yesno':
            pred, ref = get_yesno_result(qid, pred_result, ref_result)
        elif task == 'all':
            pred, ref = get_all_result(qid, pred_result, ref_result)
        elif task == 'entity':
            pred, ref = get_entity_result(qid, pred_result, ref_result)
        elif task == 'description':
            pred, ref = get_desc_result(qid, pred_result, ref_result)
        else:
            raise ValueError("Illegal task name: {}".format(task))
        if pred and ref:
            pred_list += pred
            ref_list += ref
    pred_dict = dict(pred_list)
    ref_dict = dict(ref_list)
    for qid, ans in ref_dict.items():
        ref_dict[qid] = normalize(ref_dict[qid])
        pred_dict[qid] = normalize(pred_dict.get(qid, [EMPTY]))
        if not ans or ans == [EMPTY]:
            del ref_dict[qid]
            del pred_dict[qid]

    for k, v in pred_dict.items():
        assert len(v) == 1, \
            "There should be only one predict answer. question_id: {}".format(k)
    return pred_dict, ref_dict


def get_main_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'main'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    ref_ans = ref_result[qid]['answers']
    if not ref_ans:
        ref_ans = [EMPTY]
    pred_ans = pred_result.get(qid, {}).get('answers', [])[:1]
    if not pred_ans:
        pred_ans = [EMPTY]

    return [(qid, pred_ans)], [(qid, ref_ans)]


def get_entity_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'entity'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    if ref_result[qid]['question_type'] != 'ENTITY':
        return None, None
    return get_main_result(qid, pred_result, ref_result)


def get_desc_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'description'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    if ref_result[qid]['question_type'] != 'DESCRIPTION':
        return None, None
    return get_main_result(qid, pred_result, ref_result)


def get_yesno_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'yesno'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    def _uniq(li, is_ref):
        uniq_li = []
        left = []
        keys = set()
        for k, v in li:
            if k not in keys:
                uniq_li.append((k, v))
                keys.add(k)
            else:
                left.append((k, v))

        if is_ref:
            dict_li = dict(uniq_li)
            for k, v in left:
                dict_li[k] += v
            uniq_li = [(k, v) for k, v in dict_li.items()]
        return uniq_li

    def _expand_result(uniq_li):
        expanded = uniq_li[:]
        keys = set([x[0] for x in uniq_li])
        for k in YESNO_LABELS - keys:
            expanded.append((k, [EMPTY]))
        return expanded

    def _get_yesno_ans(qid, result_dict, is_ref=False):
        if qid not in result_dict:
            return [(str(qid) + '_' + k, v) for k, v in _expand_result([])]
        yesno_answers = result_dict[qid]['yesno_answers']
        answers = result_dict[qid]['answers']
        lbl_ans = _uniq([(k, [v]) for k, v in zip(yesno_answers, answers)], is_ref)
        ret = [(str(qid) + '_' + k, v) for k, v in _expand_result(lbl_ans)]
        return ret

    if ref_result[qid]['question_type'] != 'YES_NO':
        return None, None

    ref_ans = _get_yesno_ans(qid, ref_result, is_ref=True)
    pred_ans = _get_yesno_ans(qid, pred_result)
    return pred_ans, ref_ans


def get_all_result(qid, pred_result, ref_result):
    """
    Prepare answers for task 'all'.

    Args:
        qid: question_id.
        pred_result: A dict include all question_id's result information read
                     from args.pred_file.
        ref_result: A dict incluce all question_id's result information read
                    from args.ref_file.
    Returns:
        Two lists, the first one contains predict result, the second
        one contains reference result of the same question_id. Each list has
        elements of tuple (question_id, answers), 'answers' is a list of strings.
    """
    if ref_result[qid]['question_type'] == 'YES_NO':
        return get_yesno_result(qid, pred_result, ref_result)
    return get_main_result(qid, pred_result, ref_result)


def format_metrics(metrics, task, err_msg):
    """
    Format metrics. 'err' field returns any error occured during evaluation.

    Args:
        metrics: A dict object contains metrics for different tasks.
        task: Task name.
        err_msg: Exception raised during evaluation.
    Returns:
        Formatted result.
    """
    result = {}
    sources = ["both", "search", "zhidao"]
    if err_msg is not None:
        return {'errorMsg': str(err_msg), 'errorCode': 1, 'data': []}
    data = []
    if task != 'all' and task != 'main':
        sources = ["both"]

    if task == 'entity':
        metric_names = ["Bleu-4", "Rouge-L"]
        metric_names_prf = ["F1", "Precision", "Recall"]
        for name in metric_names + metric_names_prf:
            for src in sources:
                obj = {
                    "name": name,
                    "value": round(metrics[src].get(name, 0) * 100, 2),
                    "type": src,
                    }
                data.append(obj)
    elif task == 'yesno':
        metric_names = ["Bleu-4", "Rouge-L"]
        details = ["Yes", "No", "Depends"]
        src = sources[0]
        for name in metric_names:
            obj = {
                "name": name,
                "value": round(metrics[src].get(name, 0) * 100, 2),
                "type": 'All',
                }
            data.append(obj)
            for d in details:
                obj = {
                    "name": name,
                    "value": \
                        round(metrics[src].get(d + '|' + name, 0) * 100, 2),
                    "type": d,
                    }
                data.append(obj)
    else:
        metric_names = ["Bleu-4", "Rouge-L"]
        for name in metric_names:
            for src in sources:
                obj = {
                    "name": name,
                    "value": \
                        round(metrics[src].get(name, 0) * 100, 2),
                    "type": src,
                    }
                data.append(obj)

    result["data"] = data
    result["errorCode"] = 0
    result["errorMsg"] = "success"

    return result


def main(args):
    """
    Do evaluation.
    """
    err = None
    metrics = {}
    try:
        pred_result = read_file(args.pred_file, args.task)
        ref_result = read_file(args.ref_file, args.task, is_ref=True)
        sources = ['both', 'search', 'zhidao']
        if args.task not in set(['main', 'all']):
            sources = sources[:1]
        for source in sources:
            metrics[source] = get_metrics(
                    pred_result, ref_result, args.task, source)
    except ValueError as ve:
        err = ve
    except AssertionError as ae:
        err = ae

    print(json.dumps(
            format_metrics(metrics, args.task, err),
            ensure_ascii=False).encode('utf8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', help='predict file')
    parser.add_argument('ref_file', help='reference file')
    parser.add_argument('task',
            help='task name: Main|Yes_No|All|Entity|Description')

    args = parser.parse_args()
    args.task = args.task.lower().replace('_', '')
    main(args)
