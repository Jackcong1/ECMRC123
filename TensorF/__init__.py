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
Empty __init__.py file


"""
from .layers.basic_rnn import rnn
from .layers.match_layer import MatchLSTMAttnCell
from .layers.match_layer import AttentionFlowMatchLayer
from .layers.pointer_net import PointerNetDecoder
from .layers.match_layer import SelfMatchingLayer
from .paragraph_extraction import paragraph_selection, compute_paragraph_score


__all__ = [
    'rnn' ,
    'MatchLSTMAttnCell',
    'AttentionFlowMatchLayer',
    'PointerNetDecoder',
    'SelfMatchingLayer',
    'paragraph_extraction',
    'compute_paragraph_score',
]