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
from .basic_rnn import rnn
from .match_layer import MatchLSTMAttnCell
from .match_layer import AttentionFlowMatchLayer
from .pointer_net import PointerNetDecoder
from .match_layer import SelfMatchingLayer


__all__ = [
    'rnn' ,
    'MatchLSTMAttnCell',
    'AttentionFlowMatchLayer',
    'PointerNetDecoder',
    'SelfMatchingLayer',
]