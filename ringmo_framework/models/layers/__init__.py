# Copyright 2022 Huawei Technologies Co., Ltd
# Copyright 2022 Aerospace Information Research Institute,
# Chinese Academy of Sciences.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""layers of ringmo-framework"""
from ringmo_framework.models.layers.vision_transformer import VisionTransformer
from ringmo_framework.models.layers.attention import Attention, WindowAttention
from ringmo_framework.models.layers.block import Block, SwinTransformerBlock
from ringmo_framework.models.layers.layers import LayerNorm, Linear, Dropout, DropPath, Identity
from ringmo_framework.models.layers.mlp import MLP
from ringmo_framework.models.layers.patch import PatchEmbed, Patchify, UnPatchify
from ringmo_framework.models.layers.utils import _ntuple
