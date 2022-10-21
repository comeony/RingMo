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
"""arch of ringmo"""
from ringmo_framework.arch.build_arch import build_model
from ringmo_framework.arch.mae import build_mae, mae_vit_base_p16,\
    mae_vit_large_p16, mae_vit_huge_p14
from ringmo_framework.arch.simmim import build_simmim, simmim_vit_base_p16, simmim_vit_large_p16,\
    simmim_swin_base_p4_w6, simmim_swin_base_p4_w7, simmim_swin_tiny_p4_w6, simmim_swin_tiny_p4_w7
from ringmo_framework.arch.ringmo import build_ringmo, ringmo_vit_base_p16, ringmo_vit_large_p16,\
    ringmo_swin_base_p4_w6, ringmo_swin_base_p4_w7, ringmo_swin_tiny_p4_w6, ringmo_swin_tiny_p4_w7
