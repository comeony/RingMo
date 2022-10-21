# RingMo

## 模型描述

中国科学院空天信息创新研究院（以下简称“空天院”）牵头研制首个面向跨模态遥感数据的生成式预训练大模型“空天·灵眸”（RingMo，Remote Sensing Foundation Model），旨在构建一个通用的多模态多任务模型，为遥感领域多行业应用提供一套通用便捷、性能优良的解决方案。

该团队深入结合光学、SAR等跨模态遥感数据的成像机理和目标特性，在模型设计、模型训练、推理优化等方向开展技术创新，并在场景分类、检测定位、细粒度识别、要素提取及变化检测等典型下游任务中进行了验证。该模型在8个国际标准数据集上达到了同类领先水平，有效填补了跨模态生成式预训练模型在遥感专业领域的空白。同时，空天院与华为公司深度技术合作，基于昇腾AI基础软硬件平台，尤其是昇思MindSpore AI框架，将联合打造灵活易用的自监督预训练通用套件，可高效支撑大模型并行训练及下游任务的开发。

论文："RingMo: A Remote Sensing Foundation Model with Masked Image Modeling," in IEEE Transactions on
 Geoscience and Remote Sensing, 2022, doi: 10.1109/TGRS.2022.3194732.

## 模型性能

| Backbone | Pretrain Datasets | Pretrain Epoch | Pretrain Loss | Finetune Datasets | Finetune Epoch | Finetune Loss | Accuracy | Log | pretrain_config | finetune_config |
| :---------: | :--------: | :---: | :----: | :-----------: | :---: | :----: | :---: | :---: | :---: | :---: |
| ringmo-vit | Aircas-200w | 200 | \ | NWPU_RESISC45 | 200 | \ | 95.35% | \ | [link](pretrain_ringmo_vit_base_p16_aircas_224_200ep.yaml) |  \ |
| ringmo-swin | Aircas-200w | 200 | \ | NWPU_RESISC45 | 200 | \ | 95.67% | \ | [link](pretrain_ringmo_swin_base_p4_w6_aircas_192_200ep.yaml) |  [link](finetune_ringmo_swin_base_p4_w7_nwpu_224_200ep.yaml) |

> Aircas预训练数据集由中科院空天信息创新研究院收集，数据量200-500w规模。