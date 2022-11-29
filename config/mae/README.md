# MAE

## 模型描述

MAE是一种基于MIM（Masked Imange Modeling）的无监督学习方法。

MAE由何凯明团队提出，将NLP领域大获成功的自监督预训练模式用在了计算机视觉任务上，效果拔群，在NLP和CV两大领域间架起了一座更简便的桥梁。

论文：He, Kaiming et al. “Masked Autoencoders Are Scalable Vision Learners.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 15979-15988.

## 模型性能

| Backbone | Pretrain Datasets | Pretrain Epoch | Pretrain Loss | Finetune Datasets | Finetune Epoch | Finetune Loss | Accuracy | Log |                      pretrain_config                      |                      finetune_config                      |
| :---------: |:-----------------:|:--------------:|:-------------:|:-----------------:| :---: |:-------------:|:--------:| :---: |:---------------------------------------------------------:|:---------------------------------------------------------:|
| mae-vit |    imageNet-1k    |      800       |       \       |        imageNet-1k        | 100 |   2.540721    |  82.95%  | \ | [link](pretrain_mae_vit_base_p16_imagenet_224_800ep.yaml) | [link](finetune_mae_vit_base_p16_imagenet_224_100ep.yaml) |

> 当前finetune使用的是从facebook-mae预训练好的权重转换的权重。