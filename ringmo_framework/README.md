# RingMo

## 模型描述

SimMIM是一种基于MIM（Masked Imange Modeling）的无监督学习方法。

SimMIM随机mask掉输入图像的一部分patch，然后通过encoder-decoder来预测masked patchs的原始像素值。

相较于MAE, SimMIM接受结构化的输入，如Swin。

论文：Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., & Hu, H. (2021). SimMIM: A Simple Framework for Masked Image Modeling. ArXiv, abs/2111.09886.

## 模型性能

| Backbone | Pretrain Datasets | Pretrain Epoch | Pretrain Loss | Finetune Datasets | Finetune Epoch | Finetune Loss | Accuracy | Log |
| :---------: | :--------: | :---: | :----: | :-----------: | :---: | :----: | :---: | :---: |
| simmim-vit  | Aircas-50w | 200 | 0.284175 | NWPU_RESISC45 | 200 | 1.974323 | 92.67% | [link](https://pan.baidu.com/s/1-pIg56jfNJV19uOE4l9Aeg?pwd=ph2c) |
| simmim-swim | Aircas-50w | 200 | 0.295708 | NWPU_RESISC45 | 200 | 1.281379 | 94.35% | [link](https://pan.baidu.com/s/1-pIg56jfNJV19uOE4l9Aeg?pwd=ph2c) |
| ringmo-swim | Aircas-200w | 200 | \ | NWPU_RESISC45 | 200 | \ | 95.67% | \ |

> Aircas预训练数据集由中科院空天信息创新研究院收集，数据量200-500w规模，提供的simmim-vit和simmim-swim预训练模型权重仅使用了其中的50w数据集。