# RingMo-Framework

RingMo-Framework 是由中科院空天信息创新研究院与华为大模型研发团队联合打造的一款用于视觉领域的全国产化自监督预训练开发套件，旨在为业界提供高质量的视觉自监督预训练模型库和降低用户开发大规模预训练模型门槛，其中集成业界如MAE、SIMMIM、A2MMIM、SimCLR等主流的视觉自监督预训练架构和专用于遥感领域的RingMo架构，包含Vit、Vit-moe、Swin(V1-V2)、Swin-moe、MobileVit(V1-V3)、和Resnet等主流Transformer和CNN类骨干网络，覆盖分类、分割、检测等下游任务的微调应用。同时基于Ascend芯片和MindSpore框架做深度优化适配，集成如MindInsight可视化、分布式并行策略配置、MOE专家系统、Profile性能分析、数据性能调优和Modelarts平台适配等模块，可极大提升开发者使用MindSpore开发大规模预训练模型体验。

## 主要特性

* **【无标签预训练】**: 集成所有架构均采用无监督预训练方式，无需数据标注成本，支持任意类型图片进行无监督学习
* **【大规模参数扩充】**: 集成Moe稀疏专家系统，为用户提供丰富的模型参数扩充策略，轻松实现百亿/千亿+大规模预训练模型的分布式训练
* **【并行能力丰富】**: 集成MindSpore数据并行、半自动并行、算子级模型并行、优化器并行、异构并行和专家并行配置模块，用户通过配置文件轻松调用MindSpore各类并行能力
* **【低成本迁移微调】**: 端到端打通模型预训练到微调流程，支持主流视觉Backbone的微调分类验证及分割、检测等复杂下游任务的迁移学习
* **【工具丰富】**: 集成MindInsight、Profile、AutoTune、图算融合和AICC Tools，轻松实现模型训练和性能的可视化分析、数据加载性能的自动调优，网络算子自动融合加速和人工智能计算中心分布式集群训练自动适配等，充分使能MindSpore大规模调试调优和训练加速能力

## 已支持模型

| Arch  | BackBone |
| :---: | :------: |
| RingMo | Swin |
| SimMIM | Vit |
| _ | Swin |
| MAE | Vit |

## 套件架构

![套件架构设计](repo/img.png)

## 快速开始

* Clone 仓库

  ```shell
  git clone https://gitee.com/mindspore/ringmo-framework.git
  ```

* 准备数据

  ```shell
  # 准备图片索引路径的json文件，如下方目录中的train_ids.json
  python ringmo_framework/datasets/tools/get_image_ids.py --image IMAGE_PATH_DIR --file JSON_FILE_NAME
  ```

  ```text
  imageNet-1k/
  ├───train/
  |  ├───n01440764/
  |  |  ├───n01440764_10026.JPEG
  |  |  ├───n01440764_10027.JPEG
  |  |  ├───...
  |  ├───n01443537/
  |  ├───...
  ├───val/
  |  ├───n01440764/
  |  |  ├───n01440764_10026.JPEG
  |  |  ├───n01440764_10027.JPEG
  |  |  ├───...
  |  ├───n01443537/
  |  ├───...
  └───train_ids.json
  ```

* 预训练

  ```shell
  # 单卡预训练
  cd ringmo-framework/
  python pretrain.py --config CONFIG_PATH --use_parallel False
  # 分布式训练
  cd ringmo-framework/
  python ringmo_framework/tools/hccl_tools.py --device_num [0,8] # 生成分布式训练所需的RANK_TABLE_FILE，后面可跳过
  cd scripts
  sh pretrain_distribute.sh RANK_TABILE_FILE CONFIG_FILE # 执行分布式预训练
  ```

* 分类微调

  ```shell
  # 单卡微调
  cd ringmo-framework/
  python finetune.py --config CONFIG_PATH --use_parallel False
  # 分布式微调
  cd scripts
  sh finetune_distribute.sh RANK_TABILE_FILE CONFIG_FILE # 执行分布式微调
  ```

* 分类评估

  ```shell
  # 单卡评估
  cd ringmo-framework/
  python eval.py --config CONFIG_PATH --use_parallel False
  # 分布式评估
  cd scripts
  sh eval_distribute.sh RANK_TABILE_FILE CONFIG_FILE CHECKPOINT_FILE # 执行分布式评估
  ```

* 下游任务迁移（待开放）

  ```shell
  # 安装ringmo-framework库
  pip install ringmo-framework
  ```

  ```python
  # 以使用MAE-Vit-Base-P16-img-224的预训练权重为例进行复杂下游任务迁移学习
  from ringmo-framework.models.backbone.Vit import vit_base_p16
  from ringmo-framework.tools import load_checkpoint
  # 在目标检测网络Faster-RCNN中，使用ringmo-framework的vit_base_p16替换原有的resnet模型作为backbone
  ...
  self.backbone = vit_base_p16(**kwargs)
  ...
  ```

### 使用说明

1. 分布式shell脚本默认使用8卡进行训练和推理，若不使用8卡，需修改脚本中的START_DEVICE、END_DEVICE和RANK_SIZE，如使用设备2、3
   进行双卡训练，需设置START_DEVICE=2、END_DEVICE=3、RANK_SIZE=2，且需使用hccl_tools.py生成对应的RANK_TABLE_FILE
2. 使用hccl_tools.py生成RANK_TABLE_FILE时，device_num的参数区间左包含右不包含，如使用设备2、3进行双卡训练，device_num为[2,4]
3. CONFIG_FILE存放在./config下，使用时可以根据需要修改学习率等参数

### 参与贡献

1. Fork 本仓库
2. 新建 Feat_xxx 分支
3. 提交代码
4. 新建 Pull Request
