# Semi-supervised Domain Adaptive Structure Learning - ASDA

This repo contains the source code and dataset for our ASDA paper.


![ASDA](/Figs/ASDA.png)
<b> Illustration of the proposed Adaptive Structure Learning for Semi-supervised Domain Adaptation (ASDA) including three modules: 1) a deep feature encoder network, 2) a source-scattering classifier network, and 3) a target-clustering classifier network.The raw data will be transformed into different formats as inputs according to the WeakAug and StrongAug operations. In this figure, both generators (in yellow) share the parameters for feature extraction. The two classifiers will take the features from the generator for classification. </b>

## Introduction
Semi-supervised domain adaptation (SSDA) is quite a challenging problem requiring methods to overcome both 1) overfitting towards poorly annotated data and 2) distribution shift across domains. Unfortunately, a simple combination of domain adaptation (DA) and semi-supervised learning (SSL)  methods often fail to address such two objects because of training data bias towards labeled samples. In this paper, we introduce an adaptive structure learning method to regularize the cooperation of SSL and DA. Inspired by the multi-views learning, our proposed framework is composed of a shared feature encoder network and two classifier networks, trained for contradictory purposes. Among them, one of the classifiers is applied to group target features to improve intra-class density, enlarging the gap of categorical clusters for robust representation learning. Meanwhile, the other classifier, serviced as a regularizer, attempts to scatter the source features to enhance the smoothness of the decision boundary. The iterations of target clustering and source expansion make the target features being well-enclosed inside the dilated boundary of the corresponding source points. For the joint address of cross-domain features alignment and partially labeled data learning, we apply the maximum mean discrepancy (MMD) distance minimization and self-training (ST) to project the contradictory structures into a shared view to make the reliable final decision. The experimental results over the standard SSDA benchmarks, including DomainNet and Office-home, demonstrate both the accuracy and robustness of our method over the state-of-the-art approaches.

## Dataset
The data processing follows the protocol of [MME](https://github.com/VisionLearningGroup/SSDA_MME).

To get data, run

`sh download_data.sh`

The images will be stored in the following way.

`../data/multi/real/category_name`,

`../data/multi/sketch/category_name`

The dataset split files are stored as follows,

`../data/txt/multi/labeled_source_images_real.txt`,

`../data/txt/multi/unlabeled_target_images_sketch_3.txt`,

`../data/txt/multi/validation_target_images_sketch_3.txt`.

The office and office home datasets are organized in the following ways,

 `../data/office/amazon/category_name`,
 
 `../data/office_home/Real/category_name`.
 
The dataset split files of office or office_home are stored as follows,

`../data/txt/office/labeled_source_images_amazon.txt`,

`../data/txt/office_home/unlabeled_target_images_Art_3.txt`,


## Requirements
`pip install -r requirements.txt`


## Train & Test
If you run the experiment on one adaptation scanerio, like real to sketch of the DomainNet,
```
python main_asda.py --dataset multi --source real --target sketch --num 3 --lr 0.01
```
or run experiments on all adaptation scenarios.
```
bash train_domainnet.sh
```

## To Do
```
- [x] Datasets Processing
- [x] DomainNet Training
- [ ] OfficeHome Training
```
The remaining implementations are coming soon.

## Acknowledgement
We would like to thank the [MME](https://github.com/VisionLearningGroup/SSDA_MME), [RandAugment](https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py) and [UODA](https://github.com/canqin001/UODA) which we used for this implementation.
