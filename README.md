This paper was accepted by INFOCOM 2022.

# Pytorch-Backdoor-Unlearning

This repository contains code for our paper ([Backdoor Defense with Machine Unlearning]()) implemented in Pytorch.

# Requirements
+ Python 3.9.2
+ Pytorch 1.9
+ Torchvision 0.1.8
+ Trojanvision 1.0.8

# Instructions

### Quick Start
We provide a demo for a quick start, directly running the main.py in this repo.
```
python main.py
```
Firstly, you need to install the above packages and enuring your environment is consistent with our environment (different environment may cause some bugs or inconsistent results of our paper).

Besides, the data of MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 should stored (downloaded) in path "..//data".

Default attack method is Badnet with the $3 \times 3$ trigger adopted random position and initialization, where random position is more threat than fixed position.

Ideally, the accuracy and ASR of the obtained model trained with default setting (backdoored model) is about 85% and 100%.

Our defense method also is included in main.py and you will see the results reported in the paper.

If you want to reproduce more experiment results or used in your research, following the below instructions.


### Training Backdoored Model
More information will quickly arrive.

### Erasing Backdoor from the model using our method
More information will quickly arrive.

# Citing this work

Liu Y, Fan M, Chen C, et al. Backdoor Defense with Machine Unlearning[J]. arXiv preprint arXiv:2201.09538, 2022.

# Other source code

Code for NAD (Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks) refer to this [site](https://gitee.com/mcdragon/NAD).

Code for GAN-based defense (Defending Neural Backdoors via Generative Distribution Modeling) refer to this [site](https://github.com/superrrpotato/Defending-Neural-Backdoors-via-Generative-Distribution-Modeling).

Code for Fine-Pruning defense is integrated in the library [TroJanZoo](https://github.com/ain-soph/trojanzoo).
