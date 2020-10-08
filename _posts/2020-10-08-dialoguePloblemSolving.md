---
title:  从“对话”的视角设计高三解题教学
categories:
- 数学通报
tags:
- 解题教学
---

&emsp;&emsp;解题教学是高三数学教学的主要形式之一,师生都为此花费了不少精力,但现实中的教学效果却不尽如人意,不少学生的反映是上课听得懂但自己课后不会做,以致高考命题人员对高考的预期成绩与考生的实际表现经常存在较大的落差,问题的症结在哪里呢?在调查中发现:学生在解题中不能有效整合题目的信息,对题意的理解支离破碎,走不进题目描绘的"世界"中去,进入不了解题的"角色",也就不能快速寻找到解题的 batch size太大，做归一化绰绰有余，且训练时间能大幅缩短，但是GPU显存吃不消，只能在多卡上做数据并行训练，多卡数据间如何求BN也是个问题，同时由于一个epoch内steps少了，模型迭代次数不够，也会掉点；batch size太小，数据间方差波动太大，不具有统计性，此时做BN并不能对数据做有效归一化，也会影响最终精度。本文就这些问题做一些文献调研和记录，涉及Linear Scaling Rule、Gradual warmup、Cross-GPU BN、Cross-Iteration BN、Filter Response Normalization Layer等。
<!-- more -->

***
>+ # 与命题者本人进行对话

&emsp;&emsp;   数学题是人命制的，因此题目里不可避免地留下命题者的印记，或明或暗地体现了命题者的意图与设想，寻找命题者留下的编制“痕迹”与“破绽”，
接收命题者传递给我们的信息暗示或指引，如有时为了控制难度与得分，常在前面人为增加一问（实质是命题者为减少难度而设置的“台阶”），如果你感到命题者内心
的思维活动与释放出来的“善意”，与其进行隔空“对话”，并进行判断、加工，你将露出会心的微笑：我翻译了命题者的思维密码，一切都明白了！

>> ## 案例 1

![](/assets/images/bn/10.png)
&emsp;&emsp;随着数据集的不断扩充，模型的训练时间变得越来越长。为了节省训练时间，需要使用大的batch size（单卡上普通batch size 256，多卡合并成更大的batch size），从上图也可以看出batch size增大后每个epoch需要的时间能迅速下降。但在数据总量固定的情况下，一个epoch下step变少了，意味着可供模型优化调整的迭代次数少了，模型学习得不够充分，反而会造成精度的下降（如下图）。做一个形象的比喻，小步伐（小learning rate）下迈的步数（step）变少了，导致没能到达全局最优点。为了解决这个问题，需要使用大的learning rate，大步伐下迈少的步数，就能达到“小步伐下迈多的步数”同样的效果。
![](/assets/images/bn/7.png)
&emsp;&emsp;但是这个大的learning rate需要取多大？取大了会不会造成训练不稳定的情况？于是这篇paper便针对这两个问题提出了Linear Scaling Rule和Gradual warmup。

>> ## 分析

&emsp;&emsp;batch size为n，lr为$\eta$时，单步的SGD更新法则如下：
![](/assets/images/bn/1.png)
&emsp;&emsp;假设现在设超大的batch size为kn，比原来大了k倍，lr为$\hat{\eta}$，此时SGD变为：
![](/assets/images/bn/2.png)
&emsp;&emsp;为了能跟普通SGD下k步迭代的过程（如下）相同，需要有$\hat\omega_{t+1}=\omega_{t+k}$。
![](/assets/images/bn/3.png)
&emsp;&emsp;假设$\triangledown l(x,\omega_{t})\approx\triangledown l(x,\omega_{t+j})$，即梯度在训练过程中变化平稳，在临近step间保持近似相等。那么为了能满足kn与n下的近似目标，可以推出：
![](/assets/images/bn/4.png)
&emsp;&emsp;这就是 ***Linear Scaling Rule：当batch size扩大k倍时，为了保持训练过程不变，需要把lr也扩大k倍*** 。从下图可以看出，大batch size $kn=8k,\eta=0.1\cdot 32$的error，与$n=256,\eta=0.1$时的error比较接近，没有太大性能损失。
![](/assets/images/bn/8.png)
&emsp;&emsp;但是这条法则的基础是建立在梯度不变的假设上，在训练的初期，模型的loss迅速地下降，显然不满足这种假设，于是需要采用折中的策略warm up来解决。

>+ # 与题目的条件（或结论）进行对话

&emsp;&emsp;早在resnet中，就采用了warm up来解决模型训练初期不稳定的问题。在一开始使用0.01的lr直到训练loss下降至80%以下（约400步），转而使用0.1的大lr继续训练。然而作者尝试了这种方法，效果并不好（如下图(b)），所以在此基础上进行了改进，提出了渐变式的warm up。具体如下。
![](/assets/images/bn/9.png)
&emsp;&emsp;在batch size为kn的情况下，使用原本的$lr=\eta$作为起步，然后在5个epoch内线性递增到$lr=k\eta$。这样就能进行平稳的过渡，避免了之前的硬转折。在后续的训练中，也可以使用任意的lr下降策略。从上图(c)可以看出，使用了Gradual warmup策略后，训练曲线除去初期过渡阶段在后期几乎完全与baseline重合，说明大batch size的影响几乎被消除。

>> 案例 2 

&emsp;&emsp;文中的BN并没有在多GPU间进行同步，而是在各自GPU的local batch size上进行计算，应用在各自的local batch数据上，随后计算出local loss。但是最终对local loss进行了同步，求和后得出了总loss。
![](/assets/images/bn/5.png)
![](/assets/images/bn/6.png)
&emsp;&emsp;作者认为如果每个GPU的local batch size保持相同，那么是在相同的数据方差上对local数据进行归一化，由此得到的local loss是独立同分布的，不影响训练过程。原话是`the mean/variance statistics computed by BN with different n exhibit different levels of random variation.`和` if the per-worker sample size n is kept fixed and the total minibatch size is kn, it can be viewed a minibatch of k samples with each sample Bj independently selected from Xn, so the underlying loss function is unchanged and is still defined in Xn`。

>> ## 分析

&emsp;&emsp;-- 假如loss中带有L2正则项时，该项中也含有学习率lr，不要忘了对该项进行缩放；  
&emsp;&emsp;-- 带动量矫正项的SGD需要对动量项也进行缩放；  
&emsp;&emsp;-- 多个GPU在进行loss同步时，`allreduce`是相加操作而不是相加后取平均，因此需要事先对local loss除以`kn`，而不是`n`。  
&emsp;&emsp;-- 每个epoch需要对数据进行random shuffle，然后分配给各个GPU。这里的shuffle需要统一进行或者采用相同的random seed，否则不是真正的random均匀分配。

>+ # 让题目的条件与结论进行对话

&emsp;&emsp;早在resnet中，就采用了warm up来解决模型训练初期不稳定的问题。在一开始使用0.01的lr直到训练loss下降至80%以下（约400步），转而使用0.1的大lr继续训练。然而作者尝试了这种方法，效果并不好（如下图(b)），所以在此基础上进行了改进，提出了渐变式的warm up。具体如下。
![](/assets/images/bn/9.png)
&emsp;&emsp;在batch size为kn的情况下，使用原本的$lr=\eta$作为起步，然后在5个epoch内线性递增到$lr=k\eta$。这样就能进行平稳的过渡，避免了之前的硬转折。在后续的训练中，也可以使用任意的lr下降策略。从上图(c)可以看出，使用了Gradual warmup策略后，训练曲线除去初期过渡阶段在后期几乎完全与baseline重合，说明大batch size的影响几乎被消除。

>> 案例 3 

&emsp;&emsp;文中的BN并没有在多GPU间进行同步，而是在各自GPU的local batch size上进行计算，应用在各自的local batch数据上，随后计算出local loss。但是最终对local loss进行了同步，求和后得出了总loss。
![](/assets/images/bn/5.png)
![](/assets/images/bn/6.png)
&emsp;&emsp;作者认为如果每个GPU的local batch size保持相同，那么是在相同的数据方差上对local数据进行归一化，由此得到的local loss是独立同分布的，不影响训练过程。原话是`the mean/variance statistics computed by BN with different n exhibit different levels of random variation.`和` if the per-worker sample size n is kept fixed and the total minibatch size is kn, it can be viewed a minibatch of k samples with each sample Bj independently selected from Xn, so the underlying loss function is unchanged and is still defined in Xn`。

>> ## 分析

&emsp;&emsp;-- 假如loss中带有L2正则项时，该项中也含有学习率lr，不要忘了对该项进行缩放；  
&emsp;&emsp;-- 带动量矫正项的SGD需要对动量项也进行缩放；  
&emsp;&emsp;-- 多个GPU在进行loss同步时，`allreduce`是相加操作而不是相加后取平均，因此需要事先对local loss除以`kn`，而不是`n`。  
&emsp;&emsp;-- 每个epoch需要对数据进行random shuffle，然后分配给各个GPU。这里的shuffle需要统一进行或者采用相同的random seed，否则不是真正的random均匀分配。

>+ # 与题目的背景进行对话

&emsp;&emsp;早在resnet中，就采用了warm up来解决模型训练初期不稳定的问题。在一开始使用0.01的lr直到训练loss下降至80%以下（约400步），转而使用0.1的大lr继续训练。然而作者尝试了这种方法，效果并不好（如下图(b)），所以在此基础上进行了改进，提出了渐变式的warm up。具体如下。
![](/assets/images/bn/9.png)
&emsp;&emsp;在batch size为kn的情况下，使用原本的$lr=\eta$作为起步，然后在5个epoch内线性递增到$lr=k\eta$。这样就能进行平稳的过渡，避免了之前的硬转折。在后续的训练中，也可以使用任意的lr下降策略。从上图(c)可以看出，使用了Gradual warmup策略后，训练曲线除去初期过渡阶段在后期几乎完全与baseline重合，说明大batch size的影响几乎被消除。

>> ## 与题目的几何背景进行对话

&emsp;&emsp;文中的BN并没有在多GPU间进行同步，而是在各自GPU的local batch size上进行计算，应用在各自的local batch数据上，随后计算出local loss。但是最终对local loss进行了同步，求和后得出了总loss。
![](/assets/images/bn/5.png)
![](/assets/images/bn/6.png)
&emsp;&emsp;作者认为如果每个GPU的local batch size保持相同，那么是在相同的数据方差上对local数据进行归一化，由此得到的local loss是独立同分布的，不影响训练过程。原话是`the mean/variance statistics computed by BN with different n exhibit different levels of random variation.`和` if the per-worker sample size n is kept fixed and the total minibatch size is kn, it can be viewed a minibatch of k samples with each sample Bj independently selected from Xn, so the underlying loss function is unchanged and is still defined in Xn`。

>> ## 与题目的物理背景进行对话

&emsp;&emsp;-- 假如loss中带有L2正则项时，该项中也含有学习率lr，不要忘了对该项进行缩放；  
&emsp;&emsp;-- 带动量矫正项的SGD需要对动量项也进行缩放；  
&emsp;&emsp;-- 多个GPU在进行loss同步时，`allreduce`是相加操作而不是相加后取平均，因此需要事先对local loss除以`kn`，而不是`n`。  
&emsp;&emsp;-- 每个epoch需要对数据进行random shuffle，然后分配给各个GPU。这里的shuffle需要统一进行或者采用相同的random seed，否则不是真正的random均匀分配。

>> ## 与题目的生活背景进行对话

&emsp;&emsp;-- 假如loss中带有L2正则项时，该项中也含有学习率lr，不要忘了对该项进行缩放；  
&emsp;&emsp;-- 带动量矫正项的SGD需要对动量项也进行缩放；  
&emsp;&emsp;-- 多个GPU在进行loss同步时，`allreduce`是相加操作而不是相加后取平均，因此需要事先对local loss除以`kn`，而不是`n`。  
&emsp;&emsp;-- 每个epoch需要对数据进行random shuffle，然后分配给各个GPU。这里的shuffle需要统一进行或者采用相同的random seed，否则不是真正的random均匀分配。


原文地址：[数学通报 2018.11 Journal of Mathematics (China)](https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2019&filename=SXTB201811013&v=NmDcNfwVnSX8iSycpgNmJJhyd%25mmd2Fxz%25mmd2FZIigBSC2rjE4ouvBEo%25mmd2BUSAZdZrTy0I%25mmd2FAA8t)


