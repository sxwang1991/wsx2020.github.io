---
title:  variants of BN 
categories:
- paper-reading
tags:
- regularization
- warm-up
- multi-GPU
- activation-func
---

&emsp;&emsp;BN作为神经网络的万金油，能够显著加速模型训练，并提升模型训练的稳定性和最终精度。它在一个batch的数据上做归一化，把数据重整化为零均值标准方差的分布。问题在于，如何取batch size的大小？batch size太大，做归一化绰绰有余，且训练时间能大幅缩短，但是GPU显存吃不消，只能在多卡上做数据并行训练，多卡数据间如何求BN也是个问题，同时由于一个epoch内steps少了，模型迭代次数不够，也会掉点；batch size太小，数据间方差波动太大，不具有统计性，此时做BN并不能对数据做有效归一化，也会影响最终精度。本文就这些问题做一些文献调研和记录，涉及Linear Scaling Rule、Gradual warmup、Cross-GPU BN、Cross-Iteration BN、Filter Response Normalization Layer等。
<!-- more -->

***
>+ # Large Minibatch SGD (2018)
&emsp;&emsp;Facebook出品，作者中大佬云集，何凯明大神也在其中。文中虽然不涉及BN的改动，但是探讨了在大batch size下学习率的缩放规律，在256张卡上取batch size为8192（8192=256x32），learning rate取很大的0.1x32=3.2，在1个小时内训练完了ImageNet上的ResNet-50。多卡数据分布式训练，同步loss，不同步BN，但其中提出的学习率缩放率Linear Scaling Rule为后续的SyncBN提供了基础，所以做一下记录。

>> ## 问题背景
![](/assets/images/bn/10.png)
&emsp;&emsp;随着数据集的不断扩充，模型的训练时间变得越来越长。为了节省训练时间，需要使用大的batch size（单卡上普通batch size 256，多卡合并成更大的batch size），从上图也可以看出batch size增大后每个epoch需要的时间能迅速下降。但在数据总量固定的情况下，一个epoch下step变少了，意味着可供模型优化调整的迭代次数少了，模型学习得不够充分，反而会造成精度的下降（如下图）。做一个形象的比喻，小步伐（小learning rate）下迈的步数（step）变少了，导致没能到达全局最优点。为了解决这个问题，需要使用大的learning rate，大步伐下迈少的步数，就能达到“小步伐下迈多的步数”同样的效果。
![](/assets/images/bn/7.png)
&emsp;&emsp;但是这个大的learning rate需要取多大？取大了会不会造成训练不稳定的情况？于是这篇paper便针对这两个问题提出了Linear Scaling Rule和Gradual warmup。

>> ## Linear Scaling Rule
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

>> ## Gradual warmup
&emsp;&emsp;早在resnet中，就采用了warm up来解决模型训练初期不稳定的问题。在一开始使用0.01的lr直到训练loss下降至80%以下（约400步），转而使用0.1的大lr继续训练。然而作者尝试了这种方法，效果并不好（如下图(b)），所以在此基础上进行了改进，提出了渐变式的warm up。具体如下。
![](/assets/images/bn/9.png)
&emsp;&emsp;在batch size为kn的情况下，使用原本的$lr=\eta$作为起步，然后在5个epoch内线性递增到$lr=k\eta$。这样就能进行平稳的过渡，避免了之前的硬转折。在后续的训练中，也可以使用任意的lr下降策略。从上图(c)可以看出，使用了Gradual warmup策略后，训练曲线除去初期过渡阶段在后期几乎完全与baseline重合，说明大batch size的影响几乎被消除。

>> ## 伪multi-GPU BN
&emsp;&emsp;文中的BN并没有在多GPU间进行同步，而是在各自GPU的local batch size上进行计算，应用在各自的local batch数据上，随后计算出local loss。但是最终对local loss进行了同步，求和后得出了总loss。
![](/assets/images/bn/5.png)
![](/assets/images/bn/6.png)
&emsp;&emsp;作者认为如果每个GPU的local batch size保持相同，那么是在相同的数据方差上对local数据进行归一化，由此得到的local loss是独立同分布的，不影响训练过程。原话是`the mean/variance statistics computed by BN with different n exhibit different levels of random variation.`和` if the per-worker sample size n is kept fixed and the total minibatch size is kn, it can be viewed a minibatch of k samples with each sample Bj independently selected from Xn, so the underlying loss function is unchanged and is still defined in Xn`。

>> ## 其它细节
&emsp;&emsp;-- 假如loss中带有L2正则项时，该项中也含有学习率lr，不要忘了对该项进行缩放；  
&emsp;&emsp;-- 带动量矫正项的SGD需要对动量项也进行缩放；  
&emsp;&emsp;-- 多个GPU在进行loss同步时，`allreduce`是相加操作而不是相加后取平均，因此需要事先对local loss除以`kn`，而不是`n`。  
&emsp;&emsp;-- 每个epoch需要对数据进行random shuffle，然后分配给各个GPU。这里的shuffle需要统一进行或者采用相同的random seed，否则不是真正的random均匀分配。


原文地址：[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/pdf/1706.02677.pdf%5B3%5D%20ImageNet)


***
>+ # Cross-GPU BN (2018)
&emsp;&emsp;Cross-GPU BN，又名SyncBN，这在工业界其实已经是默认不可或缺的trick了，但在paper上缺很少有具体的描述。很奇怪，SyncBN的cite都指向商汤的一篇[Context Encoding for Semantic Segmentation(CVPR 2018)](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Context_Encoding_for_CVPR_2018_paper.pdf)，但文中却又找不到具体内容，只顺带提了几句。反倒是在旷视的[MegDet: A Large Mini-Batch Object Detector(CVPR 2018)](http://openaccess.thecvf.com/content_cvpr_2018/html/Peng_MegDet_A_Large_CVPR_2018_paper.html)上看到了具体的原理分析和实验过程。

>> ## 问题背景  
&emsp;&emsp;上一篇facebook的工作在丧心病狂地追求1小时训完模型，多卡大batch size，即使在每张卡上也是较大的batch size。而这篇paper的出发点有所不同，单卡上是很小的batch size（通常是2-16），勉强在多卡上才凑足大batch size 256。这是因为在detection和segmentation等任务上，为了提升对小目标的检测能力，需要很大的图片分辨率，限于GPU内存导致batch size很小。  
&emsp;&emsp;前面也分析过小batch size带来的缺点，一是训练时间太长，二是做BN时统计不稳定，三是在detection任务中，由于一个batch内数据太少造成候选框的正负样本比例不稳定，或者说是不平衡，也会造成掉点。因此，这篇文章采用跨卡BN的方式变相增加有效的batch size，同时也对前一篇工作提出的Linear Scaling Rule进行了新的解释。 

>> ## Linear Scaling Rule
&emsp;&emsp;第一篇facebook的工作基于梯度近似不变的假设，但是在detection等任务中每张图片有不同的候选框，loss可能变化比较剧烈，假设不容易成立，因此本文的作者提出了一种更泛化的假设：梯度的方差在临近迭代步中近似不变。  
&emsp;&emsp;由于在一个batch内，每个样本是独立同分布（i.i.d.）的，其产生的梯度也是独立同分布的，那么有：
![](/assets/images/bn/11.png)
&emsp;&emsp;同理，对于大batch size $\hat{N}=kN$，有：
![](/assets/images/bn/12.png)
&emsp;&emsp;针对假设——大batch size下单步的梯度方差，近似于“小batch size下k步的梯度方差”，有：
![](/assets/images/bn/13.png)
&emsp;&emsp;想要上式成立，就只有使得lr满足线性缩放率：
![](/assets/images/bn/14.png)
&emsp;&emsp;这是不同于上一篇工作关于Linear Scaling Rule的另一种推导和解释。

>> ## SyncBN
&emsp;&emsp;思路比较直接，既然要在整个batch上做BN，那么就要得到整个batch数据的均值和方差。首先在每个GPU上求各自数据的统计值，然后allreduce到一起，求出整个batch的均值方差，再broadcast到各个GPU做BN即可。伪代码如下图。
![](/assets/images/bn/15.png)
&emsp;&emsp;数据流的传输过程如下图。求均值$\mu_B$很简单，只需要求各个GPU上数据的$s_i=\sum_{i=1}^n x_i$即可，allreduce到一起后，$\mu_B=\frac{\sum_{j=1}^k\;s_j}{kn}$。求方差$\sigma_B^2$要复杂一点，不仅需要各个GPU的$s_i=\sum_{i=1}^n x_i$，还需要$\upsilon_i=\sum_{i=1}^n x_i^2$。按照上图的算法流程需要同步两次，要知道多GPU间每同步一次，要耗费相当大的时间，先计算完的GPU要等待作业队列中后完成的GPU，无形中造成了计算资源的浪费。
![](/assets/images/bn/16.png)
&emsp;&emsp;得到大batch size上的$\mu_B$和$\sigma_B^2$后，就可以在各个GPU对数据进行归一化了：
![](/assets/images/bn/17.png)
&emsp;&emsp;实际上，商汤版本的SyncBN表示这里并不需要同步两次，只需要一次就够了，即一次性把$\sum x_i$和$\sum x_i^2$同步即可，信息流动图和计算公式如下：
![](/assets/images/bn/18.jpg)
![](/assets/images/bn/19.png)


原文地址：[MegDet: A Large Mini-Batch Object Detector](http://openaccess.thecvf.com/content_cvpr_2018/html/Peng_MegDet_A_Large_CVPR_2018_paper.html)  
商汤版BN的tutorial：[IMPLEMENTING SYNCHRONIZED MULTI-GPU BATCH NORMALIZATION](https://hangzhang.org/PyTorch-Encoding/tutorials/syncbn.html)

***
>+ # Cross-Iteration BN (2020)
&emsp;&emsp;与前两部分工作不同，这篇paper不涉及多卡分布式训练，旨在单张GPU上解决batch size过小的问题，借助邻近step上的多个batch的统计信息集成为更大的batch。

>> ## 问题背景 
&emsp;&emsp;出发点依然是过小的batch size会造成统计的不稳定性。前两部分工作借助更多的GPU更多的显存，把batch size变相地扩充，但显然这需要花钱买装备，不是每个人都能这么壕的。而这里，作者做出一个假设，模型训练时在临近的step之间，参数并没有剧烈的变化，是一种平滑的过渡。那么就可以在单GPU上对邻近的多个step的数据进行统计，得到更稳定的均值和方差，再对当前step下的数据进行BN。这样就不需要更多的GPU，也能变相扩充batch size，不过这种方法只能用来提升精度，不能用来提升训练速度（甚至会稍微减低训练速度）。

>> ## 跨step的泰勒展开
&emsp;&emsp;首先明确一下符号记法。在第$t$次step时，有模型参数$\theta_t$，此时当前batch内数据的均值为$\mu_t$、平方均值为$\upsilon_t$。
![](/assets/images/bn/21.png)
![](/assets/images/bn/22.png)
&emsp;&emsp;在$\tau$次迭代前，即第$t-\tau$次step时，有模型参数$\theta_{t-\tau}$。根据模型参数$\theta$在短暂的迭代范围$\tau$内变化平缓的假设，可用一阶的泰勒展开来近似“当前参数$\theta_t$下，$\tau$次迭代步前批次数据的$\mu_t$和$\upsilon_t$”，表达式如下：
![](/assets/images/bn/20.png)
&emsp;&emsp;然而上式中参数$\theta_t$代表整个网络参数，无法在某一层$l$上单独计算。为了方便起见，这里再做一次简化，在第$l$层上的统计量$\mu_t^l$和$\upsilon_t^l$，只与第$l$层上的参数$\theta_t^l$相关，前面$1\sim l$层参数对它们的影响看做高阶小量略去。下图也印证了这一假设，$l-1$层和$l-2$层的梯度要远小于$l$层本身。
![](/assets/images/bn/29.png)
&emsp;&emsp;由此可做极大简化如下：
![](/assets/images/bn/23.png)
&emsp;&emsp;由此，上式中得到的统计量即可看作是：***假如前几个step中用掉的数据，在当前的模型参数下，所具有的均值和平方均值***。前k步的数据可以名正言顺地都用在当前step的参数下，来求解BN所需要的均值$\bar{\mu}_t^l$和方差$\bar{\sigma}_t^l$，计算公式如下。注意为了保险起见，求平方均值时用了max操作，因为做泰勒近似后有可能出现均值的平方大于平方均值的情况发生。
![](/assets/images/bn/24.png)
&emsp;&emsp;最后，当前step下的BN操作便可使用这种在更大batch size下统计得到的均值和方差，来进行更为准确的缩放。
![](/assets/images/bn/25.png)
&emsp;&emsp;Cross-Iteration Batch Normalization (CBN)和普通BN的对比图如下：
![](/assets/images/bn/26.png)

>> ## 性能比较
&emsp;&emsp;在backward时，与BN的计算量没有区别；只是在forward时，为了做一阶泰勒近似，需要求解统计量关于参数的一阶导数，这会产生一些额外的计算量。作者对比了它们的运行时间，可以看到training确实稍微慢了点，但是inference速度不受影响。
![](/assets/images/bn/27.png)
&emsp;&emsp;在滑窗大小k的选择上，如果k太小，无法有效增加batch size；如果k太大，step迭代次数过多，模型的参数变化过大，不满足泰勒一阶近似的条件，也会使统计量不准确。因此做了实验对比，发现$k=8$时有较好的性能，但是作者又说在很多任务上batch size大于16性能就会饱和，因此这里针对k做了截断，$k=min(\frac{16}{batch-size}, 8)$。另外，在训练初期模型参数变化剧烈，k也要取较小值。
![](/assets/images/bn/28.png)
&emsp;&emsp;CBN最终的性能如下图，可以看到尤其是在原batch非常小时（bs=1~4），CBN的性能非常出色。
![](/assets/images/bn/30.png)


&emsp;&emsp;原文地址：[Cross-Iteration Batch Normalization](https://arxiv.org/pdf/2002.05712.pdf)


***
>+ # Filter Response Normalization（FRN, 2020）
&emsp;&emsp;同样是面对batch size过小的场景，前面讲了借助多GPU的方法变相扩充batch size，但是壕无人性；还有借助多step的方法，效果挺好，不过会稍微增加训练时间。除此之外，就只能在原本BN算法的基础上下功夫琢磨改进了。

>> ## 问题背景
&emsp;&emsp;CNN中每一层的输入tensor维度为[N,C,H,W]，BN求统计量时沿[N,H,W]轴计算，每个channel间相互独立；LN（Layer Norm）求统计量时沿[C,H,W]轴计算，每个sample间相互独立；IN（Instance Norm）沿[H,W]轴计算，每个sample的每个channel都相互独立；GN（Group Norm）沿[C/G,H,W]轴计算，每个sample的每个group间相互独立。它们的重整化方法都与BN相同，只是作用的维度不同。其中，LN、IN、GN都不在整个batch上做统计，因此不受batch size的大小影响，也就不会有小batch size时性能下降的缺点，同时也可以设计更大容量的模型而无视小batch size的影响。由于各自的特点，IN在style transfer任务中表现较好，但在recognition上表现不佳；GN的表现最佳，在小batch size时性能依旧坚挺，但在大batch size（16）时效果会比BN稍差，同时由于需要满足channel数是group的倍数，在模型超参数上会有一些限制。上述的各种norm方法示意图和效果如下。
![](/assets/images/bn/31.png)
![](/assets/images/bn/32.png)

>> ## FRN & LTU
&emsp;&emsp;在这样的背景下，好像已经没有点能再把BN玩出花样了。玩儿不过它，就抛弃它。Google的这篇paper提出新的norm方法，Filter Response Normalization（FRN），在[H,W]维度上做重整化，一整个batch上每个smaple的每个channel都相互独立，维度跟IN一样，不过已经抛弃了BN求均值和方差的思路，直接简单粗暴做如下归一化（tensorflow中c在第四维度）：
![](/assets/images/bn/33.png)
![](/assets/images/bn/34.png)
![](/assets/images/bn/35.png)
![](/assets/images/bn/36.png)
&emsp;&emsp;之所以说它算是归一化，而不是重整化，是因为它并没有对输入求平均然后减掉，因此归一化后的平均值并不像BN那样是0。没有了均值，原本分母上的方差自然就退化成了平方和，相当于跟BN比起来，FRN只是没有减去均值，同时统计维度不同而已。文中也没有说具体这样做的动机和思路，好像是拍脑袋想出来的，反正只要最终效果好，怎么解释都能发paper。  
&emsp;&emsp;然而FRN这样处理一通还不行，不能直接拿来用，因为它的输出均值不为零，如果不慎均值小于0的话会在下一层的ReLU上直接落入死区。为了避免这种问题，文中又提出了一种新的激活函数，Thresholded Linear Unit (TLU)：
![](/assets/images/bn/37.png)
&emsp;&emsp;其中，新参数$\tau$是一个可学习的变量，使得原本ReLU的死区0变为$\tau$，让模型自主选择FRN的输入是dead or alive，均值不管多大都可以自适应了。通过FRN和TLU的搭配，替代经典的BN和ReLU的组合，这便是这篇文章的主要工作。
![](/assets/images/bn/38.png)
&emsp;&emsp;在具体的应用上还有一些细节。对输入做归一化时，分母中含非零小量$\epsilon$，如果碰巧做FRN的维度[H,W]为[1x1]（例如Inception的最后一层），那么此时小量$\epsilon$对整个分式的影响就不可忽略了。因此，这种情况下$\epsilon$被设置为可学习的参数，初始化为$10^{-4}$。其它正常层的$\epsilon$被设置为常量$10^{-6}$。  
&emsp;&emsp;另外，发现学习率lr阶梯衰减在FRN上并不好用，需要使用cosine decay。由于FRN的输出均值不为0，在训练初始阶段时不稳定，还需要设置warm up机制。

>> ## 模型效果及思考
![](/assets/images/bn/39.png)
&emsp;&emsp;在ImageNet的ResNetV2-50上，FRN超越了BN、BRN和GN。即使是在大batch size时，效果也比BN好，这是GN所达不到的。代码也很简洁（如下图）。
![](/assets/images/bn/40.png)
&emsp;&emsp;在消融实验中，把GN/LN所采用的减去均值除以标准差的计算公式，换为FRN中的归一化方法，命名为BFRN/GFRN/LFRN。在各项的搭配组合中，确实是FRN+TLU效果最好。但是也能看到，GFRN/LFRN居然比GN/LN有较普遍的性能提升，从ReLU换到TLU也有较普遍的性能提升，*FRN和TLU在一起也能有普遍的提升。
![](/assets/images/bn/41.png)
&emsp;&emsp;仔细想想，FRN做归一化时为了不受batch size影响，独立了维度N。把每一个channel看做一个filter，为了不影响filter间的独立性，又独立了维度C。但是不减去均值是个啥操作？文中就解释了一句，` While mean subtraction was an important aspect of Batch Normalization, it is arbitrary and without real justification for normalization schemes that are batch independent`，反倒把BN减均值的操作给批判了一番。最后均值不为0这个问题，留给了LTU来擦屁股，相当于多了一些模型参数和计算量。


原文地址：[Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](http://openaccess.thecvf.com/content_CVPR_2020/papers/Singh_Filter_Response_Normalization_Layer_Eliminating_Batch_Dependence_in_the_Training_CVPR_2020_paper.pdf)
