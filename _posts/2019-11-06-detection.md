---
title:  Paper Reading - ICCV 2019 "ThunderNet - Towards Real-time Generic Object Detection"
categories:
- paper-reading
tags:
- detection
---

&emsp;&emsp;旷视detection组的一篇轻量级two-stage目标检测论文, 起的名字很好听, ThunderNet, 所以就特意找出来看一看. 以前接触detection比较少, 就趁这个机会把一些经典的object detection论文找出来读一读, 主要有two-stage的Faster-RCNN和ont-stage的YOLO、SDD, 它们奠定了一些基本的思路和框架, 新发表的论文基本是在此基础上做延伸, 有需要的时候再细看, 下面是一些总结.

***
>+ # 问题概述

&emsp;&emsp;通用目标检测, 任务是检测出图片中存在物体的边界框坐标, 且需给出对应物体的分类. 直观地想象, 针对这样一个任务, 如果采用end-to-end的形式训练, 需要有一个网络, 它的输入是图片, 输出是可能存在物体的位置以及类别, 这是one-stage模型, 有无数个可能的位置以及很多种类别的可能, 难度是比较大的. 如果采用two-stage模型, 先让一个网络(region proposal network/RPN)负责找出可能存在物体的区域, 它只关心该区域是否存在物体, 以及该区域的大致坐标位置, 而不关系物体的分类具体是什么. 接下来把这些候选区域送进下一个分类网络(detector), 这个网络只需要关心候选框内的物体分类是什么即可, 具有较强烈的"候选区域内一定存在物体"的先验条件. 两部分网络各司其职, 就能有较好的效果, 但是也存在流程繁琐的缺点.
<center>具有代表性的经典paper</center>

![](/assets/images/detection/2.png)
<center>沿着时间线上的发展</center>

![](/assets/images/detection/1.png)
&emsp;&emsp;下面分别针对two-stage的经典模型: Faster-RCNN和one-stage的代表模型: YOLO/SDD, 记录一些基本的常识性内容.

***
>+ # Faster-RCNN (NIPS 2015)

&emsp;&emsp;MSRA出品, 作者列表中有孙剑、何凯明、任少卿等大佬, 这三位也是ResNet的部分作者. 其中任少卿是科大信院与MARS连培的博士, 我作为菜鸡真的是感叹人与人的差距比人与狗的差距还要大.
>> ## 网络结构  

![](/assets/images/detection/3.png)
&emsp;&emsp;主要有四部分: Backbone, RPN, ROI Pooling, Classifier.    
&emsp;&emsp;**Backbone:** 把固定大小为1000x600的input, 降采样为原来的1/16(用了4个2x2的pooling), 输出维度为512x60x40.  
&emsp;&emsp;**RPN:** 对每一个pixel提出9个候选框anchor, 这9种anchor如下图所示(长宽比0.5、1、2, 缩放比8、16、32), 一共有60x40x9=21600个anchor. 
![](/assets/images/detection/4.png)
&emsp;&emsp;&emsp;&emsp;有两个分支, 一路做regression, 输出anchor的四个坐标(中心点x、y, 宽高w、h), 维度是36x60x40, 36即是9种anchor的4个坐标点(其实不是绝对坐标, 而是坐标的偏差, 类似于残差的思想, 在后面loss一节会再做详细介绍); 另一路做classification, 输出该anchor是否存在物体(0或1), 维度是18x60x40, 18即是9种anchor被softmax分成0或1的两类.  
&emsp;&emsp;&emsp;&emsp;这些anchor数量太多了, 需要进行剔除. 首先去掉超出feature边界的框, 然后标记与Ground Truth之间IoU超过0.7的框即为正例保留, 标记与Ground Truth之间IoU小于0.3的框为反例保留, 其余框剔除. 然后在剩下的框中, 正例随机取128个, 反例随即取128个进行训练. (不同的是, 在inference时, 如果有anchor越界了, 不会进行剔除, 会对其进行clip, 限定在图像区域内.)  
&emsp;&emsp;&emsp;&emsp;RPN最后还有一个Proposal层, 细节也比较多. proposal接受经过残差修正后的新anchor(如何修正见下节loss介绍), 再次进行剔除(越界剔除、宽高过小的anchor剔除), 接下来进行非极大值抑制, 旨在去掉重复区域较多的anchors提高效率. 然后按照前面的输出softmax score从大到小对anchors进行排序, 选择前面topN的anchors(即只保留正例anchors). 至此, anchor就是比较干净和较准确的框了, 可以送入下一步进行分类识别.  
>>> -- 非极大值抑制(Non-Maximun Suppression/NMS): 
![](/assets/images/detection/5.png)
a. 对所有anchor的得分从大到小排序, 如上图[0.98, 0.83, 0.81, 0.75, 0.67].  
b. 以当前最大值0.98为参考, 依次判断其它anchor是否与参考框的IoU超过某一阈值(文中取的是0.7), 剔除重复面积过大的框, 如删除[0.83, 0.75], 保留最大值0.98并以后不参与筛选.  
c. 找出当前最大值0.81, 重复步骤b, 直到没有框可供筛选.
![](/assets/images/detection/6.png)
&emsp;&emsp;如图最后只留下了每一轮的最大值0.98和0.81.  

&emsp;&emsp;**ROI Pooling:** 输入是由RPN筛选出的大约300个anchors坐标(300x4), 以及backbone生成的维度为512x60x40的feature. 前面过程中幸存下来的这些anchors, 宽高比可能不一致, 需要把它们映射成固定大小的feature, 方便后面带有全连接的分类器进行分类. 方法比较粗暴, 即把anchor在feature上所对应的区域, 分割成7x7的块儿, 每个块内取最大值, 即得到300x512x7x7的feature.
![](/assets/images/detection/7.png)
>>> -- RoI Pooling可能存在的问题:  
&emsp;&emsp; 一, anchor映射回feature会存在量化误差, 如50/3=16.66, 实际中只能取16; 二, 映射后的feature的候选区域宽度不是7的整数倍时, pooling也会带来量化误差, 对结果产生影响, 如第一步中的16, 16/7=2.28, 实际中只能取2.   
&emsp;&emsp;因此这里有替代方案--RoIAlign: 一, 保留映射时的浮点数; 二, pooling时, 对虚拟中心点采用双线性插值, 再进行pooling.
![](/assets/images/detection/8.png)

&emsp;&emsp;**Classifier:** 输入是300x512x7x7的anchor feature, 输出有两路: 一路是每一个anchor的类别概率, 维度是300x81(有81个类); 另一路是每个anchor的坐标偏移修正法量, 维度是300x(81x4), 对框的位置进行精细修正. 这两个输出即是我们最终想要的"物体类别"和"物体位置".

>> ## 损失函数

&emsp;&emsp;RPN和Classifier是分开训练的, 因为目的不一样. RPN目的是找出存在物体的框, 只关心是否存在物体以及框的粗修正, 不关心框内是什么物体; Classifier只认为送给它的输入框内一定存在物体, 有一个很强的先验假设, 然后它在此基础上放心地去预测这个物理到底属于哪一类, 同时进行精细框修正.  
&emsp;&emsp;**RPN loss:** 一个是对物体是否存在的分类loss, 原文中采用的是log loss, 其实就是cross entropy loss; 一个是对边界框偏移量进行修正的regression loss, 原文中采用的是smooth L1 loss, 俗称huber loss. 由于两部分loss的数量级差了将近10倍, 所以取$\lambda=10$进行量级平衡.
![](/assets/images/detection/9.png)
&emsp;&emsp;&emsp;&emsp;再来看, anchor的四个坐标表示到底是什么, 摘抄原文中的一段如下, 即用$(t_x, t_y, t_w, t_h)$表示偏移量, 它们越小表示box重叠地越好. 回归收敛后, 可使用$(t_x, t_y, t_w, t_h)$转换得到真正的$(x_a, y_a, w_a, h_a)$.
![](/assets/images/detection/10.png)
&emsp;&emsp;**Classfier loss:** 基本同上, 一个分类loss, 对应着不同的类别(不同的是, RPN分类只有0和1两类, 而这里有81类); 一个是回归loss, 对应着精细框位置修正.  
&emsp;&emsp;**训练过程:** 一, 导入ImageNet预训练模型, 单独训练RPN; 二, 单独训练Classifier, 它的输入是上一步训练好的RPN提供的proposal; 三, 再次单独训练RPN, 这次会先把Classifier的部分Conv层权重导入到RPN并冻结, 这里就是共享权重的地方, 节省了计算时间; 四, 单独训练Classifier, 这里在上一步中共享给了RPN的权重会被冻结, 输入是上一步训练好的RPN提供的proposal. 本质就是一种迭代循环, Classifier不断地共享它的部分权重给RPN, 二者轮流fine tune, 只不过文中只循环了两次, 作者说两次循环就够了, 再往下循环下去对性能也没有太大提升.  
<br
/>
&emsp;&emsp;至此two-stage的Faster RCNN便介绍完了, 可以看到流程比较复杂, 而且NIPS原文有一些细节也没讲清楚, 这里参考了一些资料.  
&emsp;&emsp;Faster RCNN原文: [Faster RCNN](http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  
&emsp;&emsp;知乎专栏: [一文读懂Faster RCNN](https://zhuanlan.zhihu.com/p/31426458)  
&emsp;&emsp;一篇博客: [Faster RCNN 学习笔记](https://www.cnblogs.com/wangyong/p/8513563.html)

***
>+ ## YOLO

>+ ## SDD

>+ ## ThunderNET


<br
/>


[link1]: http://papers.nips.cc/paper/5633-texture-synthesis-using-convolutional-neural-networks
[link2]: https://arxiv.xilesou.top/abs/1508.06576
