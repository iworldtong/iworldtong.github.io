---
layout: post
title: "【论文整理】Temporal Sentence Localization in Video方向近期总结"
subtitle: ''
author: "iworld"
header-img: img/2019-09-09-【论文整理】Temporal Sentence Localization in Video方向近期总结.jpg
mathjax: true
tags:
  - Deep Learning	
  - Video Understanding
---

### 0.Introduce

Temporal Sentence Localization in Video，该任务内容为给定一个query（包含对activity的描述），在一段视频中找到描述动作（事件）发生的起止时间，如下图所示：

<div align="center"><img height="200"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1554267644/Awesome%20Language%20Moment%20Retrieval/TALL_-_2.png"></div>
该任务在17年被提出，经过两年发展，目前主流工作大致分为以下三种思路：

1. [Proposal-based](#1)
2. [Proposal-free](#2)
3. [Reinforcement-learning-based](#3)



### <span id="1">1.P</span>roposal-based

为解决这种根据描述定位的任务，一个比较直观想法是多模态匹配。即用滑动窗对整个视频提取clips，选候选clip中相似度最高的作为预测结果，同时为了使定位更加精确，除分类损失还需要有回归损失，对clip进行微调。这种方式比较有代表性的是CTRL、MCN、ACL-K等。

由于滑动窗法计算量比较大，之后有人提出了QSPN，其思想是在生成proposal的时候结合query信息，可以大量减少无关proposal的生成。该模型结合R-C3D，使其在生成proposal的时候引入query信息进行指导，这实际上是Faster-RCNN思想的迁移。

之后图卷积开始被瞩目，大量应用于各个领域。考虑到描述中存在“the second time”、“after”等具有高层时序关系的关键词，仅仅局限于某一段就会丢失这种时序的结构信息，因此需要考虑clip之间的关系，而这种关系的建模非常适用于图卷积的框架。受此启发，有研究者提出了MAN，通过迭代、残差的图卷积结构挖掘clip之间的关系。

除video特征外，sentence中同样包含着很多对定位至关重要的信息，如何有效挖掘这种linguistic特征也是一个亟需解决的问题。TMN从语法树的角度动态推理视频中的关系；CMIN利用依存句法构建图，利用图卷积提取描述特征；TCMN直接将sentence解析成树，提出了一种Tree Attention Network。

#### 1.1 CTRL

主要思想是将编码后的视觉、文本特征映射到一个子空间中，语义匹配的pairs在此空间中相似度会比较高。注意作者在编码视觉特征（central clip）时，还计算了相邻的context特征作为补充。

<div align="center"><img src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291245/iblog/LMR%20summary/ctrl-1.png"></div>
#### 1.2 ACL-K

该模型是对CTRL模型的改进，提出了一个action concept的概念，并将此特征与描述中的verb-obj pairs的semantic concepts进行匹配，以此提高性能。

<div align="center"><img src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291245/iblog/LMR%20summary/acl-k-1.png"></div>
#### 1.3 QSPN

将query信息作为注意力，嵌入到类似R-C3D的proposal生成网络中去。

<div align="center"><img height="200px"  src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291245/iblog/LMR%20summary/qspn-1.png"></div>
除此之外作者还设计了多任务损失，添加了重新生成query的损失。

<div align="center"><img height="200px" src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291245/iblog/LMR%20summary/qspn-2.png"></div>
#### 1.4 MAN

为解决一下两类misalignment问题，提出了利用graph convolution挖掘clip之间的关系。

<div align="center"><img height="400px" src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291516/iblog/LMR%20summary/man-1.png"></div>
网络框架：

<div align="center"><img src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291518/iblog/LMR%20summary/man-2.png"></div>
作者可视化了根据图卷积计算出的clip之间关系的例子：

<div align="center"><img height="350px" src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291518/iblog/LMR%20summary/man-3.png"></div>
#### 1.5 CMIN

通过如下依存句法关系对sentence建图，使用graph convolution提取文本特征。

<div align="center"><img src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291525/iblog/LMR%20summary/cmin-1.png" ></div>
同时使用multi-head self attention提取视频特征，模型框架如下：

<div align="center"><img src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568291525/iblog/LMR%20summary/cmin-2.png" ></div>

### <span id="2">2.Proposal-free</span>

上面介绍的方法都是**“scan and localize”**这种结构的，该框架在早期虽然取得了一定效果，但目前看来仍存在很多局限性，下面列举几条比较明显的缺点：

1. 用滑动窗等方法产生video clip之间是独立的，打断了隐含的时序结构和视频整体的全局信息，**不能有效利用整个视频序列**；
2. 一些方法将整个句子编码成一个全局特征（coarse feature），其中**某些对定位至关重要的句子信息容易被忽略或未被充分挖掘**；
3. 为提高性能，滑动窗法往往需要对视频密集采样，造成**极大的计算冗余**；
4. 预定义的clips，使得**预测结果变长扩展变得困难**；
5. 。。。。

#### 2.1 ABLR

[《To Find Where You Talk: Temporal Sentence Localization in Video with Attention Based Location Regression》](https://arxiv.org/abs/1804.07014)，发表于AAAI19。

本文主要贡献如下：

* 直接根据句子对整个视频预测起止时间；
* 提出了一种multi-modal co-attention机制

模型框架如下：

<div align="center"><img   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568021479/iblog/LMR%20summary/ABLR-1.png"></div>
为了使不同视频片段之间能够建立关联，将视频均分为$N_{split}$个clip分别得到C3D特征后继续送给Bi-LSTM。之后进行注意力计算，其计算公式如下：


$$
\begin{aligned} \mathbf{H} &=\tanh \left(\mathbf{U}_{z} \mathbf{Z}+\left(\mathbf{U}_{g} \mathbf{g}\right) \mathbf{1}^{T}+\mathbf{b}_{a} \mathbf{1}^{T}\right) \\ \mathbf{a}^{z} &=\operatorname{softmax}\left(\mathbf{u}_{a}^{T} \mathbf{H}\right) \\ \tilde{\mathbf{z}} &=\sum a_{j}^{z} \mathbf{z}_{j} \end{aligned}
$$


Multi-Modal Co-Attention Interaction过程共有三步：

1. **“Where to look”**，对视频进行注意力加权，$\mathbf{Z}=\mathbf{V},\mathbf{g}=mean\_pool(\mathbf{S})$
2. **"Which words to guide"**，根据加权后的视频，对句子注意力加权，$\mathbf{Z}=\mathbf{S},\mathbf{g}=\widetilde{v}$
3. 根据加权后的句子特征，再次对视频进行注意力加权，$\mathbf{Z}=\mathbf{V}_a,\mathbf{g}=\widetilde{s}$

最终进行坐标预测时作者设计了两种计算方式：

1. **Attention weight based regression**：直接用注意力权重$\mathbf{a}^V$输入FC输出坐标
2. **Attention feature based regression**：将加权拼接后的$[\widetilde{v}\|\|\widetilde{s}]$送到一双层FC输出坐标

为了增强事件内部的clip注意力权重，作者还设计了一种attention calibration loss（见下式），与reg loss一起构成总的损失函数。


$$
L_{c a l}=-\sum_{i=1}^{K} \frac{\sum_{j=1}^{M} m_{i, j} \log \left(a_{j}^{V_{i}}\right)}{\sum_{j=1}^{M} m_{i, j}}
$$


在模型完整（full）的情况下，前者效果更好：

<div align="center"><img height="300"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568021479/iblog/LMR%20summary/ABLR-2.png"></div>
其实计算出的注意力直接关系到哪些clip是包含在sentence语义内的，因此直接输入attention取得了最高的效果。

同时实验也说明了这种Proposal-free的方法更加高效：

<div align="center"><img height="150"  src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568021479/iblog/LMR%20summary/ABLR-3.png"></div>
**Note**：本文提出的注意力机制比较繁琐，有优化的空间。

#### 2.2 ExCL

[《ExCL: Extractive Clip Localization Using Natural Language Descriptions》](https://arxiv.org/abs/1904.02755)，发表于NAACL19。

本文通过text和video两模态之间的交互来直接预测起止时间，text特征使用text Bi-LSTM最后的隐含层状态$\mathbf{h}^T$，具体有如下三种多模态交互方式：

<div align="center"><img height="350"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568084144/iblog/LMR%20summary/ExCL-1.png"></div>
* **MLP predictor**：在每一时间步，使用拼接的$[\mathbf{h}\_t^V\|\mathbf{h}^T]$，输入MLP分别得到$S_{start}(t),S_{end}(t)$

* **Tied LSTM predictor**：上述拼接的特征首先经过一个Bi-LSTM，得到$\mathbf{h}_t^P$，将$[\mathbf{h}_t^P\|\mathbf{h}_t^V\|\mathbf{h}^T]$送给MLP

* **Conditioned-LSTM predictor**：由于前面两方法没有考虑$S_{start}(t)$与$S_{end}(t)$的时间次序关系，因此在计算结束时间时，引入起始时间的信息，以期使输出更加合理，公式如下：
  
  $$
  \begin{aligned} \mathbf{h}_{t}^{P_{0}} &=\mathrm{LSTM}_{\text {start }}\left(\left[\mathbf{h}_{t}^{V} ; \mathbf{h}^{T}\right], \mathbf{h}_{t-1}^{P_{0}}\right) \\ \mathbf{h}_{t}^{P_{1}} &=\mathrm{LSTM}_{\text {end }}\left(\mathbf{h}_{t}^{P_{0}}, \mathbf{h}_{t-1}^{P_{1}}\right) \\ S_{\text {start }}(t) &=\mathbf{W}_{s}\left(\left[\mathbf{h}_{t}^{P_{0}} ; \mathbf{h}_{t}^{V} ; \mathbf{h}^{T}\right]\right)+\mathbf{b}_{s} \\ S_{\text {end }}(t) &=\mathbf{W}_{e}\left(\left[\mathbf{h}_{t}^{P_{1}} ; \mathbf{h}_{t}^{V} ; \mathbf{h}^{T}\right]\right)+\mathbf{b}_{e} \end{aligned}
  $$



实验结果如下：

<div align="center"><img height="400"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568084144/iblog/LMR%20summary/ExCL-2.png"></div>
#### 2.3 Proposal-free with Guided Attention

[《Proposal-free Temporal Moment Localization of a Natural-Language Query in Video using Guided Attention》](https://arxiv.org/abs/1908.07236)，目前在Arxiv。

本文主要贡献有三点：

* A dynamic filter融合多模态特征，目的是根据给定的句子，动态的对视频特征进行过滤；
* 一种新的损失函数，指导模型关注视频中最相关的部分；
* 使用soft labels建模标签的模糊性。

模型整体结构如下（编码句子特征时使用mean pool）：

<div align="center"><img height="300"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568086287/iblog/LMR%20summary/others-1.png"></div>
Guided Attention计算公式如下（$\mathbf{G}$为视频特征）：


$$
\begin{array}{l}
{A=\operatorname{softmax}\left(\frac{G^{\top} \theta(\overline{h})}{\sqrt{n}}\right) \in \mathbb{R}^{n}} \\ 
{\overline{G}=A \odot G \in \mathbb{R}^{n \times d}} \\

{其中\ \theta(x)=\tanh \left(W_{\theta} x+b_{\theta}\right) \in \mathbb{R}^{d}}
\end{array}
$$


作者假设segments外部的视频更不可能对训练有帮助，因此设计att loss使外部的注意力更小（类似的思想见ABLR）：


$$
L_{a t t}=-\sum_{i=1}^{n}\left(1-\delta_{\tau^{s} \leq i \leq \tau^{e}}\right) \log \left(1-a_{i}\right)
$$


考虑到即使对人来说，识别一段事件发生的起止时间也是非常模糊的，有很大的主观成分，因此作者使用了soft labels，针对概率分布进行优化。

设$\hat{\boldsymbol{\tau}}^{s}, \hat{\boldsymbol{\tau}}^{e} \in \mathbb{R}^{n}$是得到的预测分布，${\boldsymbol{\tau}}^{s}, {\boldsymbol{\tau}}^{e}$是真实标签分布，则有$\boldsymbol{\tau}^{s} \sim \mathcal{N}\left(\tau^{s}, 1\right) \in \mathbb{R}^{n}，\boldsymbol{\tau}^{s} \sim \mathcal{N}\left(\tau^{e}, 1\right) \in \mathbb{R}^{n}$，使用**KL散度**进行优化：


$$
L_{K L}=D_{\mathrm{KL}}\left(\hat{\tau}^{s} \| \tau^{s}\right)+D_{\mathrm{KL}}\left(\hat{\tau}^{e} \| \tau^{e}\right)
$$


最终损失函数：


$$
L o s s=L_{K L}+L_{a t t}
$$


实验结果：

<div align="center"><img height="400"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568086287/iblog/LMR%20summary/others-2.png"></div>
实验说明了soft labels确实有效：

<div align="center"><img height="150"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568088783/iblog/LMR%20summary/others-3.png"></div>
### <span id="3">3.Reinforcement-learning-based</span>

强化学习的思路同样也是为了解决proposal-based方法对长视频提取密集、冗余的clip特征。

<div align="center"><img  height="250" src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568188044/iblog/LMR%20summary/LMR-RL-1.png"></div>
#### 3.1 TripNet

[《Tripping through time: Efficient Localization of Activities in Videos》](https://arxiv.org/abs/1904.09936)，Arxiv。

>We are motivated by human annotators who observe a short clip and make a decision to skip forward or backward in the video by some number of frames, until they can narrow in on the target clip.

人在做此项任务时，往往首先从视频开始粗略的看，在此过程中采样到一些列稀疏的且与描述相关的帧，之后再进行frame-by-frame的定位。受此启发，本文提出了一个端到端、gated-attention的强化学习框架。主要思想是先初始化一个clip，根据当前状态计算当前策略，调整窗口直至靠近目标片段。

模型分为两个主要部分：

* **the state processing module**：计算当前状态的visual-linguistic编码特征
* **the policy learning module**：生成action policy

<div align="center"><img   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568106487/iblog/LMR%20summary/tripnet-1.png"></div>
计算状态$s_t$使用了注意力的形式（见state processing module），注意作者在提取visual feature时使用的是C3D+mean_pool。

使用全连接网络计算value与policy。为了使模型尽快定位到目标片段，value中的reward除了计算当前窗口与标签的IOU，还添加了一个**随搜索步数$t$增长的负奖励**。对于policy，作者定义了7种动作（$W_{start},W_{end}$分别向前/向后移动$N/5,N/10$，一秒范围的帧或终止），policy输出的就是当前动作在该动作空间中的概率分布。

实验结果：

<div align="center"><img   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568106487/iblog/LMR%20summary/tripnet-24.png"></div>
一些搜索过程统计与可视化：

<div align="center"><img height="600"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568106487/iblog/LMR%20summary/tripnet-5.png"></div>
效率验证：

<div align="center"><img height="120"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568106487/iblog/LMR%20summary/tripnet-6.png"></div>
**Note**：计算当前状态时只考虑了当前窗口内的视频特征，没有考虑全局特征。

#### 3.2 SM-RL

[《Language-driven Temporal Activity Localization: A Semantic Matching Reinforcement Learning Model》](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Language-Driven_Temporal_Activity_Localization_A_Semantic_Matching_Reinforcement_Learning_Model_CVPR_2019_paper.pdf)，发表于CVPR19。

video与sentence之间有着很大的语义鸿沟，但考虑到sentence中往往有一些semantic concept（如actor，action和object等），为了通过语义层面使video与sentence之间更加容易匹配，作者提出了一种基于强化学习的semantic matching方法。

<div align="center"><img   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568171060/iblog/LMR%20summary/SM-RL-1.png"></div>
在计算semantic concept时作者对比了**multi-label classification based model**（通过描述建立标签，如nouns，verbs等）与**faster r-cnn based model**（在visual gnome上预训练）两种生成semantic的方法，后者效果更优。

损失函数对比了**boundary regression loss**与**frame-level regression loss**，以下为实验结果。

<div align="center"><img   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568171060/iblog/LMR%20summary/SM-RL-2.png"></div>
**Note**：有点像视频描述中[topic-embed](https://arxiv.org/abs/1811.02765v1)模型，semantic标签预测的准确性可能对模型性能影响很大。

#### 3.3 Read, Watch, and Move

[《Read, Watch, and Move: Reinforcement Learning for Temporally Grounding Natural Language Descriptions in Videos》](https://arxiv.org/abs/1901.06829)，发表于AAAI19。

本文模型与TripNet较为相似，模型框架如下：

<div align="center"><img   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568188046/iblog/LMR%20summary/watch-1.png"></div>
计算当前状态时，除了局部特征，还考虑了整个视频的全局特征。此外，作者实验发现加入location feature可以有效提高指标。（见observation network）

对强化学习，在第$t$步，如果当前的tIoU增大，则给一个正奖励；减小则无；当tIoU为负值时（即预测起始时间在结束时间之后），给负奖励。一般来说，策略网络搜索的步数更多，结果可能会更加精确，但随之而来的计算代价也会增大。为缓解这个问题，作者又设置了随执行步数增大而增大的惩罚。则第$t$步的reward计算公式如下：


$$
r_{t}=\left\{\begin{array}{ll}{1-\phi * t,} & {t I o U^{(t)}>t I o U^{(t-1)} \geq 0} \\ {-\phi * t,} & {0 \leq t I o U^{(t)} \leq t \operatorname{IoU}^{(t-1)}} \\ {-1-\phi * t,} & {\text { otherwise }}\end{array}\right.
$$


值得注意的是，为了使模型学习到更好的state vector，作者将**supervised learning**（L1 tIoU regression loss + L1 location regression loss） 与**reinforcement learning**（actor-critic algorithm）相结合，组成一个**multi-task learning**框架，提高模型性能。

实验结果：

<div align="center"><img height="150"  src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568188046/iblog/LMR%20summary/watch-2.png"></div>
对比实验：

<div align="center"><img width="350"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568188046/iblog/LMR%20summary/watch-3.png">
<img width="350"  src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568188046/iblog/LMR%20summary/watch-4.png"></div>

最后作者分析了location feature、监督学习中两个不同损失对指标的影响：

<div align="center"><img height="200"   src="https://res.cloudinary.com/dzu6x6nqi/image/upload/v1568188046/iblog/LMR%20summary/watch-5.png"></div>
**Note**：observation network融合不同特征的方法还有待挖掘。