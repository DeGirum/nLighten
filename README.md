# Cloud Quality AI Models for the Edge
AI at the edge is still limited to running toy models at low resolution. Recently, there has been significant effort to design specialized HW for the edge. The aim of **nLighten** is to develop state-of-art AI models capable of running efficiently on such devices. We will develop highly optimized models for a variety of applications. 

The main aims of **nLighten** are:
1. Encourage ML researchers and developers to work on edge deployment related problems. 
2. Develop pruned versions of the latest models such as EfficientNetLite and detection models based on EfficientNetLite backbones.
3. Use the newly developed backbones for other tasks such as segmentation, transfer learning and key-point detection.
4. Develop tools necessary to go from model development to model deployment.

## Call for Collaboration
While neural network optimization techniques such as pruning have been known for a while, there is a lack of deployment ready models that take advantage of such techniques. One reason is that there is no mainstream HW that can realize all the gains afforded by the pruned networks. Other reason is that the whole process of training-pruning-retraining is computationally expensive, thereby creating a high entry barrier. As a result, pruning has been restricted to academic research and toy models. 

At DeGirum, we have designed HW from scratch to benefit from pruned networks. Other HW vendors may follow suit and offer HW that can exploit the sparsity in these networks. As pruned networks start going mainstream, it is important to develop a set of models for various applications. 

We would like to encourage developers to participate in our effort in the following ways:
1. Sponsor developers to work on model optimization related problems. We have identified some problems (see below) but would love to hear suggestions, comments and directions from the community at large. Sponsorship amount will depend on the scope of the problem and the effort needed. Currently, we are able to sponsor only using GitHub sponsors. For developers outside of the regions supported by GitHub, we would like to hear if anyone has any suggestions.
2. Since the compute load is not trivial for some of the tasks, we can run large scale training jobs on our HW if the developer provides us the script. At this point, we cannot yet provide direct access to HW.

If you are interested, you can contact us using this [link](https://degirum.com/AboutUscontacts).

## Problems
### EfficientNets for Classification
The table below summarizes the performance of various EfficientNet models. [EfficientNets](https://arxiv.org/pdf/1905.11946.pdf), proposed by Tan and Le, belong to a class of scalable image classification models that employs a compound scaling method to design models of increasing size and accuracy. They also use squeeze-and-excite modules and employ SiLU activation function (SiLU(x)=x*sigmoid(x)). Squeeze-and-excitation networks were proposed by Hu et al in this [paper](https://arxiv.org/pdf/1709.01507.pdf). The SiLU activation function was originally proposed by Elfwing, Uchibe and Doya in this [paper](https://arxiv.org/pdf/1702.03118.pdf). The Lite versions are optimized for mobile and other compute limited environments by eliminating the squeeze-and-excite modules and using simpler activation functions (such as ReLU6). This makes them more amenable to quantization. Details of the approach can be found at this [TensorFlow blog](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html). Code is available at the [TensorFlow TPU EfficientNet_Lite models repo.](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite) The EgdeTPU versions are optimized by taking into account the efficiency of the models when executed on Google's EdgeTPU HW. Details of the approach can be found in this [Google AI blog](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html).

|Model Name             | # of Params   | # of MADDS    | Top1 Acc. | Top 5 Acc.    | Input Size|
|---------------------- |------------:  |-----------:   |----------:|-----------:   |----------:|
|EfficientNet-B0        |	5.3M        |	0.39B       |	77.3    |	93.5        |	224     |
|EfficientNet-B1        |	7.8M        |	0.70B       |	79.2    |	94.5        |	240     |
|EfficientNet-B2        |	9.2M        |	1.0B        |	80.3    |	95.0        |	260     |
|EfficientNet-B3        |	12M         |	1.0B        |	81.7    |	95.6        |	300     |
|EfficientNet-B4        |	19M         |	4.2B        |	83.0    |	96.3        |	380     |
|EfficientNet-B5        |	30M         |	9.9B        |	83.7    |	96.7        |	456     |
|EfficientNet-B6        |  	43M         |	19B         |	84.2    |	96.8        |	528     |
|EfficientNet-B7        |	66M         |	37B         |	84.4    |	97.1        |	600     |
|EfficientNet-Lite0     |	4.65M       |               |	74.84   |	92.17       |	224     |
|EfficientNet-Lite1     |	5.42M       |               |	76.64   |	93.23       |	240     |
|EfficientNet-Lite2     |	6.09M       |               |	77.46   |	93.75       |	260     |
|EfficientNet-Lite3     |	8.2M        |               |	79.81   |	94.91       |	300     |
|EfficientNet-Lite4     |	13.0M       |               |	81.53   |	95.67       |	380     |
|EfficientNet-EdgeTPU-S |	5.44M       |		        |   77.26   |   93.6	    |   224     |
|EfficientNet-EdgeTPU-M |	6.90M       |		        |   78.74   |   94.33	    |   240     |
|EfficientNet-EdgeTPU-L |	10.59M      |		        |   80.53   |	95.19       |   300     |

**Note1**: EfficientNet-B* params, MADDS, Accuracy and Input size taken from the [original EfficientNet paper](https://arxiv.org/pdf/1905.11946.pdf) by Mingxing Tan and Quoc V. Le.

**Note2**: EfficientNet-Lite* params, Accuracy and Input size are from the [gen-efficientnet-pytorch repo](https://github.com/rwightman/gen-efficientnet-pytorch) by [Ross Wightman](https://github.com/rwightman) and the [TensorFlow TPU EfficientNet_Lite models repo.](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite)

**Note3**: EfficientNet-EdgeTPU-* params, Accuracy and Input size are from the [gen-efficientnet-pytorch repo](https://github.com/rwightman/gen-efficientnet-pytorch) by [Ross Wightman](https://github.com/rwightman)

In this project, we aim to develop Pruned versions of EfficientNet-Lite* and/or EfficientNet-EdgeTPU-*. Pruning the models leads to reducing the number of parameters as well as improving latency on HW that supports pruned models. The gains can be quite significant in some cases, where localizing a model on-chip can deliver much higher performance. A dense model may suffer from memory bottlenecks while the pruned model can benefit from fitting completely in the SRAM of the chip. This can also lead to decreasing the system cost as it can potentially eliminate DRAM usage for some cases.

Since Pruning allows us to compress the models, we also aim to develop more accurate Lite and EdgeTPU models. These models can be thought of being the counterparts for B4-B7 in terms of accuracy.

### EfficientDets for Object Detection
Tan, Pang and Le proposed a class of scalable object detection models known as [EfficientDets](https://arxiv.org/pdf/1911.09070.pdf), that employs weighted Bi-directional Feature Pyramid Networks(BiFPN) for fusing features across different scales along with efficient backbone networks for feature extraction. A compound scaling method is used to design models of increasing complexity and accuracy. The performance of various EfficientDets is summarized in the table below.

|Model Name     | Efficient Backbone| # of Params | # of MADDS  | val AP | Input Size|
|---------------|------------------:|------------:|------------:|-------:|----------:|
|EfficientDet-D0|	B0              |	3.9M      |	2.5B        |	34.3 |	512      |
|EfficientDet-D1|	B1              |	6.6M      |	6.1B        |	40.2 |	640      |
|EfficientDet-D2|	B2              |	8.1M      |	11B         |	43.5 |	768      |
|EfficientDet-D3|	B3              |	12M       |	25B         |	46.8 |	896      |
|EfficientDet-D4|	B4              |	21M       |	55B         |	49.3 |	1024     |
|EfficientDet-D5|	B5              |	34M       |	135B        |	51.3 |	1280     |
|EfficientDet-D6|	B6              |	52M       |	226B        |	52.2 |	1280     |
|EfficientDet-D7|	B6              |	52M       |	325B        |	53.4 |	1536     |

In this project, we aim to develop Lite versions of EfficientDets along with their pruned counterparts. There has already been some effort in this direction. [Soyeb Nagori](https://github.com/soyebn) trained an EfficientDet-Lite0 config using the code [efficientdet-pytorch repo](https://github.com/rwightman/efficientdet-pytorch) by [Ross Wightman](https://github.com/rwightman) and made the pre-trained weights available. See this [link](https://github.com/rwightman/efficientdet-pytorch#models) for details.

### Models for Other Applications
Another aim of this project is to use the classification models developed as backbones for other tasks such as segmentation, transfer learning and key-point detection (such as PoseNet). We will also study if pruned models perform well for transfer learning tasks and as backbones for other tasks.

### Pruning Methodology
Pruning is a well known neural network optimization technique. See the [Documentation Page](https://nervanasystems.github.io/distiller/index.html) of the [Distiller](https://github.com/NervanaSystems/distiller) repo from [Nervana Systems](https://github.com/NervanaSystems) for a very good introduction and overview of various pruning techniques. We do not aim to cover the theory or overview of pruning techniques here. 

Iterative pruning has previously given us very good results. See our [pruned models repo](https://github.com/DeGirum/pruned-models) for results on ResNet50. In this project, we will explore the validity of the [Lottery Ticket Hypothesis](https://arxiv.org/pdf/1803.03635.pdf) proposed by Frankle and Carbin, when applied to already optimized networks such as EfficientNtes. Specifically, we will explore if we can find _winning tickets_ that lead us to pruned models. Pruning 20% of the non-zero weights and retraining the network by rewinding to the original initialized weights should produce a series of models with decreasing number of non-zero parameters (0.8, 0.64, 0.512, 0.4096, 0.33, 0.26, 0.21 and so on). 
