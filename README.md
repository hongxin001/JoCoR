# JoCoR 
CVPR'20: [Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_with_CVPR_2020_paper.html).
(Pytorch implementation)





# Abstract

Deep Learning with noisy labels is a practically challenging problem in weakly-supervised learning. The state-of-the-art approaches "Decoupling" and "Co-teaching+" claim that the "disagreement" strategy is crucial for alleviating the problem of learning with noisy labels. In this paper, we start from a different perspective and propose a robust learning paradigm called JoCoR, which aims to reduce the diversity of two networks during training. Specifically, we first use two networks to make predictions on the same mini-batch data and calculate a joint loss with Co-Regularization for each training example. Then we select small-loss examples to update the parameters of both two networks simultaneously. Trained by the joint loss, these two networks would be more and more similar due to the effect of Co-Regularization. Extensive experimental results on corrupted data from benchmark datasets including MNIST, CIFAR-10, CIFAR-100 and Clothing1M demonstrate that JoCoR is superior to many state-of-the-art approaches for learning with noisy labels.


## Running JoCoR on benchmark datasets (MNIST, CIFAR-10 and CIFAR-100)
Here is an example: 

```bash
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5 
```

## About Lambda
Here is an example with lambda setting: 

```bash
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.5 --co_lambda 0.9
```

Note: 
The best Lambda depends on the datasets, noisy rate and basic models. Generally, the best lambda is between 0.5-0.9;

For example, in my experiment, the best lambda for CIFAR10 with 20%,50% and asym 40% noise is 0.9. When it comes to 80% noise, the best is 0.65.
For Cifar100 in my experiment, the best lambda should be 0.85.

If you change the basic model to ResNet or add normalization in dataloader, you need to try different lambda for the best.


## Citation

```
@article{wei2020combating,
  title={Combating noisy labels by agreement: A joint training method with co-regularization},
  author={Wei, Hongxin and Feng, Lei and Chen, Xiangyu and An, Bo},
  journal={CVPR},
  year={2020}
}
```
