

图像去噪:

许多实际应用的必要步骤

对于一个噪声图像y=x+v,x原始图像,v为噪声

DNCNN学习预测V

DNCNN假设v是高斯噪声

# 重建后图像的评判:

## 峰值信噪比PSNR

给定一个大小为![[公式]](https://www.zhihu.com/equation?tex=m%C3%97n)的干净图像![[公式]](https://www.zhihu.com/equation?tex=I)和噪声图像![[公式]](https://www.zhihu.com/equation?tex=K)，均方误差![[公式]](https://www.zhihu.com/equation?tex=%28MSE%29)定义为：

![[公式]](https://www.zhihu.com/equation?tex=MSE+%3D+%5Cfrac%7B1%7D%7Bmn%7D%5Csum_%7Bi%3D0%7D%5E%7Bm-1%7D%5Csum_%7Bj%3D0%7D%5E%7Bn-1%7D%5BI%28i%2C+j%29-K%28i%2Cj%29%5D%5E2)

然后![[公式]](https://www.zhihu.com/equation?tex=PSNR+%28dB%29)就定义为：

![[公式]](https://www.zhihu.com/equation?tex=PSNR+%3D+10+%5Ccdot+log_%7B10%7D%28%5Cfrac%7BMAX_I%5E2%7D%7BMSE%7D%29)

其中![[公式]](https://www.zhihu.com/equation?tex=MAX_I%5E2)为图片可能的最大像素值。如果每个像素都由 8 位二进制来表示，那么就为 255。通常，如果像素值由![[公式]](https://www.zhihu.com/equation?tex=B)位二进制来表示，那么![[公式]](https://www.zhihu.com/equation?tex=MAX_I+%3D+2%5EB-1)。

一般地，针对 uint8 数据，最大像素值为 255,；针对浮点型数据，最大像素值为 1。

上面是针对灰度图像的计算方法，如果是彩色图像，通常有三种方法来计算。

- 分别计算 RGB 三个通道的 PSNR，然后取平均值。
- 计算 RGB 三通道的 MSE ，然后再除以 3 。
- 将图片转化为 YCbCr 格式，然后只计算 Y 分量也就是亮度分量的 PSNR。



## SSIM (Structural SIMilarity) 结构相似性

![[公式]](https://www.zhihu.com/equation?tex=SSIM)公式基于样本![[公式]](https://www.zhihu.com/equation?tex=x)和![[公式]](https://www.zhihu.com/equation?tex=y)之间的三个比较衡量：亮度 (luminance)、对比度 (contrast) 和结构 (structure)。

![[公式]](https://www.zhihu.com/equation?tex=l%28x%2Cy%29+%3D+%5Cfrac%7B2%5Cmu_x+%5Cmu_y+%2B+c_1%7D%7B%5Cmu_x%5E2%2B+%5Cmu_y%5E2+%2B+c_1%7D)![[公式]](https://www.zhihu.com/equation?tex=c%28x%2Cy%29+%3D+%5Cfrac%7B2%5Csigma_x+%5Csigma_y+%2B+c_2%7D%7B%5Csigma_x%5E2%2B+%5Csigma_y%5E2+%2B+c_2%7D)![[公式]](https://www.zhihu.com/equation?tex=s%28x%2Cy%29+%3D+%5Cfrac%7B%5Csigma_%7Bxy%7D+%2B+c_3%7D%7B%5Csigma_x+%5Csigma_y+%2B+c_3%7D)

一般取![[公式]](https://www.zhihu.com/equation?tex=c_3+%3D+c_2+%2F+2)。

- ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_x)为![[公式]](https://www.zhihu.com/equation?tex=x)的均值
- ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu_y)为![[公式]](https://www.zhihu.com/equation?tex=y)的均值
- ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_x%5E2)为![[公式]](https://www.zhihu.com/equation?tex=x)的方差
- ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_y%5E2)为![[公式]](https://www.zhihu.com/equation?tex=y)的方差
- ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_%7Bxy%7D)为![[公式]](https://www.zhihu.com/equation?tex=x)和![[公式]](https://www.zhihu.com/equation?tex=y)的协方差
- ![[公式]](https://www.zhihu.com/equation?tex=c_1+%3D+%28k_1L%29%5E2%2C+c_2+%3D+%28k_2L%29%5E2)为两个常数，避免除零
- ![[公式]](https://www.zhihu.com/equation?tex=L)为像素值的范围，![[公式]](https://www.zhihu.com/equation?tex=2%5EB-1)
- ![[公式]](https://www.zhihu.com/equation?tex=k_1%3D0.01%2C+k_2%3D0.03)为默认值

那么

![[公式]](https://www.zhihu.com/equation?tex=SSIM%28x%2C+y%29+%3D+%5Bl%28x%2Cy%29%5E%7B%5Calpha%7D+%5Ccdot+c%28x%2Cy%29%5E%7B%5Cbeta%7D+%5Ccdot+s%28x%2Cy%29%5E%7B%5Cgamma%7D%5D)

将![[公式]](https://www.zhihu.com/equation?tex=%5Calpha%2C%5Cbeta%2C%5Cgamma)设为 1，可以得到

![[公式]](https://www.zhihu.com/equation?tex=SSIM%28x%2C+y%29+%3D+%5Cfrac%7B%282%5Cmu_x+%5Cmu_y+%2B+c_1%29%282%5Csigma_%7Bxy%7D%2Bc_2%29%7D%7B%28%5Cmu_x%5E2%2B+%5Cmu_y%5E2+%2B+c_1%29%28%5Csigma_x%5E2%2B%5Csigma_y%5E2%2Bc_2%29%7D)

每次计算的时候都从图片上取一个![[公式]](https://www.zhihu.com/equation?tex=N%C3%97N)的窗口，然后不断滑动窗口进行计算，最后取平均值作为全局的 SSIM。

```python
# im1 和 im2 都为灰度图像，uint8 类型
ssim = skimage.measure.compare_ssim(im1, im2, data_range=255)
```

# 网络结构:

```
import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3                         #卷积核大小为3
        padding = 1                             #填充1    
        features = 64                           #特征图维度=64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))                    #3x3 same卷积+Relu激活
        for _ in range(num_of_layers-2):                        #16层，每层都是3x3卷积+BN+RELU
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

model=DnCNN(3)

print(model)
```

1层3x3+relu,15层3x3same +BN+relu+1层3x3same



# 训练设置:

输入噪声图像，输出噪声

损失函数:MSELOSS

使用adam训练

