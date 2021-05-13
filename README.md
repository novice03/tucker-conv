## Tucker Convolutional Layers

PyTorch implementation of Tucker Convolutional Layers as introduced in [MobileDets: Searching for Object Detection Architectures for Mobile Accelerators](https://arxiv.org/abs/2004.14525v3). Ross Wightman's timm library has been used for some helper functions and inspiration for syntax style.

## Installation

```bash
$ pip install tucker-conv
```

## Usage

```python
from tucker_conv.conv import TuckerConv
import torch

tucker = TuckerConv(30, 30, in_comp_ratio = 0.25, out_comp_ratio = 0.75)
input = torch.randn([1, 30, 512, 512])

output = tucker(input)
```