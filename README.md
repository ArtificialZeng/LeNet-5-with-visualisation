# LeNet-5
![image](https://github.com/ArtificialZeng/readme_add_pic/raw/master/imagePath/111.jpg)

#![image](http://github.com/itmyhome2013/readme_add_pic/raw/master/images/nongshalie.jpg)
![image](https://github.com/ArtificialZeng/readme_add_pic/raw/master/imagePath/2020-03-04%2004-47-16.png)



This implements a slightly modified LeNet-5 [LeCun et al., 1998a] and achieves an accuracy of ~99% on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).


![Epoch Train Loss visualization](https://i.imgur.com/h4h7CrF.gif)

## Setup

Install all dependencies using the following command
numpy>=1.17.0

torch>=1.4.0

torchvision>=0.4.0

visdom>=0.1.6

Pillow==6.2.0

onnx==1.6.0
```
$ pip install -r requirements.txt
```

## Usage

Start the `visdom` server for visualization

```
$ python -m visdom.server
```

Start the training procedure

```
$ python run.py
```

See epoch train loss live graph at [`http://localhost:8097`](http://localhost:8097).

The trained model will be exported as ONNX to `lenet.onnx`. The `lenet.onnx` file can be viewed with [Neutron](https://www.electronjs.org/apps/netron)

## References

[[1](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.
