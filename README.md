# 45种猫分类器及UI实现
## 1.物体检测部分（Faster RCNN）

### 环境配置：
* 详细环境配置见```requirements.txt```

### 文件结构：
```
  ├── backbone: 特征提取网络
  ├── network_files: Faster R-CNN网络
  ├── train_utils: 训练验证相关模块
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train_resnet50_fpn.py: 以resnet50+FPN做为backbone进行训练
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据，并生成record_mAP.txt文件
  ├── extraction.py: 利用训练好的权重识别指定图像的边界框，用于后续分类网络的处理
  ├── premanage.py: 对图片数据集进行增强处理
  └── pascal_voc_classes.json: pascal_voc标签文件
```

### 预训练权重下载地址：
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
 
 
### 数据集，本例程使用的是PASCAL VOC2017数据集

### 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重

## 2.分类器部分（UI）
###文件结构
```
  ├── models: 预训练参数
  ├── classifer.py: 分类网络框架
  ├── classification.py: 猫分类器
  ├── UI.py: 图形界面
```
### 使用方法
* python ./UI.py
* 选择识别的图片
* 点击预测框获取预测信息

## 3.数据集
在网络上关于猫种类的数据集较为缺乏，我们利用质量较高的原数据集训练得到的模型，对新数据集进行一定的筛选。
我们构建出了一个拥有45种纯种猫品种，共约2万9千张图片的相对高质量的数据集，填补了这一方面这一空白，
为后续的工作提供了便利。

###数据集链接：
* https://pan.baidu.com/s/16Kk7XhNAurjLNiokWzFLPg 提取码：v4gw 
