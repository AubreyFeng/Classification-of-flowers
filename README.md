# 人工智能课程设计 *五种花进行分类*

1. 使用VGGNet、GoogleNet、ResNet、DenseNet、EfficientNet和数据集中80%的数据训练识别模型，并对剩下20%的数据集进行测试 ；
2. 使用不同的评价指标（如accuracy、precision、recall等）对各种算法进行评价；
3. 鼓励基于现有的算法提出改进，进一步提高算法的性能。

## 该文件夹存放使用pytorch实现的代码版本

**model.py**： 是模型文件

**train.py**： 是调用模型训练的文件

**predict.py**： 是调用模型进行预测的文件

**class_indices.json**： 是训练数据集对应的标签文件

---

### Install

```
git clone https://github.com/AubreyFeng/Classification-of-flowers.git
cd Classification-of-flowers
pip3 install -r requestments.txt
```

### 数据集下载

下载好数据集，代码中默认使用的是花分类数据集，下载地址:

[https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)

### Train

在进入目录后，修改数据集路径，输入以下命令。

```bash
python train.py
```

### Predict

```
python predict.py
```
