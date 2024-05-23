# 基于GCN的因子合成

## net.py
对神经网络的定义

## utils.py
一些辅助函数

## utils_data.py
定义FeatureStorage类，包括处理因子，以及生成graph data


## main.ipynb
一个训练、测试的demo

## Remark
Data flow: csv data --> graph data --> input to neural network
each month has a graph
- Create graph and save 
    - node features
    - edges
    - label 
- input to neural networks
