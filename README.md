# SpatiotemporaL predicting
## 模型架构示意图
![image](https://github.com/xugh1997/spatial_tempora_predictionng/blob/main/introduction.png)
该时空预测模型使用自适应的超图卷积建模高阶多元空间依赖，使用基于全连接图的图注意力机制建模二元对偶的空间依赖，并使用基于因果卷积的TCN建模时间依赖关系
## Dependencies:
### cudf
### torch
### geopandas
### geopy
### numpy
### tqdm
## 运行步骤
### 第一步：python extract_pop_from_mobile_phone.py.py
#### 提取城市居民活动强度
### 第二步： python main.py
#### 进行模型的训练和推断
## 文件说明
### 核心代码
- `extract_pop_from_mobile_phone.py` 提前人口热力值
- `main.py` - 程序主入口
- `config.yml` - 配置文件，包含常量和配置参数
- `dataloader.py` - 加载数据
- `compute_loss.py` - 损失函数
### 模型相关
- `ST_BLOCK.py` - 主模型架构
- `Temporal_learning.py` - 时间依赖关系学习
- `Spatial_learning.py` - 空间依赖关系学习
