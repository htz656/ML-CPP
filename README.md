# 机器学习算法实现

## 项目介绍
本项目是机器学习算法的实现，参考《[【机器学习】【白板推导系列】【合集 1～33】](https://www.bilibili.com/video/BV1aE411o7qd/?share_source=copy_web&vd_source=b31fed0153f0e717f42b5b9b2f35ef08)》，实现了其中的一些经典模型，包括：贝叶斯网络、高斯过程回归、隐马尔可夫模型、PCA降维算法、线性回归模型、逻辑回归模型、支持向量机等。

## 文件结构
```
机器学习模型实现/
├── .gitignore
├── CMakeLists.txt
├── LICENSE
├── README.md
├── data/
├── notes/
├── include/
│   ├── base/
│   │   ├── math_util.h
│   │   └── types.h
│   ├── models/
│   │   └── ...
│   └── utils/
│       ├── file_util.h
│       └── log_util.h
├── src/
│   ├── main.cpp 
│   ├── base/
│   ├── models/
│   └── utils/
├── test/
└── ...             
```