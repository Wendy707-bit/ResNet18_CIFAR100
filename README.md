1.项目名称：
ResNet18迁移实现CIFAR100图像分类

2.项目简介
基于PyTorch框架，使用ResNet18模型做迁移，完成了CIFAR100数据集100类图像的分类任务，实现了训练、验证、测试全流程，包含自动保存最佳模型、训练指标可视化、日志记录等功能，代码完全模块化拆分。

3.项目文件结构
├── model.py        #搭建ResNet18迁移学习模型
├── train.py        #核心训练逻辑（验证、最佳模型自动保存、训练日志记录）
├── test.py         #测试最佳模型，生成错误样本可视化、混淆矩阵
├── val.py          #验证最佳模型，输出验证集准确率
├── config.py       #相关超参数配置
├── dataset.py      #加载CIFAR100数据集，做数据预处理、数据集拆分
├── utils.py        #工具函数：计算数据集均值与标准差、图片反归一化

4.运行环境
torch
torchvision
matplotlib
numpy
tqdm
scikit-learn
seaborn
torchsummary

5.快速运行
先运行train.py，再运行val.py，最后运行test.py

6.核心实现要点
（1）加载ResNet18的ImageNet预训练权重，替换最后一层全连接层，适配CIFAR100的100分类任务；
（2）实现学习率预热与余弦退火、梯度累积、权重衰减策略，提升训练稳定性并防止过拟合；
（3）验证集准确率达到历史最高时，自动保存best_model.ckpt模型权重，覆盖效果较差的模型；
（4）自动计算CIFAR100训练集均值/标准差做针对性归一化，训练集加入数据增强（随机裁剪、翻转等）提升模型泛化能力；
（5）结果可视化，自动生成训练损失&准确率曲线、测试集错误预测样本图、混淆矩阵热力图；
（6）代码自动检测GPU/CPU；
（7）自动保存训练超参数、批次训练日志、验证、测试结果。

7.结果保存
所有训练、验证、测试结果均自动保存至./train_results_cifar100/目录，包括：
（1）验证集表现最佳的模型(best_model.ckpt , resnet18_cifar100.ckpt)；
（2）损失曲线(loss_curve.png)、准确率曲线(acc_curve.png)、错误样本可视化图(misclassified_samples.png)、混淆矩阵热力图(confusion_matrix.png)；
（3）验证集(best_model_val_result.json)&测试集(best_model_test_result.json)准确率json文件、训练超参数配置文件(train_config.json , train_config_with_logs.json)、训练批次详细日志(train_log.txt)、终端输出文件(terminal_output.txt) ；

8.核心结果
测试集准确率：82.53%
验证集表现：最高准确率74.88%（第18轮），20轮验证准确率稳定在71%-75%区间
训练集表现：20轮训练平均准确率逐步提升，第19轮达到最高83.95%，最终第20轮为83.66%，整体训练趋势平稳