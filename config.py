#超参数配置文件

#数据相关配置
batch_size = 64          #批次大小
data_root = "./data"     #数据集存放路径
val_split=0.1

#训练相关配置
num_epochs = 20           #训练轮数
learning_rate = 2e-3     #初始学习率
accumulation_steps = 1   #梯度累积步数（模拟大批次训练）

#优化器配置（SGD）
momentum = 0.9           #动量参数（让训练更稳定）
weight_decay = 3e-4      #权重衰减（防止过拟合）

#动态学习率（warmup+余弦退火）配置
warmup_epochs = 2
T_max = num_epochs - warmup_epochs

#结果保存配置
save_dir = "./train_results_cifar100"  #训练结果（日志/模型/图片）保存路径