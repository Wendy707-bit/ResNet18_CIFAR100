import torch#用来搭建模型、计算损失、反向传播
from torch import nn#用来搭建模型、计算损失、反向传播
from datetime import datetime#打印训练时间，方便看训练进度
import sys#打印 Python / 库版本，校验环境
from tqdm import tqdm#生成训练 / 测试进度条，直观看到当前跑了多少数据
import matplotlib.pyplot as plt
import numpy as np 
from dataset import get_dataloaders#导入数据集加载函数
from utils import get_mean_and_std
from utils import UnNormalize
from model import build_model,print_model_structure
import json  #保存参数用
import os    #创建文件夹用
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
from config import *

plt.rcParams['axes.unicode_minus'] = False#使得负号正常显示
plt.rcParams['font.family'] = 'DejaVu Sans'

#环境校验，确认工具能用
print("Python版本：",sys.version_info)
print("PyTorch版本：",torch.__version__)
print("GPU是否可用：",torch.cuda.is_available())
device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
if torch.cuda.is_available():
    print("GPU名称：",torch.cuda.get_device_name())
print("训练设备：",device)
#打印 Python、PyTorch、TorchVision 版本
#检查 GPU（CUDA）是否可用


#创建保存目录，不存在就创建，存在不报错
os.makedirs(save_dir, exist_ok=True) 



#对接dataset.py
train_dataloader, val_dataloader,test_dataloader, classes,mean,std = get_dataloaders(
    batch_size=batch_size,
    data_root=data_root, 
    val_split=val_split
)
unnormalize = UnNormalize(mean=mean, std=std)#反归一化

#定义要保存的核心超参数
train_config = {
    "batch_size": batch_size,
    "data_root":data_root,
    "val_split":val_split,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "accumulation_steps": accumulation_steps,
    "optimizer": f"SGD (momentum={momentum}, weight_decay={weight_decay})",
    "scheduler": f"CosineAnnealingLR (warmup={warmup_epochs}epochs, T_max={T_max})",
    "save_dir":save_dir,
    "loss_function": "CrossEntropyLoss",
    "device": device.type,
    "dataset": "CIFAR100",
    "model": "ResNet18 (transfer learning)",
    "mean": mean,
    "std": std,
    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

#保存基础超参数到JSON文件
config_path = os.path.join(save_dir, "train_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(train_config, f, indent=4, ensure_ascii=False)
print(f"超参数已保存到：{config_path}")


#模型训练
model =build_model(device)# 把模型放到GPU/CPU上
print_model_structure(model,device.type)


# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器：SGD，负责更新模型参数
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
total_batch = len(train_dataloader)# 每轮训练的批次总数（5万张/batch_size≈3125批）




#训练循环
train_logs = []#保存日志列表
best_val_acc = 0.0#初始化最佳准确率
val_acc_list = []  #初始化验证准确率列表


for epoch in range(num_epochs):#训练几轮
    epoch_start = time.time() 
    model.train()
    #warmup预热逻辑
    if epoch < warmup_epochs:
        warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    # warmup结束  

    #清空梯度
    optimizer.zero_grad()
    epoch_total_loss = 0.0  #初始化轮次损失
    epoch_total_acc = 0.0   #初始化轮次准确率

    for batch_idx, (images, labels) in enumerate(train_dataloader):#分批
        #把数据移到GPU/CPU
        images = images.to(device)
        labels = labels.to(device)
        
        #前向传播：模型预测&计算损失
        out = model(images)#模型输出
        loss = criterion(out, labels) / accumulation_steps#计算损失
        
        #算当前批次的准确率
        n_corrects = (out.argmax(axis=1) == labels).sum().item()#预测对的数量
        acc = n_corrects/labels.size(0)#准确率=对的数量/批次大小
        
        #累加每轮的损失和准确率
        epoch_total_loss += loss.item() * accumulation_steps
        epoch_total_acc += acc

        #反向传播&更新参数
        loss.backward()#反向求梯度
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()   #按梯度调整参数，让损失变小
            optimizer.zero_grad()# 梯度清零
        
        #每100批打印日志
        if (batch_idx+1) % 100 == 0:
            log_info = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch+1,
            "batch": batch_idx+1,
            "total_batch": total_batch,
            "loss": round(loss.item()*accumulation_steps, 4),
            "acc": round(acc, 4)
            }
            train_logs.append(log_info)
            print(f'{datetime.now()}, {epoch+1}/{num_epochs}, {batch_idx+1}/{total_batch}: {loss.item()*accumulation_steps:.4f}, acc: {acc:.4f}')



    epoch_time = time.time() - epoch_start
    avg_epoch_loss = epoch_total_loss / total_batch
    avg_epoch_acc = epoch_total_acc / total_batch
    print(f"第{epoch+1}轮结束 | 平均损失：{avg_epoch_loss:.4f} | 平均准确率：{avg_epoch_acc:.4f} | 耗时：{epoch_time:.2f}秒")
    train_config[f"epoch_{epoch+1}_time"] = f"{epoch_time:.2f}秒"
    

    #学习率调度
    if epoch >= warmup_epochs:
        scheduler.step()
    
    #切换到验证模式
    model.eval()
    val_total = 0
    val_correct = 0
    

    with torch.no_grad():
    #验证阶段，禁用梯度计算

        #遍历验证集数据    
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            #验证集前向传播&预测
            out = model(images)
            preds = torch.argmax(out, dim=1)
            
            #统计验证集正确数/总数
            val_total += images.size(0)
            val_correct += (preds == labels).sum().item()

    #计算本轮验证集准确率        
    val_acc = val_correct / val_total
    print(f"第{epoch+1}轮验证准确率：{val_acc:.4f}")

    #保存每轮验证准确率
    val_acc_list.append(val_acc)

    #保存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.ckpt"))
        print(f"发现更佳模型，验证准确率：{best_val_acc:.4f}，已保存")

    


#记录训练结束时间&整理日志
train_config["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
train_config["train_logs"] = train_logs

#保存成JSON文件
final_config_path = os.path.join(save_dir, "train_config_with_logs.json")
with open(final_config_path, "w", encoding="utf-8") as f:
    json.dump(train_config, f, indent=4, ensure_ascii=False)

#保存纯文本日志
txt_log_path = os.path.join(save_dir, "train_log.txt")
with open(txt_log_path, "w", encoding="utf-8") as f:
    f.write("="*50 + " 训练日志 " + "="*50 + "\n")
    f.write(f"训练配置：{json.dumps(train_config, indent=2, ensure_ascii=False)}\n")
    f.write("="*50 + " 批次日志 " + "="*50 + "\n")
    for log in train_logs:
        f.write(f'{log["time"]} | 轮数：{log["epoch"]} | 批次：{log["batch"]} | 损失：{log["loss"]} | 准确率：{log["acc"]}\n')
    f.write("="*50 + " 最终结果 " + "="*50 + "\n")
    f.write(f"测试集准确率：请查看test.py结果\n")
    f.write(f"训练开始时间：{train_config['start_time']}\n")
    f.write(f"训练结束时间：{train_config['end_time']}\n")


#从日志里提取数据
epochs = [log["epoch"] for log in train_logs]
batches = [log["batch"] for log in train_logs]
losses = [log["loss"] for log in train_logs]
accs = [log["acc"] for log in train_logs]


iterations = list(range(1, len(train_logs) + 1))

#loss曲线
plt.figure(figsize=(12, 5))
plt.plot(iterations, losses, label='Training Loss', color='#1f77b4')
plt.xlabel('Iteration')
plt.ylabel('Loss Value')
plt.title('Training Loss Curve')
plt.grid(True, alpha=0.3)
plt.legend()
loss_img_path = os.path.join(save_dir, "loss_curve.png")
plt.savefig(loss_img_path)
plt.close()

#acc曲线
plt.figure(figsize=(8, 4))
plt.plot(iterations, accs, label='Training Acc', color='#ff7f0e')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Curve')
plt.grid(True, alpha=0.3)
plt.legend()
acc_img_path = os.path.join(save_dir, "acc_curve.png")
plt.savefig(acc_img_path)
plt.close()



#保存最终模型权重
model_path = os.path.join(save_dir, "resnet18_cifar100.ckpt")
torch.save(model.state_dict(), model_path)

print(f"模型权重已保存到：{model_path}")
print(f"\n所有训练结果已统一保存到文件夹：{save_dir}")