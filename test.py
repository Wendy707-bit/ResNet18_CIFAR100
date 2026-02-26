import torch
import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataset import get_dataloaders
from model import build_model
from utils import UnNormalize
from config import *


#环境校验
print("Python版本：",sys.version_info)
print("PyTorch版本：",torch.__version__)
print("GPU是否可用：",torch.cuda.is_available())
device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
if torch.cuda.is_available():
    print("GPU名称：",torch.cuda.get_device_name())
print("测试设备：",device)  



#创建保存目录
os.makedirs(save_dir, exist_ok=True)

#加载测试集，下划线占位无需使用的类别
_, _, test_dataloader, classes, mean, std = get_dataloaders(
    batch_size=batch_size,
    data_root=data_root,
    val_split=val_split
)
unnormalize = UnNormalize(mean=mean, std=std)


#初始化模型&加载训练好的最优模型
model = build_model(device)  
best_model_path = os.path.join(save_dir, "best_model.ckpt")
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"先运行train.py训练！未找到模型：{best_model_path}")
model.load_state_dict(torch.load(best_model_path, map_location=device))
print(f"成功加载最佳模型：{best_model_path}")



#切换评估模式
model.eval()  
total = 0
correct = 0


all_mis_preds = []# 错误预测的类别
all_mis_labels = []# 真实类别
all_mis_images = []# 错误预测的图片

#禁用梯度计算，减少显存占用，提升运行速度
with torch.no_grad():  
    
    for images, labels in tqdm(test_dataloader):#tqdm显示测试进度
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)# 模型预测
        preds = torch.argmax(out, dim=1)# 取预测类别
        # 累计总样本数和正确数
        total += images.size(0)
        correct += (preds == labels).sum().item()
    
        # 收集错误预测的样本
        mis_preds_indice = torch.flatten((preds != labels).nonzero())# 找预测错的样本索引
        mis_preds = preds[mis_preds_indice]# 错判的类别
        mis_labels = labels[mis_preds_indice]# 真实类别
        mis_images = images[mis_preds_indice]# 错判的图片
        # 把错误样本存起来
        all_mis_preds.extend(mis_preds.cpu().numpy())
        all_mis_labels.extend(mis_labels.cpu().numpy())
        for i in range(mis_images.size(0)):
            all_mis_images.append(unnormalize(mis_images[i]).cpu())# 还原图片颜色
# 打印测试集整体准确率  
test_acc = correct / total  
print(f'CIFAR100 最佳模型测试集准确率：{correct}/{total}={test_acc}')




#错误样本图
fig = plt.figure(figsize=(12, 12))
plot_num = min(25, len(all_mis_images))
for i in range(plot_num):#画 25 张预测错误的图片
    plt.subplot(5, 5, i+1)
    plt.tight_layout()
    #将图片张量从(C,H,W)转为(H,W,C)
    img_np = np.transpose(all_mis_images[i].cpu(), (1, 2, 0))
    img_np = np.clip(img_np, 0, 1) #裁剪像素值到0-1
    plt.imshow(img_np, interpolation='none')
    plt.title("pred: {}, gt: {}".format(classes[all_mis_preds[i]], classes[all_mis_labels[i]]))
    plt.xticks([])
    plt.yticks([])

vis_img_path = os.path.join(save_dir, "misclassified_samples.png")
plt.savefig(vis_img_path)  
plt.close()
print(f"错误样本可视化图已保存到：{vis_img_path}")



#混淆矩阵
all_preds = []#初始化存储列表
all_labels = []


model.eval()
with torch.no_grad():
    for images, labels in test_dataloader:
        
        images = images.to(device)
        labels = labels.to(device)

        #模型前向传播
        out = model(images)
        preds = torch.argmax(out, dim=1)

        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

#混淆矩阵热力图
plt.figure(figsize=(20, 20))
sns.heatmap(cm, cmap='Blues', annot=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
cm_path = os.path.join(save_dir, "confusion_matrix.png")
plt.savefig(cm_path,bbox_inches='tight')
plt.close()
print(f"混淆矩阵已保存到：{cm_path}")


#整理测试结果字典
test_result = {
    "test_accuracy": round(test_acc, 4),
    "total_samples": total,
    "correct_samples": correct,
    "data_mean": mean,
    "data_std": std,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "device": str(device),
    "total_misclassified_samples": len(all_mis_images)
}


#保存入JSON文件
with open(os.path.join(save_dir, "best_model_test_result.json"), "w", encoding="utf-8") as f:
    json.dump(test_result, f, indent=4, ensure_ascii=False)


print(f"测试结果已完整保存到：{save_dir}/best_model_test_result.json")