import torch
import os
import json
import sys
from dataset import get_dataloaders
from model import build_model
from config import *

#环境校验
print("Python版本：",sys.version_info)
print("PyTorch版本：",torch.__version__)
print("GPU是否可用：",torch.cuda.is_available())
device=torch.device('cuda'if torch.cuda.is_available()else'cpu')
if torch.cuda.is_available():
    print("GPU名称：",torch.cuda.get_device_name())
print("验证设备：",device)  



#创建保存目录
os.makedirs(save_dir, exist_ok=True)

#加载验证集，下划线占位不使用的类别
_, val_dataloader, _, _, mean, std = get_dataloaders(
    batch_size=batch_size,
    data_root=data_root,
    val_split=val_split
)


#初始化模型&加载训练好的最优模型
model = build_model(device)  
best_model_path = os.path.join(save_dir, "best_model.ckpt")
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"先运行train.py训练！未找到模型：{best_model_path}")
model.load_state_dict(torch.load(best_model_path, map_location=device))#加载之前训练好的模型
print(f"成功加载最佳模型：{best_model_path}")



#验证开始
model.eval()#切换为测试模式
val_total = 0
val_correct = 0
#验证阶段禁用梯度计算
with torch.no_grad():
    
    for images, labels in val_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        preds = torch.argmax(out, dim=1)#取概率最大的类别作为预测结果
        val_total += images.size(0)
        val_correct += (preds == labels).sum().item()
val_acc = val_correct / val_total
print(f"最佳模型验证集准确率：{val_acc:.4f}({val_correct}/{val_total})")


#将结果保存入json
val_result = {"val_accuracy": round(val_acc, 4), "total_samples": val_total,"correct_samples": val_correct,"data_mean": mean,
    "data_std": std,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "device": str(device)}
with open(os.path.join(save_dir, "best_model_val_result.json"), "w", encoding="utf-8") as f:
    json.dump(val_result, f, indent=4)
print(f"验证结果已保存到：{save_dir}/best_model_val_result.json")