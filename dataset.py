import torch
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from utils import get_mean_and_std




#定义数据集加载函数
def get_dataloaders(batch_size=16, data_root='./data',val_split=0.1):
    
    train_dataset_temp= datasets.CIFAR100(root=data_root, 
                               download=False, 
                               train=True,
                               transform=transforms.ToTensor())
    mean,std=get_mean_and_std(train_dataset_temp)
    print(f"计算出CIFAR100均值：{mean}，标准差：{std}")
    

    
    #训练集预处理
    train_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),  
        transforms.RandomCrop(224, padding=16),         
        transforms.RandomHorizontalFlip(p=0.5),         
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.3)                 
        ])

    #验证集&测试集预处理
    val_test_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        ])


    #加载完整训练集
    full_train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=False,
        transform=train_transform
    )
    
    #拆分训练集和验证集
    val_size = int(len(full_train_dataset) * val_split)  
    train_size = len(full_train_dataset) - val_size      
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  #固定随机种子
    )




    #加载数据集
    test_dataset = datasets.CIFAR100(root=data_root, 
                               download=False, 
                               train=False, 
                               transform=val_test_transform)


    #自动获取CIFAR100的100个类别名
    classes = full_train_dataset.classes  


    #训练集 DataLoader
    #shuffle=True：每轮迭代打乱图片顺序
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               shuffle=True, 
                                               batch_size=batch_size)
    #验证集 DataLoader
    #shuffle=False：验证集不需要打乱
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False  )
    
    #测试集 DataLoader
    #shuffle=False：测试集不打乱
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                               shuffle=False, 
                                               batch_size=batch_size)
    
    
    return train_dataloader, val_dataloader, test_dataloader,classes,mean,std


