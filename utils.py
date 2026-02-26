import torch  


def get_mean_and_std(dataset):

    #构建分批加载数据，每批1024张
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)

    
    mean = torch.zeros(3)  # 初始化3通道均值
    std = torch.zeros(3)   # 初始化3通道标准差
    total_samples = 0      # 累计样本数
    
    
    for data, _ in dataloader:
        batch_samples = data.size(0)  # 当前批次的样本数
        data = data.view(batch_samples, 3, -1)  # 展平H×W维度，只保留「批次×通道×像素数」
        mean += data.mean(2).sum(0)  # 按通道累加均值
        std += data.std(2).sum(0)    # 按通道累加标准差
        total_samples += batch_samples
    
    
    mean /= total_samples
    std /= total_samples
    return mean.tolist(), std.tolist()



#反归一化，还原图片用于可视化
class UnNormalize(object):
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    
    def __call__(self, tensor):
        #把标准化后的图片还原成原始像素范围（0-1），方便画图
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        tensor = torch.clamp(tensor, 0, 1)
        return tensor




