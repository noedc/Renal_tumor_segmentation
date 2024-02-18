# 导入相关包
import os
from torchvision.utils import save_image
from utils.dataset import Kidney_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.utils.data
from model.UNetPP import NestedUNet


def train_net(net, device, data_path, epochs=50, batch_size=4, lr=1e-3):
    # 加载训练集
    kidney_dataset = Kidney_Loader(data_path)
    per_epoch_num = len(kidney_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=kidney_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义AdamW优化器更新网络参数
    optimizer = optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-8)
    # 定义二分类交叉熵损失函数
    criterion = nn.BCELoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    with tqdm(total=epochs * per_epoch_num) as pbar:
        for epoch in range(epochs):
            for image, label in train_loader:
                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # 使用网络参数，输出预测结果
                pred = net(image)
                # 使用sigmoid归一化满足二分类交叉熵损失函数期望的输入
                pred = torch.sigmoid(pred)
                # 计算预测值和真实值之间的损失
                loss = criterion(pred, label)
                print('{}/{}：Loss/train'.format(epoch + 1, epochs), loss.item())
                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'params/best_model.pth')
                # 更新参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
                # 拼接标签和模型预测结果
                _label_image = label[0]
                _pred_image = pred[0]
                img = torch.stack([_label_image, _pred_image], dim=0)
                save_image(img, f'{save_path}/{epoch+1}.png')


if __name__ == "__main__":
    # 加载预训练模型路径
    weight_path = 'params/best_model.pth'
    # 设置保存训练过程中生成的图片路径
    save_path = 'train_img'
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，分类为1，图片经dataset处理由RGB3通道变为单通道1。
    net = NestedUNet(num_classes=1, input_channels=1)
    # 加载训练权重文件
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight')
    else:
        print('not successful load weight')
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "D:\\Semantic Segmention\\Datasets\\kidney_datasets"  # todo 修改为你本地的数据集位置
    print("进度条出现卡着不动不是程序问题，是他正在计算，请耐心等待")
    train_net(net, device, data_path, epochs=50, batch_size=4, lr=1e-3)
