# 导入相关包
from tqdm import tqdm
from model.UNetPP import NestedUNet
from utils.utils_metrics import compute_index, show_results
import numpy as np
import torch
import os
import cv2


# 设置测试集中超声图片、对应标签和模型预测结果的路径
def cal_miou(test_dir="D:\\Semantic Segmention\\Datasets\\kidney_datasets\\Test_Images",
             gt_dir="D:\\Semantic Segmention\\Datasets\\kidney_datasets\\Test_Labels",
             pred_dir="D:\\Semantic Segmention\\Datasets\\kidney_datasets\\Results\\ResMSCA_UNetPP"
             ):
    # miou_mode用于指定该文件运行时计算的内容
    # miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    global image_ids
    miou_mode = 0
    # 分类个数
    num_classes = 2
    # 区分的种类
    name_classes = ["background", "tumour"]
    # 加载模型
    if miou_mode == 0 or miou_mode == 1:
        # 如果没有提前建立预测结果保存的文件夹则先建立文件夹
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        net = NestedUNet(num_classes=1, input_channels=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('./params/best_model.pth', map_location=device))  # todo
        # 测试模式
        # net.eval()
        # print("Load model done.")
        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]
        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            img = cv2.imread(image_path)
            origin_shape = img.shape
            # 转为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (256, 256))
            # 转为batch为1，通道为1，大小为256*256的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # 预测
            pred = net(img_tensor)
            # 提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            # 使预测结果图片大小与原图一致，并恢复成RGB格式
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)
        print("Get predict result done.")
    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        # 执行计算指标的函数
        hist, MIoU, Dice, Precision, Recall = compute_index(gt_dir, pred_dir, image_ids, num_classes,
                                                           name_classes)
        print("Get miou done.")
        miou_out_path = "results/"
        show_results(miou_out_path, hist, MIoU, Dice, Precision, Recall, name_classes)


if __name__ == '__main__':
    cal_miou()
