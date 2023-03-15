import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import threading
import pandas as pd

# 定义EAST模型
class EAST(torch.nn.Module):
    def __init__(self, **kwargs):
        super(EAST, self).__init__()
        self.backbone = ...
        self.output_layer = ...

    def forward(self, x):
        features = self.backbone(x)
        output = self.output_layer(features)
        return output

# 定义数据转换器
def data_transform(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# 定义批量识别函数
def process_images(model, images, output_dir, thread_num):
    results = []
    for i, image in enumerate(images):
        # 读取图像
        img = cv2.imread(image)
        # 转换为PIL Image
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 数据转换
        img_tensor = data_transform(img)
        # 添加批量维度
        img_tensor = img_tensor.unsqueeze(0)
        # 模型预测
        with torch.no_grad():
            preds = model(img_tensor)
        # 处理预测结果
        ...
        # 保存结果
        filename = os.path.basename(image)
        cv2.imwrite(os.path.join(output_dir, filename), img)
        print(f"Thread {thread_num}: Processed image {i+1}/{len(images)}")
        results.append((filename, table))

    return results

# 定义主函数
def main():
    # 加载模型
    model = EAST(...)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # 定义输入输出路径
    input_dir = "input_images/"
    output_dir = "output_images/"
    output_file = "output.xlsx"

    # 获取输入图像列表
    images = glob.glob(os.path.join(input_dir, "*.jpg"))

    # 分割图像列表
    num_threads = 4
    num_images_per_thread = len(images) // num_threads
    thread_list = []
    for i in range(num_threads):
        start_idx = i * num_images_per_thread
        end_idx = start_idx + num_images_per_thread
        if i == num_threads - 1:
            end_idx = len(images)
        thread_images = images[start_idx:end_idx]
        t = threading.Thread(target=process_images, args=(model, thread_images, output_dir, i+1))
        t.start()
        thread_list.append(t)

    # 等待线程结束
    results = []
    for t in thread_list:
        t.join()
        results += t._target

    # 将结果输出为Excel
    df = pd.DataFrame(results, columns=["filename", "table"])
    df.to_excel(output_file, index=False)

if __name__ == "__main__":
    main()
