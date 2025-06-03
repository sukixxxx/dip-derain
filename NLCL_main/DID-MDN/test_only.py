from __future__ import print_function
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from misc import getLoader, weights_init
import models.derain_dense as net1
import models.derain_residual as net2
from myutils.vgg16 import Vgg16
from myutils import utils
import subprocess
from PIL import Image
import numpy as np
import torchvision.transforms as T
from preprocessor.freq_filter import freq_highpass_filter
from preprocessor.hilbert_filter import hilbert_transform
from preprocessor.shallow_cnn import ShallowCNN
import cv2
import pywt

input_dir = "/home/workspace/ryc/code/NLCL/datasets/RealRain/valA"
tmp_dir = "/home/workspace/ryc/code/NLCL/data_temp"  # NLCL会从这里读取
nlcl_dir = os.path.join(tmp_dir, "NLCL")
model_name = "NLCL"
exp_name = "RealRain"
mode = "freq"  # 可选 "freq" 或 "hilbert"
epoch = "latest"  # 或者指定具体的epoch
output_dir = os.path.join("/home/workspace/ryc/code/NLCL/results", exp_name, "test_{}".format(epoch), "images", "pred_B")

# 模型权重路径
DERAIN_MODEL_PATH = './netG_epoch_9_main.pth'
LABEL_MODEL_PATH = './classification/netG_epoch_9.pth'
RESIDUAL_MODEL_PATH = './residual_heavy/netG_epoch_6.pth'

# 参数设置
BATCH_SIZE = 1
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def preprocess_image(image_path, mode='freq'):
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    tensor = transform(image).unsqueeze(0)

    if mode == 'freq':
        filtered = freq_highpass_filter(tensor)
        alpha = 0.8
        return alpha * tensor + (1 - alpha) * filtered
    elif mode == 'hilbert':
        filtered = hilbert_transform(tensor)
        alpha = 0.85
        return alpha * tensor + (1 - alpha) * filtered
    # elif mode == 'cnn':
    #     shallow_cnn = ShallowCNN()
    #     shallow_cnn.load_state_dict(torch.load("checkpoints/shallow_cnn.pth"))
    #     shallow_cnn.eval() # 假设你有一个预训练的CNN模型
    else:
        raise ValueError("Unsupported preprocess mode.")

def save_tensor_as_image(tensor, save_path):
    image = tensor.squeeze(0).permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(save_path)

def color_median_filter_with_edge_protection(img_color, ksize=3):
    result = img_color.copy()
    for c in range(3):  # B, G, R
        channel = img_color[:, :, c]
        edges = cv2.Canny(channel, 50, 150)
        denoised = cv2.medianBlur(channel, ksize)
        channel[edges == 0] = denoised[edges == 0]
        result[:, :, c] = channel
    return result

def wavelet_denoise_color(img_color, wavelet='db1'):
    def wavelet_denoise_channel(channel):
        coeffs = pywt.wavedec2(channel, wavelet, level=2)
        cA, (cH1, cV1, cD1), (cH2, cV2, cD2) = coeffs

        sigma = np.median(np.abs(cD2)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(channel.size))

        def thresholding(data):
            return pywt.threshold(data, threshold, mode='hard')

        cH1, cV1, cD1 = map(thresholding, (cH1, cV1, cD1))
        cH2, cV2, cD2 = map(thresholding, (cH2, cV2, cD2))

        coeffs = (cA, (cH1, cV1, cD1), (cH2, cV2, cD2))
        return pywt.waverec2(coeffs, wavelet)

    result = np.zeros_like(img_color)
    for c in range(3):  # BGR
        denoised = wavelet_denoise_channel(img_color[:, :, c])
        result[:, :, c] = np.clip(denoised, 0, 255)
    return result.astype(np.uint8)


# 加载模型
netG = net1.Dense_rain().to(DEVICE)
netG.load_state_dict(torch.load(DERAIN_MODEL_PATH, map_location=DEVICE))
netG.eval()

net_label = net1.vgg19ca().to(DEVICE)
net_label.load_state_dict(torch.load(LABEL_MODEL_PATH, map_location=DEVICE))
net_label.eval()

residue_net = net2.Dense_rain_residual().to(DEVICE)
residue_net.load_state_dict(torch.load(RESIDUAL_MODEL_PATH, map_location=DEVICE))
residue_net.eval()

# Softmax层
softmax = nn.Softmax(dim=1)

# 归一化函数
def norm_range(t):
    t = t.clone()
    t = t.clamp(t.min(), t.max())
    t = (t - t.min()) / (t.max() - t.min())
    return t

# 创建保存目录
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

count_nlcl = 0
count_all = 0
os.makedirs(nlcl_dir, exist_ok=True)


# 加载验证数据
val_loader = getLoader(
    datasetName='pix2pix_val',     # 必须匹配你自定义的数据集模块
    dataroot=input_dir,   #若跳过频域分析：input_dir, 否则tmp_dir  # 输入目录
    originalSize=IMG_SIZE,         # 比如512
    imageSize=IMG_SIZE,            # 对应fineSize
    batchSize=1,                   # 通常测试1张图
    workers=1,                     # 线程数（按需调整）
    mean=(0.5, 0.5, 0.5),          # 与训练保持一致
    std=(0.5, 0.5, 0.5),
    split='val',                   # 测试模式下不裁剪
    shuffle=False,                 # 保证顺序一致
    seed=42                        # 保证可复现
)
all_files = sorted([f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"))])
# 开始测试
for i, data in enumerate(val_loader):
    print(f"Processing image {i}...")

    val_input_cpu, _, image_path, originsize = data
    val_input_cpu = val_input_cpu.float().to(DEVICE)
    fname = os.path.basename(image_path[0])
    with torch.no_grad():
        input_var = Variable(val_input_cpu)
        
        # 1. 残差预测
        fake_clear = residue_net(input_var, torch.zeros(BATCH_SIZE, dtype=torch.long).to(DEVICE))
        residue = input_var - fake_clear

        # 2. 雨量密度分类
        label_logits = net_label(residue)
        label = softmax(label_logits)
        label_class = label.argmax(1) + 1  # 从1开始
        num = label_class.item()
        count_all += 1

        if num == 3:
            count_nlcl += 1
            nlcl_path = os.path.join(nlcl_dir, fname)
            os.system(f"cp '{image_path}' '{nlcl_path}'")
        else:
            # # 3. 去雨推理
            # _, derain_result = netG(input_var, label_class)
            # # 4. 保存图像
            # derain_img = derain_result[0].cpu()  # 只取batch中的第一个
            # #derain_img = norm_range(derain_img)
            # derain_img = derain_img * 0.5 + 0.5  # 反归一化到[0, 1]
            # ndarr = derain_img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            # im = Image.fromarray(ndarr)
            # im = im.resize(originsize, Image.BILINEAR)


            # im.save(os.path.join(output_dir, fname))
            img = cv2.imread(image_path)
            img_preprocessed = color_median_filter_with_edge_protection(img) #  color_median_filter_with_edge_protection or wavelet_denoise_color
            cv2.imwrite(os.path.join(nlcl_dir, fname), img_preprocessed)
all_files = sorted([f for f in os.listdir(nlcl_dir) if f.endswith((".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"))])
for fname in all_files:
    image_path = os.path.join(input_dir, fname)
    tmp_path = os.path.join(tmp_dir, fname)
    tensor = preprocess_image(image_path, mode=mode)
    save_tensor_as_image(tensor, tmp_path)

print("[INFO] NLCL test bash...")
cmd = [
    "python", "/home/workspace/ryc/code/NLCL/test.py",
    "--dataroot", tmp_dir,
    "--model", model_name,
    "--name", exp_name,
    "--dataset_mode", "single",
    "--preprocess", "None",
    "--epoch", epoch
]
subprocess.run(cmd, check=True)
print(f"[INFO] Processed {count_all} images, {count_nlcl} of which were sent to NLCL preprocessing.")
print(f"[INFO] Results saved to {output_dir}")
print("all done!")
