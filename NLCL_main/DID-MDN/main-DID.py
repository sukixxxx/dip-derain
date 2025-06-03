import os
import subprocess
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch
import torch.nn as nn
from torch.autograd import Variable
from preprocessor.freq_filter import freq_highpass_filter
from classification.classification import Classifier
from misc import getLoader
import models.derain_dense as net1
import models.derain_residual as net2
from myutils import utils

# 模型路径
DERAIN_MODEL_PATH = './netG_epoch_9_main.pth'
LABEL_MODEL_PATH = './classification/netG_epoch_9.pth'
RESIDUAL_MODEL_PATH = './residual_heavy/netG_epoch_6.pth'

# 设置路径
INPUT_DIR = "/home/workspace/ryc/code/NLCL/datasets/SPA-Testing/Real_Internet"
NLCL_INPUT_DIR = "/home/workspace/ryc/code/NLCL/data_temp"
RESULT_SAVE_DIR = './result'
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
os.makedirs(NLCL_INPUT_DIR, exist_ok=True)

# 分类器
classifier = Classifier(
    window_size=(10,10),
    whether_client=False,
    workers=8,
    verbose=False,
    grid_search=False,
    model_path='./classification/svm_model-4-grid_searched-200x200.joblib'
)

# Torch 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

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

softmax = nn.Softmax(dim=1)

# 预处理函数
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = T.ToTensor()(image).unsqueeze(0)
    filtered = freq_highpass_filter(tensor)
    alpha = 0.8
    return alpha * tensor + (1 - alpha) * filtered

# 保存tensor图像
def save_tensor_as_image(tensor, save_path):
    image = tensor.squeeze(0).permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(save_path)

# 用去雨模型推理并保存结果
def derain_and_save(image_path, save_path):
    image = Image.open(image_path).convert("RGB")
    origin_size = image.size
    tensor = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        input_var = Variable(tensor)
        fake_clear = residue_net(input_var, torch.zeros(1, dtype=torch.long).to(DEVICE))
        residue = input_var - fake_clear
        label_logits = net_label(residue)
        label = softmax(label_logits)
        label_class = label.argmax(1) + 1

        _, derain_result = netG(input_var, label_class)
        derain_img = derain_result[0].cpu()
        derain_img = derain_img * 0.5 + 0.5  # 反归一化
        ndarr = derain_img.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        im = Image.fromarray(ndarr)
        im = im.resize(origin_size, Image.BILINEAR)
        im.save(save_path)

# 主流程
def main():
    all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"))])

    for fname in all_files:
        image_path = os.path.join(INPUT_DIR, fname)
        result_path = os.path.join(RESULT_SAVE_DIR, fname)
        nlcl_path = os.path.join(NLCL_INPUT_DIR, fname)

        print(f"\n[INFO] Processing {fname} ...")
        rain_class = classifier.test(classifier.model_path, cv2.imread(image_path))[0]
        print(f"[INFO] Rain class: {rain_class}")

        if rain_class == "heavy":
            print(f"[INFO] Using NLCL preprocessing for {fname}")
            tensor = preprocess_image(image_path)
            save_tensor_as_image(tensor, nlcl_path)
        else:
            print(f"[INFO] Using general derain model for {fname}")
            derain_and_save(image_path, result_path)

    # 如果存在预处理图像，运行NLCL模型
    if len(os.listdir(NLCL_INPUT_DIR)) > 0:
        print("\n[INFO] Running NLCL inference...")
        cmd = [
            "python", "test.py",
            "--dataroot", NLCL_INPUT_DIR,
            "--model", "NLCL",
            "--name", "RealRain",
            "--dataset_mode", "single",
            "--preprocess", "None"
        ]
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
