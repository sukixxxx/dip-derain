from skimage.metrics import structural_similarity as ssim
import torch
#from piq import niqe
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import math
import os

def calculate_psnr_rgb(img1, img2):
    """
    img1, img2: RGB图像，格式为 numpy.ndarray，shape = (H, W, 3)，类型为 uint8
    """
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_ssim_rgb(img1, img2):
    """
    img1, img2: RGB图像，格式为 numpy.ndarray，shape = (H, W, 3)
    """
    min_hw = min(img1.shape[0], img1.shape[1])
    # win_size必须是奇数，且 <= 图像最小边长
    win_size = min(7, min_hw if min_hw % 2 == 1 else min_hw - 1)

    if win_size < 3:
        print(f"跳过图像太小: {img1.shape}")
        return None

    score = ssim(img1, img2, data_range=255, channel_axis=2, win_size=win_size)
    return score


# def calculate_niqe_rgb(image_path):
#     """
#     输入图像路径，计算 RnvGB 图像 NIQE 值（越小越好）
#     """
#     img = Image.open(image_path).convert("RGB")
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),  # NIQE要求尺寸不能太小
#         transforms.ToTensor()  # [0, 1] float32
#     ])
#     img_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, H, W]
#     score = niqe(img_tensor, data_range=1.0)
#     return score.item()

def evaluate_folder(gt_dir, pred_dir):
    psnr_list, ssim_list, niqe_list = [], [], []
    
    filenames = sorted(os.listdir(pred_dir))
    
    for fname in filenames:
        #gt_fname = fname.replace(".png", "gt.png")  # 修改为你的GT命名方式
        #gt_fname = fname
        gt_fname = fname.replace("-R-", "-C-") 
        gt_path = os.path.join(gt_dir, gt_fname)
        pred_path = os.path.join(pred_dir, fname)
        
        if not os.path.exists(gt_path):
            print(f"跳过未匹配的文件：{fname}")
            continue
        
        # 读取 RGB 图像
        gt = cv2.imread(gt_path)
        pred = cv2.imread(pred_path)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        # Resize pred 与 gt 尺寸一致
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        # PSNR & SSIM
        psnr_list.append(calculate_psnr_rgb(gt, pred))
        score = calculate_ssim_rgb(gt, pred)
        if score is not None:
            ssim_list.append(score)
        # NIQE（只对生成图计算）
        #niqe_list.append(calculate_niqe_rgb(pred_path))
    
    print(f"平均 PSNR: {np.mean(psnr_list):.4f}")
    print(f"平均 SSIM: {np.mean(ssim_list):.4f}")
    #print(f"平均 NIQE: {np.mean(niqe_list):.4f}")

# 用法示例
gt_dir = "/home/workspace/ryc/code/NLCL/datasets/GT-RAIN/test/gt"       # ground truth 图像目录
pred_dir = "/home/workspace/ryc/code/NLCL/results/rain1400_pretrain_attn2/test_latest/images/pred_B"   # 预测图像目录
evaluate_folder(gt_dir, pred_dir)


# gt = cv2.imread("/home/workspace/ryc/code/NLCL/datasets/LHP/gt/2_0.png")
# pred = cv2.imread("/home/workspace/ryc/code/NLCL/results/RealRain/test_latest/images/pred_B/preprocessed.png")
# gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
# pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

# # Resize pred 与 gt 尺寸一致
# if pred.shape[:2] != gt.shape[:2]:
#     pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
#         # PSNR & SSIM
# print(calculate_psnr_rgb(gt, pred))
# print(calculate_ssim_rgb(gt, pred))
