import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as T

def freq_highpass_filter(img_tensor):
    # 输入 img_tensor: [1, 3, H, W] → 输出 [1, 3, H, W]
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255
    img_np = img_np.astype(np.uint8)

    filtered = []
    for c in range(3):
        channel = img_np[:, :, c]
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        # 构造高通滤波器
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        r = 20  # 半径
        mask = np.ones((rows, cols), np.uint8)
        mask[crow - r:crow + r, ccol - r:ccol + r] = 0

        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        filtered.append(img_back)

    filtered_img = np.stack(filtered, axis=2) / 255.0
    filtered_img = torch.tensor(filtered_img).permute(2, 0, 1).unsqueeze(0).float()
    return filtered_img
