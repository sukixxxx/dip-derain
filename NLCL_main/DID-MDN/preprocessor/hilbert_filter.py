import numpy as np
import torch
import torchvision.transforms as T
from scipy.signal import hilbert

def hilbert_transform(img_tensor):
    # 输入 img_tensor: [1, 3, H, W] → 输出 [1, 3, H, W]
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output = []

    for c in range(3):
        channel = img_np[:, :, c]
        # 每一行做Hilbert变换
        analytic = hilbert(channel, axis=0)
        amplitude = np.abs(analytic)
        output.append(amplitude)

    result = np.stack(output, axis=2)
    result = torch.tensor(result).permute(2, 0, 1).unsqueeze(0).float()
    return result
