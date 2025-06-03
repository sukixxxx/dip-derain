import cv2
import numpy as np
import pywt


# 加载彩色图像（BGR）
img_color = cv2.imread('/home/workspace/ryc/code/NLCL/datasets/LHP/input/10_0.png')  # 替换为你的图像路径

def color_median_filter_with_edge_protection(img_color, ksize=3):
    result = img_color.copy()
    for c in range(3):  # B, G, R
        channel = img_color[:, :, c]
        edges = cv2.Canny(channel, 50, 150)
        denoised = cv2.medianBlur(channel, ksize)
        channel[edges == 0] = denoised[edges == 0]
        result[:, :, c] = channel
    return result

result_median = color_median_filter_with_edge_protection(img_color)

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

result_wavelet = wavelet_denoise_color(img_color)



cv2.imwrite('/home/workspace/ryc/code/NLCL/Eg2.png', result_median)
cv2.imwrite('/home/workspace/ryc/code/NLCL/Eg3.png', result_wavelet)
