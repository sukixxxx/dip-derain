import os
import subprocess
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch
from preprocessor.freq_filter import freq_highpass_filter
from preprocessor.hilbert_filter import hilbert_transform
from preprocessor.shallow_cnn import ShallowCNN
from classification.classification import Classifier
import cv2

def classify_rain_image(image_path, classifier):
    # 用OpenCV读取图像以兼容分类器
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image at {image_path}")
    result = classifier.test(classifier.model_path, img)
    return result[0]  # 返回类别名字符串，如 'no rain'

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

def main():
    input_dir = "/home/workspace/ryc/code/NLCL/datasets/SPA-Testing/Real_Internet"
    output_dir = "/home/workspace/ryc/code/NLCL/data_temp"  # NLCL会从这里读取
    model_name = "NLCL"
    exp_name = "RealRain"
    mode = "freq"  # 可选 "freq" 或 "hilbert"

    classifier = Classifier(
        window_size=(10,10),
        whether_client=False,
        workers=8,
        verbose=False,
        grid_search=False,
        model_path='./classification/svm_model-4-grid_searched-200x200.joblib'
    )

    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"))])

    print(f"[INFO] all {len(all_files)} pictures,start batch preprocessing...")

    count_all = 0
    count_preprocess = 0

    for fname in all_files:
        image_path = os.path.join(input_dir, fname)
        save_path = os.path.join(output_dir, fname)
        count_all += 1

        rain_class = classify_rain_image(image_path, classifier)
        print(f"[{fname}] classification results: {rain_class}")

        if rain_class in ["heavy"]:
            print(f"[{fname}] belongs to {rain_class},start preprocessing ({mode})...")
            tensor = preprocess_image(image_path, mode=mode)
            save_tensor_as_image(tensor, save_path)
            count_preprocess += 1
        else:
            print(f"[{fname}] belongs to {rain_class},skip preprocessing, directly copy...")
            os.system(f"cp '{image_path}' '{save_path}'")

    print(f"[INFO] finish processing, totally {count_all} pictures,  {count_preprocess} of which are preprocessed.")

    # 只执行一次NLCL测试命令
    print("[INFO] NLCL test bash...")
    cmd = [
        "python", "test.py",
        "--dataroot", output_dir,
        "--model", model_name,
        "--name", exp_name,
        "--dataset_mode", "single",
        "--preprocess", "None",
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
