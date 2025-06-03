import os

# 设置目标文件夹路径
folder_path = "/home/workspace/ryc/code/NLCL/datasets/GT-RAIN/train/trainB"
count=0
# 遍历该文件夹下的所有文件
for filename in os.listdir(folder_path):
    # 检查文件名中是否包含 "-C-"
    if "-C-" in filename:
        count+=1
        # 替换为 "-R-"
        new_filename = filename.replace("-C-", "-R-")
        # 构造完整路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")
print(f"{count}")

