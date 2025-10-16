import os
import random
import shutil
from tqdm import tqdm
import zipfile
import requests
from PIL import Image
from pathlib import Path


# ============================================================
# 数据准备脚本：下载、划分并生成 YOLOv5 兼容的模拟标注
# ============================================================

def run_data_preparation(base_dir):
    BASE_DIR = Path(base_dir)
    DATA_DIR = BASE_DIR / "datasets" / "bees_ants"
    IMG_DIR = DATA_DIR / "images"
    LBL_DIR = DATA_DIR / "labels"

    # 创建必要的顶级目录
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. 下载官方数据集
    # ============================================================
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    dataset_zip = DATA_DIR / "dataset.zip"
    hymenoptera_data_path = DATA_DIR / "hymenoptera_data"

    if not hymenoptera_data_path.exists():
        print("📥 正在下载 hymenoptera_data 数据集...")
        # 使用 shutil 方便地下载文件
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(dataset_zip, "wb") as f, tqdm(
                    desc=str(dataset_zip.name),
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

        with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(dataset_zip)
        print("✅ 数据集下载并解压完成。")
    else:
        print("📦 已检测到数据集，跳过下载。")

    # ============================================================
    # 2. 生成 YOLOv5 数据格式 (划分 train/val, 生成模拟标注)
    # ============================================================
    bee_dir = hymenoptera_data_path / "train" / "bees"
    ant_dir = hymenoptera_data_path / "train" / "ants"

    # 清理旧的 images/labels 划分，确保重新运行时的清洁性
    if IMG_DIR.exists():
        shutil.rmtree(IMG_DIR)
        IMG_DIR.mkdir()
    if LBL_DIR.exists():
        shutil.rmtree(LBL_DIR)
        LBL_DIR.mkdir()

    def prepare_class(img_dir: Path, class_id: int, cls_name: str, split_ratio: float = 0.8):
        imgs = [f for f in img_dir.iterdir() if f.suffix.lower() == ".jpg"]
        random.shuffle(imgs)
        split_index = int(split_ratio * len(imgs))

        for i, src_path in enumerate(tqdm(imgs, desc=f"Processing {cls_name}")):
            subset = "train" if i < split_index else "val"
            dst_img_dir = IMG_DIR / subset
            dst_lbl_dir = LBL_DIR / subset

            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)

            dst_path = dst_img_dir / src_path.name
            shutil.copy(src_path, dst_path)

            # 模拟生成中心框标注（YOLO格式）
            # 警告：这是模拟标注，训练效果会很差，仅用于项目流程测试
            x_center, y_center, bw, bh = 0.5, 0.5, 0.8, 0.8
            label_file = dst_lbl_dir / src_path.name.replace(".jpg", ".txt")
            with open(label_file, "w") as f:
                f.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")

    print("\n📐 正在划分数据集并生成模拟标注...")
    prepare_class(bee_dir, 0, "bee")
    prepare_class(ant_dir, 1, "ant")
    print("✅ 数据划分和模拟标注完成。")

    # ============================================================
    # 3. 写入 YAML 配置文件
    # ============================================================
    yaml_path = DATA_DIR / "bees_ants.yaml"
    yaml_content = f"""
# YOLOv5 配置文件
train: {IMG_DIR.relative_to(DATA_DIR)}/train
val: {IMG_DIR.relative_to(DATA_DIR)}/val

# 类别数
nc: 2

# 类别名称
names: ['bee', 'ant']
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"✅ bees_ants.yaml 已生成：{yaml_path}")
    print(f"🎉 数据准备流程结束！数据位于 {DATA_DIR}")
    return str(DATA_DIR), str(hymenoptera_data_path)


if __name__ == '__main__':
    # 示例调用
    BASE_DIR = r"D:\LEARN\deeplearning\deepproject\PythonProject2\yolov5_antsbees"
    run_data_preparation(BASE_DIR)