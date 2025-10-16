import os
import sys
import torch
import subprocess
import matplotlib.pyplot as plt
import multiprocessing
import shutil
import random
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path


# ============================================================
# 1. 数据准备函数 (包含 YAML 绝对路径修复)
# ============================================================

# 辅助函数：将路径转换为适合命令行参数和 YAML 的格式 (正斜杠)
def format_path_for_cli(path_str):
    """将 Windows 反斜杠路径转换为正斜杠，用于 CLI 和 YAML"""
    return str(Path(path_str)).replace('\\', '/')


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
    # 1.1 下载官方数据集
    # ============================================================
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
    dataset_zip = DATA_DIR / "dataset.zip"
    hymenoptera_data_path = DATA_DIR / "hymenoptera_data"

    if not hymenoptera_data_path.exists():
        print("📥 正在下载 hymenoptera_data 数据集...")
        try:
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
        except Exception as e:
            print(f"❌ 下载或解压数据集失败: {e}")
            sys.exit(1)
    else:
        print("📦 已检测到数据集，跳过下载。")

    # ============================================================
    # 1.2 生成 YOLOv5 数据格式 (划分 train/val, 生成模拟标注)
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
        if not img_dir.exists():
            print(f"❌ 警告：分类目录 {img_dir} 不存在，跳过。")
            return

        imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")]
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
            x_center, y_center, bw, bh = 0.5, 0.5, 0.8, 0.8
            label_file = dst_lbl_dir / src_path.name.replace(src_path.suffix, ".txt")
            with open(label_file, "w") as f:
                f.write(f"{class_id} {x_center} {y_center} {bw} {bh}\n")

    print("\n📐 正在划分数据集并生成模拟标注...")
    prepare_class(bee_dir, 0, "bee")
    prepare_class(ant_dir, 1, "ant")
    print("✅ 数据划分和模拟标注完成。")

    # ============================================================
    # 1.3 写入 YAML 配置文件 (关键修复：使用绝对路径)
    # ============================================================
    yaml_path = DATA_DIR / "bees_ants.yaml"

    # 使用绝对路径和正斜杠
    train_abs_path = format_path_for_cli(str(IMG_DIR / "train"))
    val_abs_path = format_path_for_cli(str(IMG_DIR / "val"))

    yaml_content = f"""
# YOLOv5 配置文件 (使用绝对路径)
train: {train_abs_path}
val: {val_abs_path}

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


# ============================================================
# 2. YOLOv5 权重加载兼容修复
# ============================================================
try:
    from yolov5.models.yolo import Model

    torch.serialization.add_safe_globals([Model])

    _original_load = torch.load


    def torch_load_patch(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)


    torch.load = torch_load_patch
    print("✅ YOLOv5 权重加载兼容修复完成")

except ImportError:
    print("⚠️ 警告：yolov5 模块未找到。请确保已安装 (pip install yolov5)。")
    sys.exit(1)


# ============================================================
# 3. 改进的绘图函数
# ============================================================
def plot_training_results(results_path):
    """绘制训练损失和准确率曲线（改进版）"""
    if not results_path.exists():
        print(f"⚠️ 未找到训练结果文件：{results_path}")
        return

    print("\n📈 正在绘制训练结果曲线...")
    try:
        # 读取CSV，处理可能的空格和注释
        df = pd.read_csv(results_path, skipinitialspace=True)

        # 清理列名（去除空格）
        df.columns = df.columns.str.strip()

        # 打印可用的列名，方便调试
        print(f"📋 可用列名: {list(df.columns)}")

        # 确保有epoch列
        if 'epoch' not in df.columns:
            df['epoch'] = range(len(df))

        epochs = df['epoch']

        # 创建2x2子图布局
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('YOLOv5 训练结果分析', fontsize=16, fontweight='bold')

        # ============ 子图1：训练和验证损失 ============
        ax1 = axes[0, 0]

        # 找到损失相关的列（兼容不同版本的YOLOv5）
        loss_cols = {
            'train_box': None,
            'train_obj': None,
            'train_cls': None,
            'val_box': None,
            'val_obj': None
        }

        # 尝试匹配列名
        for col in df.columns:
            col_lower = col.lower()
            if 'train' in col_lower and 'box' in col_lower:
                loss_cols['train_box'] = col
            elif 'train' in col_lower and 'obj' in col_lower:
                loss_cols['train_obj'] = col
            elif 'train' in col_lower and 'cls' in col_lower:
                loss_cols['train_cls'] = col
            elif 'val' in col_lower and 'box' in col_lower:
                loss_cols['val_box'] = col
            elif 'val' in col_lower and 'obj' in col_lower:
                loss_cols['val_obj'] = col

        # 绘制损失曲线
        if loss_cols['train_box']:
            ax1.plot(epochs, df[loss_cols['train_box']], 'b-', label='训练Box Loss', linewidth=2)
        if loss_cols['val_box']:
            ax1.plot(epochs, df[loss_cols['val_box']], 'b--', label='验证Box Loss', linewidth=2)
        if loss_cols['train_obj']:
            ax1.plot(epochs, df[loss_cols['train_obj']], 'r-', label='训练Obj Loss', linewidth=2)
        if loss_cols['val_obj']:
            ax1.plot(epochs, df[loss_cols['val_obj']], 'r--', label='验证Obj Loss', linewidth=2)

        ax1.set_title('损失函数曲线', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss Value', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # ============ 子图2：mAP指标 ============
        ax2 = axes[0, 1]

        # 找到mAP相关列
        map_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'map' in col_lower:
                if '0.5:0.95' in col_lower or 'map_0.5:0.95' in col_lower:
                    map_cols['map50_95'] = col
                elif '0.5' in col_lower and '0.95' not in col_lower:
                    map_cols['map50'] = col

        if map_cols.get('map50'):
            ax2.plot(epochs, df[map_cols['map50']], 'g-', label='mAP@0.5',
                     marker='o', markersize=4, linewidth=2)
        if map_cols.get('map50_95'):
            ax2.plot(epochs, df[map_cols['map50_95']], 'm-', label='mAP@0.5:0.95',
                     marker='s', markersize=4, linewidth=2)

        ax2.set_title('平均精度(mAP)曲线', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('mAP', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # ============ 子图3：Precision和Recall ============
        ax3 = axes[1, 0]

        # 找到precision和recall列
        perf_cols = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'precision' in col_lower:
                perf_cols['precision'] = col
            elif 'recall' in col_lower:
                perf_cols['recall'] = col

        if perf_cols.get('precision'):
            ax3.plot(epochs, df[perf_cols['precision']], 'b-', label='Precision',
                     marker='o', markersize=4, linewidth=2)
        if perf_cols.get('recall'):
            ax3.plot(epochs, df[perf_cols['recall']], 'r-', label='Recall',
                     marker='s', markersize=4, linewidth=2)

        # 计算F1-score（如果有precision和recall）
        if perf_cols.get('precision') and perf_cols.get('recall'):
            precision = df[perf_cols['precision']]
            recall = df[perf_cols['recall']]
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            ax3.plot(epochs, f1, 'g-', label='F1-Score',
                     marker='^', markersize=4, linewidth=2)

        ax3.set_title('精确率与召回率', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])

        # ============ 子图4：综合损失 ============
        ax4 = axes[1, 1]

        # 计算总损失（如果可能）
        train_total = None
        val_total = None

        if all([loss_cols['train_box'], loss_cols['train_obj']]):
            train_total = df[loss_cols['train_box']] + df[loss_cols['train_obj']]
            if loss_cols['train_cls']:
                train_total += df[loss_cols['train_cls']]
            ax4.plot(epochs, train_total, 'b-', label='训练总损失', linewidth=2.5)

        if all([loss_cols['val_box'], loss_cols['val_obj']]):
            val_total = df[loss_cols['val_box']] + df[loss_cols['val_obj']]
            ax4.plot(epochs, val_total, 'r--', label='验证总损失', linewidth=2.5)

        ax4.set_title('总损失趋势', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Total Loss', fontsize=12)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        save_path = results_path.parent / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练曲线已保存至: {save_path}")

        plt.show()

        # 打印最终结果摘要
        print("\n📊 训练结果摘要:")
        print(f"   最终Epoch: {epochs.iloc[-1]}")
        if map_cols.get('map50'):
            print(f"   最佳mAP@0.5: {df[map_cols['map50']].max():.4f}")
        if perf_cols.get('precision'):
            print(f"   最终Precision: {df[perf_cols['precision']].iloc[-1]:.4f}")
        if perf_cols.get('recall'):
            print(f"   最终Recall: {df[perf_cols['recall']].iloc[-1]:.4f}")

    except Exception as e:
        print(f"⚠️ 绘图时出错: {e}")
        print(f"   请检查文件: {results_path}")
        import traceback
        traceback.print_exc()


# ============================================================
# 4. 主程序
# ============================================================
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # ❗❗❗ 项目根目录配置 ❗❗❗
    ROOT_DIR = r"D:\LEARN\deeplearning\deepproject\PythonProject2\yolov5_antsbees"

    # ============================================================
    # 4.1 运行数据准备
    # ============================================================
    DATA_ROOT_DIR, RAW_DATA_DIR = run_data_preparation(ROOT_DIR)

    DATA_YAML = os.path.join(DATA_ROOT_DIR, "bees_ants.yaml")
    TRAIN_RUN_DIR = os.path.join(ROOT_DIR, "runs", "train", "bees_ants_run")
    TEST_IMG_DIR = os.path.join(DATA_ROOT_DIR, "images", "val")

    # 检查必要文件
    assert os.path.exists(DATA_YAML), f"❌ 数据集配置不存在：{DATA_YAML}"
    assert os.path.exists(TEST_IMG_DIR), f"❌ 验证集图片目录不存在：{TEST_IMG_DIR}"

    # ============================================================
    # 4.2 子进程补丁（用于 train/detect 运行时的模型加载）
    # ============================================================
    patch_code = r"""
import torch, torch.serialization
from yolov5.models.yolo import Model
torch.serialization.add_safe_globals([Model])
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load
"""
    patch_exec = f"exec({patch_code!r})"

    # 转换为命令行安全路径格式
    cli_data_yaml = format_path_for_cli(DATA_YAML)
    cli_root_dir = format_path_for_cli(ROOT_DIR)

    # ============================================================
    # 4.3 启动训练
    # ============================================================
    print("\n🚀 正在启动 YOLOv5 训练（蜜蜂🐝 vs 蚂蚁🐜）...")

    num_workers = os.cpu_count() or 4

    train_cmd = [
        sys.executable, "-c",
        f"{patch_exec}; import yolov5.train; yolov5.train.run()",
        "--data", cli_data_yaml,
        "--weights", "yolov5s.pt",
        "--epochs", "10",
        "--batch-size", "8",
        "--img", "640",
        "--project", os.path.join(cli_root_dir, "runs", "train"),
        "--name", "bees_ants_run",
        "--exist-ok",
        "--workers", str(num_workers),
        "--noplots"
    ]

    try:
        subprocess.run(train_cmd, check=True)
        print(f"✅ 训练完成！结果保存于：{TRAIN_RUN_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败！请检查错误信息：\n{e.stderr}")
        sys.exit(1)

    # ============================================================
    # 4.4 启动检测
    # ============================================================
    print("\n🔍 正在启动测试集检测...")
    best_model = os.path.join(TRAIN_RUN_DIR, "weights", "best.pt")
    detect_model = best_model if os.path.exists(best_model) else os.path.join(TRAIN_RUN_DIR, "weights", "last.pt")
    assert os.path.exists(detect_model), f"❌ 未找到模型文件：{detect_model}"

    cli_detect_model = format_path_for_cli(detect_model)
    cli_test_img_dir = format_path_for_cli(TEST_IMG_DIR)

    detect_project = os.path.join(ROOT_DIR, "runs", "detect")
    detect_name = "bees_ants_detect"
    detect_save_dir = os.path.join(detect_project, detect_name)
    cli_detect_project = format_path_for_cli(detect_project)

    detect_cmd = [
        sys.executable, "-c",
        f"{patch_exec}; import yolov5.detect; yolov5.detect.run("
        f"weights='{cli_detect_model}', "
        f"source='{cli_test_img_dir}/*',"
        f"conf_thres=0.25, "
        f"project='{cli_detect_project}', "
        f"name='{detect_name}', "
        f"exist_ok=True, "
        f"save_conf=True"
        f")"
    ]

    try:
        result = subprocess.run(
            detect_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            check=True
        )
        print("检测输出：", result.stdout)
    except subprocess.CalledProcessError as e:
        print("检测错误：", e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"检测过程出错：{str(e)}")
        sys.exit(1)

    print(f"✅ 检测完成！结果保存于：{detect_save_dir}")

    # ============================================================
    # 4.5 显示检测结果 (前3张图)
    # ============================================================
    imgs = []
    detect_save_path = Path(detect_save_dir)
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        imgs.extend(detect_save_path.glob(ext))

    if imgs:
        print(f"\n🖼️ 自动显示检测结果：{len(imgs)} 张图片")
        for img_path in imgs[:3]:
            try:
                img = plt.imread(img_path)
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.title(f"检测结果: {img_path.name}")
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"⚠️ 无法显示图片 {img_path}：{e}")
    else:
        print(f"⚠️ 未找到检测输出图片，路径：{detect_save_dir}")

    # ============================================================
    # 4.6 使用改进的绘图函数绘制训练曲线
    # ============================================================
    results_path = Path(TRAIN_RUN_DIR) / "results.csv"
    plot_training_results(results_path)

    # ============================================================
    # 4.7 清理和结束
    # ============================================================
    if os.path.exists(RAW_DATA_DIR):
        shutil.rmtree(RAW_DATA_DIR)
        print(f"\n🧹 已清理原始解压目录：{RAW_DATA_DIR}")

    print("\n🎉 全流程已完成！")