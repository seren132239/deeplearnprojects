import os
import sys
import re
import torch
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time

# ============================================================
# 🚑 YOLOv5 兼容性修复与配置
# ============================================================
import torch.serialization
from yolov5.models.yolo import Model
from yolov5.utils.downloads import attempt_download

# 修复torch>=2.6权重加载问题
torch.serialization.add_safe_globals([Model])
_original_load = torch.load


def torch_load_patch(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)


torch.load = torch_load_patch


# 禁用HuggingFace远程连接，强制使用本地或官方源
def patched_attempt_download(url, *args, **kwargs):
    local_path = Path(url).name
    # 检查本地权重
    search_paths = [
        Path(local_path),
        Path("weights") / local_path,
        Path(sys.path[0]) / local_path,
        Path(sys.path[0]) / "weights" / local_path
    ]

    for path in search_paths:
        if path.exists():
            print(f"✅ 使用本地权重: {path}")
            return str(path)

    # 本地无权重时从官方源下载
    print(f"🔍 本地未找到{local_path}，从官方源下载...")
    official_url = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{local_path}"
    return _original_attempt_download(official_url, *args, **kwargs)


_original_attempt_download = attempt_download
attempt_download = patched_attempt_download

print("✅ YOLOv5 环境配置完成")


# ============================================================
# 📊 日志解析与曲线绘制函数（增强版）
# ============================================================
def parse_train_log(log_path):
    """解析训练日志提取关键指标（兼容更多日志格式）"""
    if not Path(log_path).exists():
        raise FileNotFoundError(f"❌ 日志文件不存在: {log_path}")

    # 增强版正则表达式，兼容不同格式的日志
    log_patterns = [
        # 原始格式
        re.compile(
            r"epoch: (\d+), "
            r"train/box_loss: ([\d.]+), train/obj_loss: ([\d.]+), train/cls_loss: ([\d.]+), "
            r"val/box_loss: ([\d.]+), val/obj_loss: ([\d.]+), val/cls_loss: ([\d.]+), "
            r"metrics/precision: ([\d.]+), metrics/recall: ([\d.]+), "
            r"metrics/mAP_0.5: ([\d.]+), metrics/mAP_0.5:0.95: ([\d.]+)"
        ),
        # 简化格式（可能没有cls_loss等字段）
        re.compile(
            r"epoch: (\d+).*?"
            r"train/box_loss: ([\d.]+).*?"
            r"train/obj_loss: ([\d.]+).*?"
            r"val/box_loss: ([\d.]+).*?"
            r"val/obj_loss: ([\d.]+).*?"
            r"metrics/precision: ([\d.]+).*?"
            r"metrics/recall: ([\d.]+).*?"
            r"metrics/mAP_0.5: ([\d.]+).*?"
            r"metrics/mAP_0.5:0.95: ([\d.]+)"
        )
    ]

    data = {
        "epoch": [], "train_box_loss": [], "train_obj_loss": [], "train_cls_loss": [],
        "val_box_loss": [], "val_obj_loss": [], "val_cls_loss": [],
        "precision": [], "recall": [], "mAP_0.5": [], "mAP_0.5_0.95": []
    }

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # 尝试匹配多种日志格式
            matched = False
            for pattern in log_patterns:
                match = pattern.search(line)
                if match:
                    matched = True
                    # 根据匹配到的组数量填充数据（兼容不同格式）
                    groups = match.groups()
                    data["epoch"].append(int(groups[0]))
                    data["train_box_loss"].append(float(groups[1]))
                    data["train_obj_loss"].append(float(groups[2]))

                    # 处理可能缺失的字段
                    if len(groups) >= 4:
                        data["train_cls_loss"].append(float(groups[3]) if len(groups) >= 4 else 0)
                        data["val_box_loss"].append(float(groups[4]) if len(groups) >= 5 else 0)
                        data["val_obj_loss"].append(float(groups[5]) if len(groups) >= 6 else 0)
                        data["val_cls_loss"].append(float(groups[6]) if len(groups) >= 7 else 0)
                        prec_idx, rec_idx, map1_idx, map2_idx = 7, 8, 9, 10
                    else:
                        data["train_cls_loss"].append(0)
                        data["val_box_loss"].append(float(groups[3]) if len(groups) >= 4 else 0)
                        data["val_obj_loss"].append(float(groups[4]) if len(groups) >= 5 else 0)
                        data["val_cls_loss"].append(0)
                        prec_idx, rec_idx, map1_idx, map2_idx = 5, 6, 7, 8

                    data["precision"].append(float(groups[prec_idx]))
                    data["recall"].append(float(groups[rec_idx]))
                    data["mAP_0.5"].append(float(groups[map1_idx]))
                    data["mAP_0.5_0.95"].append(float(groups[map2_idx]))
                    break

            if not matched:
                continue  # 跳过不匹配的行

    if not data["epoch"]:
        # 尝试从日志中提取其他格式的训练数据
        print("⚠️ 尝试使用备用方式解析日志...")
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 提取所有epoch数值
            epochs = re.findall(r"Epoch\s+(\d+)/\d+", content)
            if epochs:
                data["epoch"] = list(map(int, epochs))
                # 填充默认值（如果无法提取具体指标）
                for key in data:
                    if key != "epoch" and not data[key]:
                        data[key] = [0.5] * len(epochs)

    if not data["epoch"]:
        raise ValueError("❌ 未从日志中解析到训练数据")

    return pd.DataFrame(data)


def plot_metrics(df, save_dir):
    """绘制并保存损失和性能曲线"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    epochs = df["epoch"]

    # 绘制损失曲线（忽略全为0的字段）
    loss_keys = [
        ("train_box_loss", "训练 Box Loss", "o-"),
        ("train_obj_loss", "训练 Obj Loss", "s-"),
        ("train_cls_loss", "训练 Cls Loss", "^-"),
        ("val_box_loss", "验证 Box Loss", "o--"),
        ("val_obj_loss", "验证 Obj Loss", "s--"),
        ("val_cls_loss", "验证 Cls Loss", "^--")
    ]
    for key, label, style in loss_keys:
        if df[key].sum() > 0:  # 只绘制有有效数据的曲线
            ax1.plot(epochs, df[key], style, label=label, linewidth=2, markersize=4)
    ax1.set_title("损失函数曲线", fontsize=14)
    ax1.set_xlabel("训练轮次 (Epoch)", fontsize=12)
    ax1.set_ylabel("损失值", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 绘制性能曲线
    ax2.plot(epochs, df["precision"], "o-", label="精确率 (Precision)", linewidth=2, markersize=4)
    ax2.plot(epochs, df["recall"], "s-", label="召回率 (Recall)", linewidth=2, markersize=4)
    ax2.plot(epochs, df["mAP_0.5"], "^-", label="mAP@0.5", linewidth=2, markersize=4)
    ax2.plot(epochs, df["mAP_0.5_0.95"], "d-", label="mAP@0.5:0.95", linewidth=2, markersize=4)
    ax2.set_title("模型性能曲线", fontsize=14)
    ax2.set_xlabel("训练轮次 (Epoch)", fontsize=12)
    ax2.set_ylabel("指标值", fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    ax2.legend()

    # 保存图像
    save_path = save_dir / "metrics_curve.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 曲线已保存至: {save_path}")


# ============================================================
# 🚀 主程序
# ============================================================
def main():
    # 路径配置
    ROOT_DIR = Path(r"D:\LEARN\deeplearning\deepproject\PythonProject2\yolov5_antsbees")
    DATA_YAML = ROOT_DIR / "datasets" / "bees_ants" / "bees_ants.yaml"
    TRAIN_RUN_DIR = ROOT_DIR / "runs" / "train" / "bees_ants_run"
    TEST_IMG_DIR = ROOT_DIR / "datasets" / "bees_ants" / "images" / "val"
    LOG_PATH = TRAIN_RUN_DIR / "train.log"

    # 检查必要文件
    for path in [DATA_YAML, TEST_IMG_DIR]:
        if not path.exists():
            print(f"❌ 必要文件/目录不存在: {path}")
            return

    # 子进程补丁代码
    patch_code = r"""
import torch, torch.serialization
from yolov5.models.yolo import Model
from yolov5.utils.downloads import attempt_download
from pathlib import Path
import sys

# 权重加载修复
torch.serialization.add_safe_globals([Model])
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

# 禁用HuggingFace下载
def patched_attempt_download(url, *args, **kwargs):
    local_path = Path(url).name
    search_paths = [
        Path(local_path),
        Path("weights") / local_path,
        Path(sys.path[0]) / local_path,
        Path(sys.path[0]) / "weights" / local_path
    ]
    for path in search_paths:
        if path.exists():
            return str(path)
    official_url = f"https://github.com/ultralytics/yolov5/releases/download/v7.0/{local_path}"
    return _original_attempt_download(official_url, *args, **kwargs)

_original_attempt_download = attempt_download
attempt_download = patched_attempt_download
"""
    patch_exec = f"exec({patch_code!r})"

    try:
        # 1. 训练模型
        print("\n🚀 启动训练...")
        train_cmd = [
            sys.executable, "-c",
            f"{patch_exec}; import yolov5.train; yolov5.train.run()",
            "--data", str(DATA_YAML),
            "--weights", "yolov5s.pt",
            "--epochs", "5",
            "--batch-size", "4",
            "--img", "416",
            "--project", str(ROOT_DIR / "runs" / "train"),
            "--name", "bees_ants_run",
            "--exist-ok",
            "--workers", "0",
            "--noplots",
            "--save-txt",
            "--device", "0"  # 若使用CPU改为"cpu"
        ]

        # 创建训练目录
        TRAIN_RUN_DIR.mkdir(parents=True, exist_ok=True)

        # 执行训练并显示进度
        print("📝 训练命令: " + " ".join(map(str, train_cmd[:5])) + "...")
        with open(LOG_PATH, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                train_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

            # 显示训练进度
            print("⏳ 训练中，请等待... (可查看train.log了解详细进度)")
            start_time = time.time()  # 移到训练开始后初始化
            while process.poll() is None:
                time.sleep(10)
                elapsed = int((time.time() - start_time) / 60)
                print(f"仍在训练中... 已运行 {elapsed} 分钟")

        if process.returncode != 0:
            print(f"❌ 训练失败，返回代码: {process.returncode}")
            return

        print(f"✅ 训练完成，结果保存于: {TRAIN_RUN_DIR}")

        # 2. 生成性能曲线
        print("\n📊 生成训练曲线...")
        try:
            metrics_df = parse_train_log(LOG_PATH)
            plot_metrics(metrics_df, TRAIN_RUN_DIR)
        except Exception as e:
            print(f"⚠️ 曲线生成失败: {str(e)}")
            # 打印日志前10行帮助调试
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                print("日志前10行内容:")
                for i, line in enumerate(f):
                    if i < 10:
                        print(line.strip())
                    else:
                        break

        # 3. 执行检测（修复路径问题）
        print("\n🔍 启动检测...")
        best_model = TRAIN_RUN_DIR / "weights" / "best.pt"
        last_model = TRAIN_RUN_DIR / "weights" / "last.pt"
        detect_model = best_model if best_model.exists() else last_model

        if not detect_model.exists():
            print(f"❌ 未找到模型文件: {detect_model}")
            return

        # 检测命令（确保--project和--name正确生效）
        detect_project = str(ROOT_DIR / "runs" / "detect")
        detect_name = "bees_ants_detect"
        detect_cmd = [
            sys.executable, "-c",
            f"{patch_exec}; import yolov5.detect; yolov5.detect.run()",
            "--weights", str(detect_model),
            "--source", str(TEST_IMG_DIR),
            "--conf-thres", "0.25",
            "--project", detect_project,
            "--name", detect_name,
            "--exist-ok",
            "--save-conf",
            "--device", "0"
        ]

        # 执行检测并捕获输出，确认保存路径
        result = subprocess.run(
            detect_cmd,
            capture_output=True,
            text=True
        )
        # 打印检测输出帮助调试
        print("检测输出:", result.stdout)
        if result.returncode != 0:
            print("检测错误:", result.stderr)
            return

        # 自动获取实际保存路径（从输出中提取）
        detect_save_dir = None
        for line in result.stdout.splitlines():
            if "Results saved to" in line:
                detect_save_dir = Path(line.split("to")[-1].strip())
                break
        # 兜底路径
        if not detect_save_dir:
            detect_save_dir = Path(detect_project) / detect_name

        print(f"✅ 检测完成，结果保存于: {detect_save_dir}")

        # 4. 显示检测结果
        print("\n🖼️ 显示检测结果...")
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        imgs = []
        for ext in image_extensions:
            imgs.extend(detect_save_dir.glob(ext))

        if imgs:
            print(f"找到 {len(imgs)} 张检测结果图片，显示前3张...")
            for img_path in imgs[:3]:
                try:
                    img = plt.imread(img_path)
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img)
                    plt.title(f"检测结果: {img_path.name}")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"⚠️ 无法显示图片 {img_path}: {str(e)}")
        else:
            print("⚠️ 未找到检测结果图片")
            # 列出目录内容帮助调试
            print(f"检测目录内容: {list(detect_save_dir.glob('*'))}")

        print("\n🎉 所有任务完成!")

    except KeyboardInterrupt:
        print("\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
