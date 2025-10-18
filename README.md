# deeplearnprojects
记录深度学习
# Bees vs Ants YOLOv5 目标检测项目

项目简介
本项目基于YOLOv5框架实现蜜蜂与蚂蚁的二分类目标检测，包含全自动数据准备、模型训练、测试检测、结果可视化全流程。代码内置路径兼容、权重加载修复等功能，可直接在Windows/macOS/Linux系统运行，无需手动处理数据集格式或配置文件。



运行环境要求
 基础环境
- Python版本：3.8 - 3.10（推荐3.9，兼容性最佳）
- 操作系统：Windows 10+/macOS 12+/Linux（Ubuntu 20.04+）
-硬件建议
  - CPU：4核及以上（训练较慢，适合测试）
  - GPU：NVIDIA显卡（支持CUDA 11.3+，训练效率提升10-20倍）


第三方依赖库
通过`pip`安装以下依赖（版本为最低兼容版，推荐使用最新稳定版）：
```bash
 基础依赖（必装）
pip install torch>=1.7.0 torchvision>=0.8.1 yolov5>=6.0
数据处理与可视化依赖
pip install matplotlib>=3.3.0 pandas>=1.1.0 tqdm>=4.50.0
 辅助工具依赖
pip install requests>=2.24.0 Pillow>=8.0.0 numpy>=1.19.0
```

> 说明：PyTorch GPU版本需根据显卡配置安装，参考[PyTorch官方指南](https://pytorch.org/get-started/locally/)，例如CUDA 11.8版本安装命令：
> ```bash
> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```


快速开始
 1. 项目目录配置
1. 下载项目文件`train_and_detect.py`，保存到任意目录（如`D:\yolov5_bees_ants`）。
2. 打开`train_and_detect.py`，修改项目根目录（第340行）为实际保存路径：
   ```python
   # ❗ 必须修改为你的项目实际路径
   ROOT_DIR = r"D:\yolov5_bees_ants"  # Windows示例
   # ROOT_DIR = "/home/user/yolov5_bees_ants"  # Linux/macOS示例
   ```


 2. 运行项目
直接执行Python脚本，全流程自动运行（无需手动干预）：
```bash
python train_and_detect.py
```

 3. 运行流程说明
脚本运行时会依次执行以下步骤，控制台会显示进度和状态（带emoji标识）：
1. 📥 下载`hymenoptera_data`数据集（约100MB，已存在则跳过）。
2. 📐 划分训练/验证集，生成模拟标注文件（YOLO格式）。
3. ✅ 生成`bees_ants.yaml`配置文件（含绝对路径，避免路径错误）。
4. 🚀 启动YOLOv5训练（10个epoch，batch size=8，输入图像尺寸640x640）。
5. 🔍 用训练好的最优模型（`best.pt`）检测验证集图像。
6. 🖼️ 自动显示前3张检测结果图。
7. 📈 绘制训练曲线并保存为`training_curves.png`。
8. 🧹 清理原始数据集解压目录，输出全流程完成提示。


结果输出位置
所有运行结果会保存在项目根目录的`runs`文件夹下，结构如下：
```
yolov5_bees_ants/
└── runs/
    ├── train/                # 训练结果目录
    │   └── bees_ants_run/    # 本次训练文件夹
    │       ├── weights/      # 模型权重（best.pt/last.pt）
    │       ├── results.csv   # 训练日志（损失、mAP等数据）
    │       └── training_curves.png  # 训练曲线图表
    └── detect/               # 检测结果目录
        └── bees_ants_detect/ # 本次检测文件夹
            └── 带标注的检测图像  # 每张图含类别、置信度和 bounding box
```


关键参数调整（可选）
若需优化训练效果或适配硬件，可修改`train_and_detect.py`中的以下参数（第368-375行）：
| 参数          | 说明                          | 建议调整方向                  |
|---------------|-------------------------------|-----------------------------|
| `--epochs`    | 训练轮次                      | 显存不足减至5-8，追求精度增至15-20 |
| `--batch-size`| 批次大小                      | GPU显存不足时减至4（最小2），充足时增至16 |
| `--img`       | 输入图像尺寸                  | 显存不足减至416，追求精度增至800 |
| `--weights`   | 预训练权重                    | 可改为`yolov5m.pt`（中模型，精度更高）或`yolov5n.pt`（ nano模型，速度更快） |


常见问题解决
 1. 数据集下载失败
- 问题：控制台显示“❌ 下载或解压数据集失败”。
- 解决：手动下载数据集[hymenoptera_data.zip](https://download.pytorch.org/tutorial/hymenoptera_data.zip)，解压到`项目根目录/datasets/bees_ants/`下，确保解压后有`hymenoptera_data`文件夹，重新运行脚本。

 2. YOLOv5模块导入错误
- 问题：控制台显示“⚠️ 警告：yolov5 模块未找到”。
- 解决：执行`pip uninstall yolov5`后，重新安装最新版：`pip install yolov5`。

 3. GPU训练报错（CUDA out of memory）
- 问题：显存不足，训练中断。
- 解决：
  1. 减小`--batch-size`（如从8改为4或2）。
  2. 减小`--img`尺寸（如从640改为416）。
  3. 改用更小的预训练权重（如`yolov5n.pt`）。


