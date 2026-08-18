# MIE-YOLO 复现指南（线上服务器版）

本文档用于在**线上服务器**复现《MIE-YOLO: A Multi-Scale Information-Enhanced Weed Detection Algorithm for Precision Agriculture》的模型架构（MS-EIS / Add-CGLU / DEC 三个创新点）。

**目标服务器环境（固定，不可更改）**
- 操作系统：Ubuntu（Linux x86_64）
- Python：**3.12.3**（无法更换为 3.11）
- GPU 驱动：CUDA **13.0**（向下兼容 cu118 / cu121 / cu124 等低版本 CUDA 运行时）
- 显卡：NVIDIA（建议显存 ≥ 8GB）

> 说明：论文第四个创新点 **DS（Double Self-Knowledge Distillation）双重自蒸馏** 在当前仓库**无任何代码实现**，本文档不含其复现步骤，仅记录已知信息（见第 7 节）。

---

## 1. 复现条件与版本对应关系

**关键约束：Python 3.12 无法使用 torch 2.2.2**（该版本无 cp312 wheel）。torch 自 2.3.0 起才支持 Python 3.12，且论文要求 `torch >= 2.2.2`，因此：

| 软件 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.3 | 服务器固定，直接在 base 主环境安装 |
| torch | **2.4.1+cu121** | py3.12 兼容的稳定版，满足 `>=2.2.2`，CUDA 12.1 运行时 |
| torchvision | **0.19.1+cu121** | ⚠️ 必须与 torch 2.4.1 **配套**（0.17.2 只配 torch 2.2.x，不可混用） |
| ultralytics | **8.3.253** | ⚠️ 必须 8.3.x：≥8.4.0 重构了检测损失（dict 输出），与本仓库的旧式检测头 `Detect_LSDECD` 不兼容 |
| numpy | 1.26.4 | 避免 numpy 2.x 与 torch 的兼容性问题 |
| einops / timm | 最新 | `timm` 提供 `DropPath` |

> 备选配套：torch **2.3.1**+cu121 ↔ torchvision **0.18.1**+cu121；torch **2.5.1**+cu121 ↔ torchvision **0.20.1**+cu121（均可，推荐 2.4.1）。
> CUDA 运行时说明：cu121 wheel 内嵌 CUDA 12.1 运行时 + cuDNN，只需服务器驱动版本 ≥ 525（CUDA 13.0 驱动完全兼容），无需再装系统 CUDA。

---

## 2. 依赖安装（完整命令）

```bash
# 2.1 检查驱动：要求 nvidia-smi 顶部 CUDA Version >= 12.1（13.0 兼容）
nvidia-smi

# 2.2 opencv 系统依赖（Ubuntu 常见坑：cv2 报 libGL.so.1 错误）
sudo apt update
sudo apt install -y libgl1 libglib2.0-0

# 2.3 不创建虚拟环境，直接在 base 主环境安装（pip 需 root 权限，或以 --user 方式安装避免污染系统包）
sudo python3.12 -m pip install --upgrade pip
# 或（当前用户级安装，后续所有命令同样加 --user 或先 sudo -s 提权）：
# python3.12 -m pip install --user --upgrade pip

# 2.4 torch / torchvision（CUDA 12.1 版，linux cp312 wheel）
sudo python3.12 -m pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2.5 ultralytics（版本固定）
sudo python3.12 -m pip install "ultralytics==8.3.253"

# 2.6 其余依赖（⚠️ 版本已针对 Python 3.12 调整）
sudo python3.12 -m pip install einops timm==1.0.14 numpy==1.26.4 opencv-python==4.9.0.80 \
    albumentations==2.0.4 onnx==1.15.1 onnxruntime==1.16.3 \
    pycocotools==2.0.7 PyYAML==6.0.1 scipy==1.13.0 onnxslim==0.1.31 \
    psutil==5.9.8 py-cpuinfo==9.0.0 huggingface-hub==0.23.2 \
    safetensors==0.4.3 supervision==0.22.0
```

**版本说明（Python 3.12 兼容性）：**
- 原 `requirements.txt` 中的 `onnx==1.14.0`、`onnxruntime==1.15.1` **无 cp312 wheel**（分别是 1.15.0、1.16.0 起才支持 Python 3.12），3.12 下安装会触发源码编译并大概率失败，因此上调为 `onnx==1.15.1`、`onnxruntime==1.16.3`；
- 其余包（numpy 1.26.4、opencv-python 4.9.0.80、pycocotools、PyYAML、scipy 1.13.0、psutil、safetensors 等）均有 cp312 预编译 wheel；
- einops、timm、onnxslim、py-cpuinfo、huggingface-hub、supervision 为纯 Python 包，无版本限制；
- 若服务器只做**训练/验证**（不导出 ONNX），可删除 `onnx`、`onnxruntime` 两行。

# 2.7 验证
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# 期望：2.4.1+cu121 0.19.1+cu121 True
```

**跳过项（务必注意）**
- `requirements.txt` 中的 `flash_attn-2.7.3+cu11torch2.2...cp311...whl`：仅适用于 **cp311 + torch 2.2**。服务器为 py3.12 无法安装，且本项目代码未使用 flash_attn，跳过无任何影响。
- `onnxruntime-gpu`：非必需，`onnxruntime` CPU 版已覆盖验证/导出需求。

---

## 3. 代码修复清单（复现前必须完成）

本仓库原始代码存在残缺。以下修复已在本机（Windows）验证通过，代码层面无需再调整，仅需在服务器上执行"注册"步骤。

### 3.1 缺失类已实现（已在项目文件内）

- `ultralytics/nn/extra_modules/block.py`：已补 import（`Conv/C3k2/DWConv/C3k/DropPath`），并实现缺失类 `DualDomainSelectionMechanism`、`ConvolutionalGLU`（原代码引用但未定义，会抛 NameError）；
- `ultralytics/nn/extra_modules/head.py`：已补 import，实现 `Conv_GN`、`DEConv_GN`、`Scale`，且 `Detect_LSDECD` **必须继承官方 `Detect`**（否则框架不会计算 stride / 初始化 bias）。

> ⚠️ `DualDomainSelectionMechanism` / `ConvolutionalGLU` / `DEConv_GN` 在作者公开代码中无定义，为按论文描述的合理重建，与原论文可能存在差异。

### 3.2 注册自定义模块（唯一需要在服务器上做的代码操作）

**推荐方式：自动补丁脚本**（已随项目提供，压缩包自带，无需虚拟环境）

```bash
cd ~/MIE-YOLO-main
python3.12 patch_mie.py    # 依赖装到系统目录需 sudo；--user 安装则直接 python3.12
```

脚本会自动完成以下工作，并打印结果：
- 自动定位已安装 ultralytics 的 `nn/tasks.py`（`/usr/lib/python3.12/site-packages/ultralytics/nn/tasks.py` 或 `~/.local/lib/python3.12/site-packages/...`，取决于安装方式）；
- 自动将项目路径写入 tasks.py（无需手动改任何路径）；
- 完成 4 处修改（模块加载器、`base_modules`、`repeat_modules`、检测头 frozenset），重复运行安全（幂等），修改前自动备份为 `tasks.py.mie_bak`；
- 修改后自动执行一次 `YOLO("MIE-YOLO.yaml")` 构建验证，输出 `MIE-YOLO build OK` 即成功。

**手动方式（备选）**：修改已安装的 `ultralytics/nn/tasks.py`，路径（base 主环境，默认 sudo 安装）：

```
/usr/lib/python3.12/site-packages/ultralytics/nn/tasks.py
```

**第 1 处**：在文件 import 区之后加入加载器（把两处 `~/MIE-YOLO-main` 换成服务器实际项目路径）：

```python
import importlib.util as _mie_ilu
import sys as _mie_sys

def _mie_load(name, path):
    _spec = _mie_ilu.spec_from_file_location(name, path)
    _mod = _mie_ilu.module_from_spec(_spec)
    _mie_sys.modules[name] = _mod
    _spec.loader.exec_module(_mod)
    return _mod

_mie_block = _mie_load("mie_extra_block", r"~/MIE-YOLO-main/ultralytics/nn/extra_modules/block.py")
_mie_head = _mie_load("mie_extra_head", r"~/MIE-YOLO-main/ultralytics/nn/extra_modules/head.py")

C3k2_MutilScaleEdgeInformationSelect = _mie_block.C3k2_MutilScaleEdgeInformationSelect
C3k2_AdditiveBlock_CGLU = _mie_block.C3k2_AdditiveBlock_CGLU
Detect_LSDECD = _mie_head.Detect_LSDECD
```

**第 2 处**：`parse_model` 内 `base_modules` frozenset（含 `C3k2` 的集合）加入：

```python
C3k2_MutilScaleEdgeInformationSelect,
C3k2_AdditiveBlock_CGLU,
```

**第 3 处**：`parse_model` 内 `repeat_modules` frozenset 也同样加入上面两个名字。

**第 4 处**：`parse_model` 内检测头分支 frozenset（含 `Detect, WorldDetect, YOLOEDetect, ...`）加入：

```python
Detect_LSDECD,
```

### 3.3 配置文件（仓库内已改好，无需再动）

- `MIE-YOLO.yaml`：`nc: 8`（Weed 数据集 8 类；换数据集时改成对应类别数）；
- `dataset/*/data.yaml`：已删除绝对路径 `path: D:/MIE-YOLO/dataset`，`train/val/test` 为相对 yaml 所在目录的 `images/...` 路径，**跨平台生效，与 CWD 无关**。

---

## 4. 数据集准备

`dataset/` 下目前只有 `labels/`，需上传图片（文件名与 labels 的 `.txt` 一一对应）：

```
/MIE-YOLO-main/dataset/Weed/
├── data.yaml          # 已改相对路径，无需修改
├── images/
│   ├── train/         # 训练图片（与 labels/train 对应）
│   ├── val/
│   └── test/
└── labels/
    ├── train/         # 已存在
    └── val/
```

上传方式示例：

```bash
rsync -avz local_images_dir/ user@server:/MIE-YOLO-main/dataset/Weed/images/train/
```

---

## 5. 训练与验证

```bash
cd ~/MIE-YOLO-main
# 已在 base 主环境，无需激活虚拟环境（依赖通过 sudo python3.12 -m pip 安装）

# 训练（train.py 位于项目根目录，内容见下）
python3.12 train.py
```

`train.py`：

```python
from ultralytics import YOLO

def main():
    model = YOLO("MIE-YOLO.yaml")
    model.train(data="dataset/Weed/data.yaml", epochs=300, imgsz=640,
                batch=16, workers=8, optimizer="SGD", device=0)
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()
```

验证 / 推理：

```bash
yolo val model=runs/detect/train/weights/best.pt data=dataset/Weed/data.yaml
yolo predict model=runs/detect/train/weights/best.pt source=<图片目录>
```

---

## 6. 常见问题（服务器）

| 问题 | 原因 | 解决 |
|---|---|---|
| `cv2.imshow` / libGL 报错 | 缺少 opencv 系统库 | `sudo apt install -y libgl1 libglib2.0-0` |
| `ImportError: No module named 'torchvision'` 或版本错误 | torch/torchvision 不配套 | 严格按 2.4.1 / 0.19.1 配套安装 |
| `onnx` / `onnxruntime` 安装时源码编译或失败 | 1.14.0 / 1.15.1 无 cp312 wheel | 用 1.15.1 / 1.16.3（文档已更新），或删除这两个包 |
| `KeyError: 'C3k2_MutilScaleEdgeInformationSelect'` | 未执行第 3.2 节注册 | 重新检查 tasks.py 四处修改 |
| `nvidia-smi` 有卡但 `cuda.is_available()` 为 False | 驱动过旧 | cu121 需驱动 ≥ 525（CUDA 13.0 驱动兼容） |
| `StringZilla` 编译失败 | 无现成 wheel | Linux 一般有预编译 wheel；若源码头文件编译，`pip install stringzilla --only-binary :all:` |
| 训练量 OOM | 8GB 显存 | 调小 `batch`（如 8/4）、`imgsz=640` 保持、加 `cache=False` |

---

## 7. 未实现部分（需自行开发）

**DS 双重自蒸馏**：两阶段自蒸馏。阶段一以训练好的 YOLO12-S 为教师、MIE-YOLO 为学生；阶段二以"学生通道翻倍版本"为教师再蒸馏一次。蒸馏损失建议采用 Shu et al. 的 Channel-wise Knowledge Distillation（CWD，通道维 softmax 后 KL 散度）。论文未给出损失权重 / 蒸馏层等细节，实现属工程重建，无法保证复现论文指标。

---

## 8. 已验证结果（本机同等修复后）

| 环节 | 结果 |
|---|---|
| 模型构建 | 562 层 / 2,493,463 参数 / 15.1 GFLOPs（n 尺度） |
| 推理前向 | 输出 `(1, 12, 8400)`，stride `[8, 16, 32]` |
| 训练（合成数据，GPU） | 1 epoch 正常，loss 收敛，权重正常保存 |

---

## 附录：Windows 本机环境对照（差异说明）

| 项 | Windows（本机 `mie` conda 环境） | Linux 服务器 |
|---|---|---|
| Python | 3.11.0 | **3.12.3**（固定） |
| torch / torchvision | 2.2.2+cu121 / 0.17.2+cu121 | **2.4.1+cu121 / 0.19.1+cu121** |
| flash_attn | 跳过 | 跳过（cp311 wheel 不适用 py312） |
| stringzilla | 需 `--only-binary :all:` 预装（无 MSVC 时） | 预编译 wheel 正常 |
| tasks.py 路径 | `F:\Anaconda_envs\envs\mie\Lib\site-packages\ultralytics\nn\tasks.py` | `/usr/lib/python3.12/site-packages/ultralytics/nn/tasks.py`（base 主环境，--user 安装则在 `~/.local/lib/python3.12/site-packages/...`） |
| 训练/验证命令 | 相同 | 相同 |