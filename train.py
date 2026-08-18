from ultralytics import YOLO

# ==================== 任务配置 ====================
MODEL = "MIE-pose.yaml"             # 姿态: MIE-pose.yaml / 检测: MIE-YOLO.yaml
PRETRAINED = "yolo11n-pose.pt"      # 预训练权重：检测用 True；pose 任务用 "yolo11n-pose.pt"
DATA = "dataset/MyData/data.yaml"   # 换成你的自定义数据集 yaml

# ==================== 训练配置 ====================
EPOCHS = 300
IMGSZ = 960                         # 640 → 960，小目标关键点回归更吃分辨率
BATCH = 32                          # 显存不足时调小（如 16），并相应调低 lr0
WORKERS = 8
DEVICE = 0
CACHE = False
PLOTS = True
SEED = 0
DETERMINISTIC = True
PATIENCE = 100                      # 早停耐心
SAVE_PERIOD = -1                    # 每隔 N epoch 存一次权重（-1 禁用）
RESUME = False
AMP = True
FRACTION = 1.0                      # 训练集使用比例
MULTI_SCALE = False
SINGLE_CLS = False
RECT = False

# ==================== 优化器与学习率 ====================
OPTIMIZER = "SGD"                   # SGD / Adam / AdamW / auto
LR0 = 0.005                         # 初始学习率（SGD 常用 0.005~0.01）
LRF = 0.01                          # 最终学习率 = lr0 * lrf
MOMENTUM = 0.937
WEIGHT_DECAY = 0.0001               # 默认 5e-4 → 1e-4，小数据集更稳
WARMUP_EPOCHS = 3.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1
COS_LR = True                       # 余弦退火，长训练收敛更好
NBS = 64                            # 名义 batch size（loss 归一化用）

# ==================== 损失权重 ====================
BOX = 7.5
CLS = 0.5
DFL = 1.5
POSE = 14.0                         # 关键点损失权重（默认 12，卡点回归时调高）
KOBJ = 1.0                          # 关键点可见性损失权重
LABEL_SMOOTHING = 0.0

# ==================== 数据增强 ====================
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
DEGREES = 0.0                       # 旋转（度），小幅旋转 5~10 可提泛化
TRANSLATE = 0.1
SCALE = 0.2                         # 默认 0.5，缩放太大对点坐标有害
SHEAR = 0.1
PERSPECTIVE = 0.0
FLIPUD = 0.5                        # 上下翻转
FLIPLR = 0.5                        # 左右翻转
MOSAIC = 0.6                        # 默认 1.0，拼接错位会引入关键点噪声，调低更利于 pose
MIXUP = 0.0
CUTMIX = 0.0
COPY_PASTE = 0.0
CLOSE_MOSAIC = 30                   # 最后 N 个 epoch 关闭 mosaic，稳定训练

# ==================== 验证/推理 ====================
VAL = True
SAVE_JSON = False
CONF = 0.15                         # 验证置信度阈值（默认 0.001，0.15 与 Taote 一致）
IOU = 0.7
MAX_DET = 300
HALF = False


def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA,
        pretrained=PRETRAINED,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        cache=CACHE,
        plots=PLOTS,
        seed=SEED,
        deterministic=DETERMINISTIC,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        resume=RESUME,
        amp=AMP,
        fraction=FRACTION,
        multi_scale=MULTI_SCALE,
        single_cls=SINGLE_CLS,
        rect=RECT,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_momentum=WARMUP_MOMENTUM,
        warmup_bias_lr=WARMUP_BIAS_LR,
        cos_lr=COS_LR,
        nbs=NBS,
        box=BOX,
        cls=CLS,
        dfl=DFL,
        pose=POSE,
        kobj=KOBJ,
        label_smoothing=LABEL_SMOOTHING,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=DEGREES,
        translate=TRANSLATE,
        scale=SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIPUD,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        cutmix=CUTMIX,
        copy_paste=COPY_PASTE,
        close_mosaic=CLOSE_MOSAIC,
        val=VAL,
        save_json=SAVE_JSON,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        half=HALF,
    )


if __name__ == "__main__":
    main()