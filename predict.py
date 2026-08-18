from ultralytics import YOLO

WEIGHTS = "runs/pose/train/weights/best.pt"
SOURCE = "dataset/MyData/images/test"
CONF = 0.25
IMGSZ = 640
DEVICE = 0


def main():
    model = YOLO(WEIGHTS)
    results = model.predict(
        source=SOURCE,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
    )
    print(f"processed {len(results)} images, results saved to runs/pose/predict/")
    for r in results:
        print(r.keypoints)  # 每帧关键点 (x, y, visible)


if __name__ == "__main__":
    main()