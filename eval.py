from ultralytics import YOLO

WEIGHTS = "runs/pose/train3/weights/best.pt"
DATA = "dataset/MyData/data.yaml"
IMGSZ = 640
DEVICE = 0


def main():
    model = YOLO(WEIGHTS)
    r = model.val(data=DATA, imgsz=IMGSZ, device=DEVICE)

    print("\n===== 评估结果 =====")
    print(f"检测 mAP50    : {r.box.map50:.4f}")
    print(f"检测 mAP50-95 : {r.box.map:.4f}")
    print(f"关键点 mAP50   : {r.pose.map50:.4f}")
    print(f"关键点 mAP50-95: {r.pose.map:.4f}")

    print("\n----- 各类别 AP50 -----")
    for cls_id, ap50 in zip(r.box.ap_class_index, r.box.ap50):
        name = model.names.get(int(cls_id), str(cls_id))
        print(f"  class {int(cls_id)} ({name}): {ap50:.4f}")


if __name__ == "__main__":
    main()
