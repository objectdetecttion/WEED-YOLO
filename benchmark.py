import time
import torch
from ultralytics import YOLO

WEIGHTS = "runs/pose/train/weights/best.pt"
IMGSZ = 640
DEVICE = 0
WARMUP = 30
REPEAT = 100


def timed(model, x, n, batch):
    torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(x, imgsz=IMGSZ, device=DEVICE, verbose=False)
    torch.cuda.synchronize(DEVICE)
    return (time.perf_counter() - t0) / n


def main():
    model = YOLO(WEIGHTS)
    layers, params, grads, gflops = model.info()
    print(f"layers: {layers} | params: {params / 1e6:.2f}M | GFLOPs: {gflops:.2f}")

    x = torch.zeros(1, 3, IMGSZ, IMGSZ, device=DEVICE)
    for _ in range(WARMUP):
        model.predict(x, imgsz=IMGSZ, device=DEVICE, verbose=False)
    torch.cuda.synchronize(DEVICE)

    lat = []
    for _ in range(REPEAT):
        lat.append(timed(model, x, 1, 1))
    lat.sort()
    avg = sum(lat) / len(lat)
    print(f"batch=1 时延: avg {avg * 1000:.2f} ms | P50 {lat[len(lat) // 2] * 1000:.2f} ms | min {lat[0] * 1000:.2f} ms | FPS {1 / avg:.1f}")

    xb = torch.zeros(16, 3, IMGSZ, IMGSZ, device=DEVICE)
    t16 = timed(model, xb, REPEAT, 16)
    print(f"batch=16 吞吐: {16 / t16:.1f} img/s（单图当量 {t16 / 16 * 1000:.2f} ms）")


if __name__ == "__main__":
    main()
