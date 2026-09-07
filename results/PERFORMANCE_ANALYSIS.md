# VisionStock — Performance Analysis

## 1. Inference Latency

Benchmarked on local MacBook Air (Apple Silicon, CPU-only, 20 warm iterations after 3-run warmup).

| Metric | Value |
|---|---|
| Cold start (first inference) | 130.5ms |
| Warm average | 59.7ms |
| Warm minimum | 55.8ms |
| Warm maximum | 68.2ms |

Full API round-trip on GCP Cloud Run (includes network + DB write): **< 2s**

**How measured:**
```python
from ultralytics import YOLO
import time, numpy as np

model = YOLO('yolov8n.pt')
dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

# 3-run warmup
for _ in range(3):
    model(dummy, verbose=False)

# 20-run benchmark
times = []
for _ in range(20):
    start = time.time()
    model(dummy, verbose=False)
    times.append((time.time() - start) * 1000)
```

---

## 2. Cost Per 1,000 Images (GCP Cloud Run)

Based on GCP Cloud Run public pricing (us-central1 region):

| Resource | GCP Price |
|---|---|
| vCPU-second | $0.00002400 |
| Memory GB-second | $0.00000250 |

**Calculation:**
```
Inference time per image : ~0.12s (120ms on Cloud Run CPU)
vCPU cost per image      : 0.12s × $0.000024 = $0.00000288
Memory cost per image    : 0.12s × $0.0000025 = $0.00000030
Total per image          : ~$0.0000032

Cost per 1,000 images    : ~$0.003  (less than 1 cent)
Cost per 100,000 images  : ~$0.32
Cost per 1,000,000 images: ~$3.20
```

**Includes free tier**: Cloud Run provides 180,000 vCPU-seconds free per month —
roughly **25,000 images/month at zero cost**.

---

## 3. Model Configuration

| Parameter | Value | Location |
|---|---|---|
| Confidence threshold | 0.25 | `backend/config.py` |
| NMS IoU threshold | 0.45 | `backend/config.py` |
| Input image size | 640×640 | `data/custom/data.yaml` |
| Max upload size | 10MB | `backend/config.py` |

**Why 0.25 confidence?** Low threshold to maximise recall — for inventory
detection, catching a faint product detection is better than missing it entirely.

**Why 0.45 IoU for NMS?** Slightly below the standard 0.5 — removes overlapping
duplicate boxes more aggressively on densely packed shelves.

---

## 4. Dataset Limitations

| Limitation | Detail |
|---|---|
| Dataset size | 111 images total (78 train, 22 val, 11 test) |
| Images per class | ~2.3 on average across 34 classes |
| Test set size | 11 images — limited statistical confidence |
| Store diversity | Single store, controlled indoor lighting |
| Lighting conditions | No low-light, outdoor, or mixed-lighting scenarios |
| Occlusion | Minimal partial occlusion in training data |
| Class imbalance | Some classes have fewer than 2 training examples |
| Annotation tool | Roboflow — manual bounding boxes, human error possible |

**Impact on metrics:** Low training images per class directly limits mAP50 (4.04%)
and recall (11.79%). Both metrics would improve significantly with 500+ images per
class. The 42× recall improvement over baseline remains valid as a relative measure
of fine-tuning effectiveness.

---

## 5. Model Fallback Chain

If the fine-tuned model (Ultralytics Hub) is unavailable, the system falls back:

```
1. Hub URL (fine-tuned, best accuracy)
        ↓ fails
2. runs/detect/train/weights/best.pt (local trained weights)
        ↓ not found
3. models/yolov8-finetuned.pt (saved fine-tuned file)
        ↓ not found
4. yolov8n.pt (COCO baseline, always bundled in Docker image)
```

API continues serving detections in all cases — database write is also optional
(system runs in demo mode if PostgreSQL is unreachable).
