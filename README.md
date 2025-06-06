
# Semantic Segmentation with ONNX Runtime

This project demonstrates **real-time semantic segmentation** using an exported ONNX model, applied to video input. The notebook includes preprocessing, ONNX inference, postprocessing, and overlaying segmentation masks on video frames.

---

## 📂 Contents

- `Semantic_segmentation.ipynb`: Main notebook for loading ONNX model, running inference on video, and saving output.
- `segmentation_model.onnx`: The exported segmentation model (ONNX format).
- `output_segmented_video_onnx_optimized.mp4`: Output video showing segmentation overlay.

---
## 🗂️ Dataset
- **Source**: [xiaose/cityscapes](https://huggingface.co/datasets/huggan/cityscapes)

- Classes used: Background, Lane, Person, Vehicle (4 out of original 19 classes)

- Format: PNG images with corresponding segmentation masks

- Preprocessing: Resized to 256×512, normalized with specified mean and std

This subset is designed to benchmark real-time segmentation performance while keeping the class complexity manageable.
---
## 🧠 Model Overview

- **Architecture**: Custom semantic segmentation model (3 classes)
- **Input shape**: Auto-detected from ONNX input (e.g., `[1, 3, 256, 512]`)
- **Normalization**:
  - Mean: `[0.2942, 0.3274, 0.2879]`
  - Std: `[0.1794, 0.1834, 0.1808]`
- **Exported with**: `torch.onnx.export(...)`
- **ONNX Runtime Providers**: Supports `CUDAExecutionProvider` and `CPUExecutionProvider`

---

## 🏷️ Semantic Classes

```python
COLOR_MAP = {
    1: (255, 0, 0),    # Lane - Red
    2: (0, 255, 0),    # Person - Green
    3: (0, 0, 255),    # Vehicle - Blue
}
```

---

## 🎞️ Video Inference Pipeline

The notebook loads a video, performs inference frame-by-frame using ONNX Runtime, and saves a result video with masks overlaid.

- **Overlay**: Blended with alpha=0.6
- **Layout**: Side-by-side (original vs. overlay)
- **Output format**: `.mp4`

---

## ⚡ Inference Performance

| Test Platform          | Inference FPS (model only) 
|------------------------|----------------------------|
| NVIDIA L4 (24GB)       | **~280 FPS**               |

> Note: Measured on 256×512 input resolution, using `onnxruntime-gpu`. FPS only reflects model inference time in the first column.

---

## ✅ Evaluation on Validation Set

Example test results using a held-out dataset:

| Metric           | Value    |
|------------------|----------|
| Mean IoU         | 73.4%    |


---

## 🚀 How to Use

### 1. Install dependencies:

```bash
pip install onnxruntime-gpu opencv-python albumentations
```

### 2. Run the notebook:

- Update `video_input_path` to your input video.
- Ensure your ONNX model path is correct.
- Run the notebook to generate predictions.

### 3. Output:

- Output video: `output_segmented_video_onnx_optimized.mp4`
- Format: side-by-side original and segmentation overlay
