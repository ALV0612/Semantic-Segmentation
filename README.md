
# Semantic Segmentation with Lyft Udacity Dataset

This repository demonstrates how to build a custom semantic segmentation model using the **Lyft Udacity Challenge** dataset. The project is implemented using **PyTorch Lightning** and leverages **custom data pipelines** with `albumentations`, visualization tools, and evaluation metrics.

---

##  Dataset

- **Source**: [kumaresanmanickavelu/lyft-udacity-challenge](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge)
- Downloaded using `kagglehub`
- Dataset structure:
  ```
  dataA/
    CameraRGB/
    CameraSeg/
  dataB/
    ...
  ```

---

## Main Libraries Used

- `PyTorch`, `PyTorch Lightning`
- `torchvision`, `albumentations`
- `sklearn`, `PIL`
- `torchmetrics` for mIoU (Jaccard Index)

---

##  Features

- **Data Loader**: Loads both RGB and segmentation mask images with matching names across folders.
- **Augmentation**: Performed using `albumentations` (resizing, normalization, etc.)
- **Custom Model**: (To be filled in depending on architecture used(MobileNet-v2 backbone combined encoder-decoder, multi-branch structure))
- **Logging**: Integrated with TensorBoard.
- **Metrics**: Supports mIoU evaluation with `MulticlassJaccardIndex`.

---

## Visualization

- Random sample of segmentation masks and RGB images displayed for inspection.
- Includes visualization code to verify preprocessing steps and data integrity.

---

## How to Run

```bash
# Install dependencies
pip install pytorch_lightning albumentations imageio kagglehub

# Run the notebook
jupyter notebook customModel_with_kaggle_ds.ipynb
```

---

## ✅ TODO

- [ ] Add training and validation loops
- [ ] Add custom loss functions (Dice, Focal, OHEM, etc.)
- [ ] Finalize and save model checkpoints
