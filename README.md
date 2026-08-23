# Severstal Steel Defect Detection

Pixel-level detection of four surface defect types on steel sheets, using a
U-Net with an ImageNet-pretrained ResNet34 encoder.

**CENG 476 — Introduction to Deep Learning, Summer 2025-2026**

---

## Task

Semantic segmentation — more precisely, **multi-label pixel classification**.
For every pixel of a 256 × 1600 image, the model makes four independent binary
decisions (one per defect class), because a pixel can belong to more than one
defect type.

| | |
|---|---|
| **Dataset** | [Severstal: Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) (Kaggle, 2019) |
| **Images** | 12,568 · 256 × 1600 px |
| **Classes** | 4 defect types |
| **Masks** | RLE-encoded, column-major |
| **Metric** | Mean Dice coefficient over all (image, class) pairs |

### Dataset characteristics that shaped our design

- **~47% of images contain no defect at all** (4,131 of 8,797 training images)
- **Severe class imbalance:** Class 3 has 3,605 training images, Class 2 has only 173
- **~98% of all pixels are background** — this is why accuracy is not a usable metric

---

## Results

### Ablation study

| Run | Pretrained | BatchNorm | Dropout | Attention | Augment | Weight decay | Scheduler | **Val Dice** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **final** | ✓ | ✓ | 0.2 | ✓ | ✓ | 1e-4 | cosine | **0.9308** |
| dropout04 | ✓ | ✓ | 0.4 | ✓ | ✓ | 1e-4 | cosine | 0.8983 |
| baseline | ✗ | ✗ | 0.0 | ✗ | ✗ | 0.0 | none | 0.8850 |

Enabling all techniques raised validation Dice from 0.8850 to 0.9308 — roughly
40% of the remaining error gap closed. The dropout sweep shows the expected
U-shape: too little leaves the model prone to memorisation, too much slows
learning down.

### Test set — final model

| Metric | Value |
|---|---|
| **Mean Dice** (tuned: th=0.5, min_size=1200) | **0.9446** |
| Mean Dice (default: th=0.5, min_size=0) | 0.9396 |
| Mean IoU | 0.9289 |
| Macro F1 (defect detection) | 0.6725 |
| Trivial baseline ("predict empty everywhere") | 0.8625 |

### Per-class results

| Class | Dice (all) | Dice (defective only) | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Class 1 | 0.9642 | 0.4997 | 0.7931 | 0.7419 | 0.7667 |
| Class 2 | 0.9867 | **0.0000** | 0.0000 | 0.0000 | 0.0000 |
| Class 3 | 0.8403 | 0.6634 | 0.9633 | 0.9147 | 0.9384 |
| Class 4 | 0.9871 | 0.7963 | 0.9706 | 1.0000 | 0.9851 |

> **The Class 2 row is the most informative finding in this table.** Its
> "Dice (all)" of 0.9867 looks excellent, but the model never actually detects
> Class 2 — recall is zero. The high score comes entirely from correctly saying
> "no Class 2 here" on images that genuinely have none. With only 173 training
> examples against Class 3's 3,605, the model never learned this class. This is
> why we report both columns.

---

## Method

| Component | Choice |
|---|---|
| Architecture | U-Net decoder + ResNet34 encoder (ImageNet pretrained) |
| Parameters | 24.46 M, all trainable |
| Attention | SCSE (spatial + channel squeeze & excitation) in every decoder block |
| Auxiliary head | Classification head from the bottleneck (multi-task learning) |
| Normalization | BatchNorm after every decoder conv (Conv → BN → ReLU) |
| Dropout | Dropout2d (p=0.2) in decoder blocks, Dropout (p=0.2) before the classifier |
| Regularization | AdamW weight decay 1e-4, flips + brightness/contrast augmentation |
| Loss | 0.5 × weighted BCE + 0.5 × soft Dice + 0.3 × auxiliary BCE |
| Optimizer | AdamW, lr 3e-4, betas (0.9, 0.999) |
| Scheduler | CosineAnnealingLR, 3e-4 → 3e-6 |
| Early stopping | Validation mean Dice, patience 6 |
| Post-processing | Threshold + min-component-size grid search on validation, flip TTA |
| Hardware | NVIDIA Tesla T4 ×2 (Kaggle Notebooks), 92.5 min for the final model |

### Why sigmoid, not softmax?

Softmax forces class outputs to compete and sum to 1, which encodes the
assumption that each pixel belongs to exactly one class. In this dataset a pixel
can carry more than one defect type, so each class is treated as an independent
binary problem with a sigmoid output.

### Why not accuracy?

98% of pixels are background. A model that predicts "clean" everywhere scores 98%
accuracy and is useless. Dice and IoU do not reward empty predictions.

---

## Repository structure

```
├── src/
│   ├── config.py          # all hyperparameters in one place
│   ├── utils.py           # seed control, RLE encode/decode, training curves
│   ├── dataset.py         # data loading, stratified split, augmentation
│   ├── model.py           # U-Net + ResNet34 + SCSE + auxiliary head
│   ├── losses.py          # combined BCE + Dice loss
│   ├── metrics.py         # Dice, IoU, precision / recall / F1
│   ├── train.py           # training loop, scheduler, early stopping, AMP
│   ├── evaluate.py        # threshold search, TTA, test metrics, plots
│   ├── predict_demo.py    # qualitative demo on unseen test images
│   └── compare_runs.py    # ablation table across all runs
├── notebooks/
│   └── severstal_main.ipynb
├── outputs/               # generated per run (curves, metrics, models)
├── requirements.txt
└── README.md
```

---

## How to run

### On Kaggle (recommended — free GPU, dataset already mounted)

Create a notebook, enable **GPU T4 ×2** and **Internet**, then:

```python
!git clone https://github.com/SedefKOC/Steel-Defect-Detection.git
%cd Steel-Defect-Detection
!pip install -q albumentations

DATA = "/kaggle/input/competitions/severstal-steel-defect-detection"

# quick sanity check (~3 min)
!python src/train.py --data_dir {DATA} --limit 300 --epochs 2 --run_name smoke_test

# baseline: every technique disabled
!python src/train.py --data_dir {DATA} --run_name baseline --pretrained 0 --use_bn 0 \
    --decoder_dropout 0.0 --use_attention 0 --weight_decay 0.0 --augment 0 \
    --cls_loss_weight 0.0 --scheduler none --epochs 15

# final model
!python src/train.py --data_dir {DATA} --run_name final --epochs 30

# dropout ablation
!python src/train.py --data_dir {DATA} --run_name dropout04 --decoder_dropout 0.4 --epochs 12

# evaluation
!python src/evaluate.py --data_dir {DATA} --run_name final --val_batch_size 2 --num_workers 0
!python src/predict_demo.py --data_dir {DATA} --run_name final
!python src/compare_runs.py --out_dir outputs
```

### Locally

```bash
pip install -r requirements.txt
python src/train.py --data_dir path/to/data --run_name final --epochs 30
```

Requires an NVIDIA GPU. On CPU a full run takes days rather than hours.

### Outputs

Each run writes to `outputs/<run_name>/`:

| File | Contents |
|---|---|
| `best_model.pt` | weights of the best epoch |
| `curves.png` | loss / Dice / learning-rate curves |
| `history.csv` | per-epoch metrics |
| `metrics_test.json` | all test metrics |
| `results_table.md` | report-ready metric table |
| `confusion_matrix.png` | per-class 2×2 confusion matrices |
| `predictions.png` | example predictions with overlays |
| `demo_predictions.png` | qualitative demo on unseen images |
| `postprocess_grid.csv` | threshold × min_size search grid |

---

## Reproducibility

All random sources are fixed through `--seed` (default 42): Python's `random`,
NumPy, PyTorch, and cuDNN deterministic mode. Train / validation / test splits
are written to CSV so evaluation always runs on the identical held-out set; if
those files are missing, `evaluate.py` regenerates the same split from the seed.

---

## Notes and limitations

- The competition's test labels were never released, so `train_images` was split
  70/15/15 (stratified on the defect-class signature) into train / validation /
  test. **Reported scores are therefore not directly comparable to the Kaggle
  leaderboard.**
- Test evaluation was run on a random 600-image subset of the test split due to
  compute constraints.
- A single seed was used; result variance was not measured.
- No cross-validation was performed.
- Class 2 is severely under-represented and the model fails to detect it.

---

## References

1. Ronneberger, O., Fischer, P., Brox, T. (2015). *U-Net: Convolutional Networks
   for Biomedical Image Segmentation.* MICCAI.
2. He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for Image
   Recognition.* CVPR.
3. Roy, A. G., Navab, N., Wachinger, C. (2018). *Concurrent Spatial and Channel
   Squeeze & Excitation in Fully Convolutional Networks.* MICCAI.
4. Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
