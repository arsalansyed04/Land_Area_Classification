# DeepGlobe Land Cover Segmentation with U-Net

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras%203-orange)
![Task](https://img.shields.io/badge/Task-Semantic%20Segmentation-green)

A U-Net, written from scratch in TensorFlow/Keras, that labels every pixel of a satellite image with one of seven land cover classes. It is trained and evaluated on the [DeepGlobe Land Cover Classification](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) dataset.

The whole pipeline lives in one notebook, `LandAreaClassification.ipynb`: dataset download, preprocessing, model, training, evaluation and visualisation.

## Results

Validation results from two runs of the same notebook code (same seed, same split):

| Run | Val. loss | Pixel accuracy | Dice | IoU |
| :--- | :---: | :---: | :---: | :---: |
| 1 | 0.7423 | 82.58% | 0.7664 | 0.6307 |
| 2 | 0.7373 | 83.20% | 0.7530 | 0.6127 |

![Training curves](assets/training_curves.png)

**How to read these numbers.**

- Dice and IoU here are **global (micro-averaged) soft scores**. They are computed over all pixels and all classes at once, on softmax outputs. They are **not** the per-class mean IoU (mIoU) used in DeepGlobe leaderboards, so don't compare them directly to published mIoU figures.
- Per-class IoU/F1 and a confusion matrix are not computed yet, so I can't say which classes the model handles well or badly. The sample predictions below suggest Rangeland and Barren land are confused with each other.
- The two runs differ by about 0.02 IoU with identical code, so differences of that size are within run-to-run noise.
- The validation set is also the set used for checkpoint selection and early stopping, and there is no separate test set. Treat the numbers as slightly optimistic.

## Sample predictions

Left to right: satellite image, ground truth, model prediction (validation images).

![Predictions](assets/predictions.png)

The colours come from matplotlib's `tab10` colormap, not the official DeepGlobe palette:

| Colour | Class |
| :--- | :--- |
| Blue | Urban |
| Orange | Agriculture |
| Red | Rangeland |
| Brown | Forest |
| Pink | Water |
| Olive | Barren |
| Cyan | Unknown |

## Method

| | |
| :--- | :--- |
| **Model** | Standard U-Net: 4 encoder blocks (64 → 512 filters), a 1024-filter bottleneck, 4 decoder blocks with skip connections. Conv → BatchNorm → ReLU, transposed-conv upsampling. 31,055,687 parameters. |
| **Input / output** | 512×512 RGB in, 512×512×7 softmax out. Images are resized to 512×512 and scaled to [0, 1]. |
| **Data** | The 803 labelled DeepGlobe training tiles, split randomly 80/20 into 642 train / 161 validation (seed 42). The dataset's `valid` and `test` folders have no masks, so they are not used. |
| **Loss** | Categorical cross-entropy + Dice loss (global, not per-class). |
| **Optimiser** | Adam, learning rate 1e-4, batch size 4, up to 50 epochs. |
| **Callbacks** | `ModelCheckpoint` (best `val_loss`), `ReduceLROnPlateau` (×0.5, patience 5), `EarlyStopping` (patience 10, restores best weights), `CSVLogger`. |
| **Speed** | About 200 s per epoch in the logged runs. |

## Getting started

### 1. Install dependencies

```bash
pip install tensorflow numpy pandas opencv-python scikit-learn matplotlib kaggle
```

The logged runs used Python 3.12 with Keras 3 (TensorFlow 2.16 or newer). A GPU is strongly recommended.

### 2. Set up Kaggle access

The notebook downloads the dataset through the Kaggle API.

1. Create an API token at <https://www.kaggle.com/settings> and download `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Linux/macOS) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows).
3. On Linux/macOS, run `chmod 600 ~/.kaggle/kaggle.json`.

Never commit `kaggle.json` to the repo.

### 3. Run

Open `LandAreaClassification.ipynb` and run the cell. It will:

1. Download and unzip the dataset to `./deepglobe_dataset`.
2. Build and train the model.
3. Write these files next to the notebook:
   - `best_unet_deepglobe.h5` (best checkpoint)
   - `final_unet_deepglobe.h5` (final weights)
   - `training_log_deepglobe.csv`
   - `deepglobe_training_history.png`
   - `deepglobe_predictions.png`

To train faster or on a smaller GPU, change `IMG_SIZE` to `(256, 256)` or `BATCH_SIZE` to `2` in the configuration block at the bottom of the notebook.

## Repository structure

```
.
├── LandAreaClassification.ipynb   # full pipeline: data, model, training, evaluation
├── assets/
│   ├── training_curves.png
│   └── predictions.png
└── README.md
```

## Limitations and next steps

- No data augmentation. The data generator has an `augment` flag, but it is not implemented. Flips and rotations are an easy first improvement.
- No class weighting. Agriculture dominates the pixels, and the Dice term is global, so it does little to counter class imbalance.
- The "Unknown" class is trained as a normal class rather than ignored.
- Images are downsampled to 512×512, which loses fine detail such as small buildings and roads.
- Evaluation is incomplete: add per-class IoU/F1 and a confusion matrix, report true mIoU, and hold out a proper test split.
- Models are saved in the legacy `.h5` format. Keras recommends `.keras`.
- Possible upgrades: a pretrained encoder (e.g. ResNet or EfficientNet backbone), tiling instead of resizing, and loss functions better suited to class imbalance (focal or weighted Dice).

## Dataset and credits

- DeepGlobe 2018: *A Challenge to Parse the Earth through Satellite Images* (Demir et al., CVPR Workshops 2018).
- Kaggle mirror used here: [`balraj98/deepglobe-land-cover-classification-dataset`](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset). Check the dataset's terms before reusing or redistributing the data.
- U-Net architecture: Ronneberger, Fischer, Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
