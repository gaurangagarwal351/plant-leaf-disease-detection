# Plant Leaf Disease Detection

A small convolutional neural network (CNN) that classifies a leaf photo as **healthy** or **diseased**. It includes data preparation, training, held-out evaluation, command-line prediction, and a Streamlit upload app. The network is trained from scratch; it does not ship with pretrained weights.

## Setup

Use Python 3.10–3.14 and install dependencies in a virtual environment. On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
```

For a machine-specific GPU installation, use the [PyTorch install selector](https://pytorch.org/get-started/locally/) before installing the remaining dependencies.

## Add images

The simplest source is a folder with two labeled subfolders:

```text
my_leaf_photos/
├── healthy/
│   ├── leaf1.jpg
│   └── ...
└── diseased/
    ├── leaf2.jpg
    └── ...
```

Use at least three images in each class to run the pipeline, and many more for a useful model. Labels should be checked by someone who knows the crop and disease. Then prepare stratified 70/15/15 train, validation, and test splits:

```bash
python prepare_data.py --source my_leaf_photos --layout binary --output data
```

You can also download the original [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset) and use its `raw/color` folder:

```bash
python prepare_data.py --source PlantVillage-Dataset/raw/color --layout plantvillage --output data
```

The preparation command copies images into `data/train`, `data/val`, and `data/test`, each with `healthy` and `diseased` subfolders. It writes `data/manifest.csv` and refuses to overwrite nonempty output. Its default seed is 42. A second run needs a new empty output folder or removal of the old prepared folder.

**Evaluation note:** this script splits individual images. If several photos show the same leaf or plant, put all of them in the same split for a trustworthy test. The PlantVillage maintainers offer [grouped train/test splits](https://github.com/spMohanty/PlantVillage-Dataset#recommended-usage-via-hugging-face) for that reason. The image-level split here is useful for learning the workflow, but its test score may overstate performance on new plants or field photos.

## Train and evaluate

```bash
python -m leaf_disease.train --data-dir data --epochs 20 --batch-size 32
python -m leaf_disease.evaluate --data-dir data
```

Training selects Apple MPS, NVIDIA CUDA, or CPU automatically. Change `--device` if needed. It uses augmentation only for training, class-weighted cross-entropy, and early stopping on validation loss. The best checkpoint is `outputs/best_model.pt`, training history is `outputs/history.json`, and test metrics are `outputs/test_metrics.json`. Test metrics include accuracy, precision, recall, F1 for the **diseased** class, and a confusion matrix.

## Predict

```bash
python -m leaf_disease.predict path/to/new_leaf.jpg
streamlit run app.py
```

Predictions contain both class scores and the highest-scoring label. These scores are model outputs, not calibrated probabilities. This binary project does not identify a crop or a particular disease. It is an educational classifier, so check important agricultural decisions with an expert and evaluate it on photos from the conditions where it will be used.

## Project files

| Path | Purpose |
| --- | --- |
| `prepare_data.py` | Create reproducible data splits and a manifest |
| `leaf_disease/model.py` | Four-block CNN |
| `leaf_disease/train.py` | Training and best-checkpoint saving |
| `leaf_disease/evaluate.py` | Held-out test metrics |
| `leaf_disease/predict.py` | Classify one image |
| `app.py` | Image upload interface |

Run the standard-library checks with `python -m unittest discover -s tests -v`.
