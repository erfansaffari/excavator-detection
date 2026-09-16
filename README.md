# Excavator Detection with YOLO11

Fine-tune a COCO-pretrained YOLO11n model to detect excavators in construction-site images and videos. The repository covers dataset inspection and conversion, a one-epoch smoke test, GPU training, held-out test evaluation, image/video/folder inference, and a Streamlit demo.

## Pipeline

```text
MOCS dataset → inspect annotations → single-class YOLO conversion
             → smoke test → YOLO11n fine-tuning → test evaluation
             → image/video inference → Streamlit demo
```

## Model and dataset

- Model: `yolo11n.pt`, initialized from COCO-pretrained weights (never trained from scratch in this workflow).
- Dataset: [MOCS on Kaggle](https://www.kaggle.com/datasets/xiaopan9802/mocs-dataset), the Moving Objects in Construction Sites dataset.
- Target: `Excavator`, remapped to class ID `0`; other object boxes are removed.

MOCS is reported to contain 41,668 images and 13 construction-site object categories with pixel-level annotations. The Kaggle copy is about 10.2 GB. Because Kaggle distributions can differ in folder layout and annotation export, the scripts inspect the downloaded archive rather than assuming its structure. COCO JSON, Pascal VOC XML, LabelMe polygons/rectangles, and existing YOLO labels are supported. The converter refuses to proceed unless the excavator class can be identified by name (or an explicitly verified YOLO class ID is supplied).

> Review the dataset's current license and terms on Kaggle before use. Do not commit the dataset or trained weights to Git.

## Repository layout

```text
configs/excavator.yaml       YOLO dataset configuration
excavator_detection/        Dataset discovery/conversion library
scripts/inspect_dataset.py  Format, class, split, and box inspection
scripts/prepare_dataset.py  Excavator filtering and YOLO conversion
scripts/train.py             Smoke test and full training
scripts/evaluate.py          Held-out test metrics and plots
scripts/predict.py           Image, video, and folder inference
app/app.py                   Streamlit image demo
notebooks/kaggle_training.ipynb
tests/                       Dataset conversion/split tests
results/                     Generated runs (ignored by Git)
figures/                     Generated inspection figures (ignored by Git)
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and extract the Kaggle dataset into `data/raw/mocs/`, or attach it as a Kaggle Notebook input. With the Kaggle CLI configured, for example:

```bash
mkdir -p data/raw/mocs
kaggle datasets download -d xiaopan9802/mocs-dataset -p data/raw/mocs --unzip
```

## 1. Inspect annotations

Always run inspection before conversion. It reports discovered images, annotation format, exact class names/counts, inferred official splits, and writes labeled samples to `figures/dataset_samples/`.

```bash
python scripts/inspect_dataset.py --data data/raw/mocs
```

Open several generated images and verify the boxes. If the source is already YOLO but has no YAML class map, first identify the excavator ID from the dataset documentation, then rerun with `--excavator-class-id ID`. Do not guess the ID.

## 2. Prepare the dataset

```bash
python scripts/prepare_dataset.py --data data/raw/mocs
```

The prepared data is written to `data/processed/{images,labels}/{train,val,test}` and `configs/excavator.yaml` is updated with an absolute dataset path. `data/processed/manifest.json` records the source, detected classes/formats, split policy, and output counts.

Split policy:

1. Preserve an existing complete train/validation/test split.
2. If train and test exist but validation does not, preserve test and deterministically take validation from train.
3. Otherwise create a deterministic 70/15/15 split with seed 42.

Images without excavator boxes are retained as negative examples by default. Add `--drop-empty` if a deliberate positive-only dataset is needed.
The default `--image-mode copy` makes the processed dataset self-contained. On Kaggle, the included notebook uses `--image-mode symlink` to avoid duplicating roughly 10 GB of input data on the working disk; those links are valid for the attached dataset during that Kaggle session.

## 3. Smoke test, then train

Select a GPU runtime. Training requires CUDA by default so a missing GPU cannot accidentally turn a short run into a very long CPU job.

```bash
# Required one-epoch pipeline check
python scripts/train.py --smoke-test --device 0

# Full initial run: YOLO11n, 640 px, 50 epochs, seed 42, early stopping
python scripts/train.py --device 0
```

The default early-stopping patience is 10 epochs. The best checkpoint is normally at:

```text
results/train/yolo11n_excavator/weights/best.pt
```

For Kaggle, open `notebooks/kaggle_training.ipynb`, attach the MOCS dataset and this repository, enable a GPU accelerator, and run the cells in order.

## 4. Evaluate the held-out test set

```bash
python scripts/evaluate.py \
  --weights results/train/yolo11n_excavator/weights/best.pt \
  --device 0
```

The evaluation run saves `metrics.json`, precision-recall curves, confusion matrices, and other Ultralytics validation plots under `results/evaluation/test/`.

## 5. Run inference

The same command accepts an image, a video, or a directory of images. Annotated media contains a bounding box and confidence score for every detection.

```bash
# Image
python scripts/predict.py --source test.jpg

# Video
python scripts/predict.py --source construction_video.mp4

# Folder
python scripts/predict.py --source path/to/images

# Custom checkpoint or threshold
python scripts/predict.py --source test.jpg --weights path/to/best.pt --conf 0.35
```

Outputs are saved under `results/predictions/predict/`. For video sources, Ultralytics processes and annotates detections frame by frame and writes an output video.

## 6. Launch the demo

```bash
streamlit run app/app.py
```

To use a checkpoint in another location:

```bash
EXCAVATOR_WEIGHTS=/absolute/path/to/best.pt streamlit run app/app.py
```

Upload an image, adjust the confidence threshold, and view annotated detections plus their confidence values.

## Results

Populate this table from `results/evaluation/test/metrics.json` after the full Kaggle run. Metrics are intentionally not fabricated before training.

| Model | AP@50 | mAP@50-95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| YOLO11n | TBD | TBD | TBD | TBD |

Add representative files from `results/predictions/` here after evaluation:

```markdown
![Example excavator detection](figures/example_prediction.jpg)
```

## Reproducibility checklist

- Confirm class names, annotation format, and split counts with `inspect_dataset.py`.
- Visually verify the generated boxes.
- Keep `data/processed/manifest.json` with the run artifacts.
- Complete the one-epoch smoke test before the 50-epoch run.
- Evaluate only `best.pt` against the held-out `test` split.
- Record package/GPU information from the first Kaggle notebook cell.
- Run conversion tests with `python -m pytest` (install `requirements-dev.txt`).

## Troubleshooting

- **No excavator class verified:** inspect the reported class names. For nameless YOLO exports only, pass a class ID after checking its class map.
- **Annotations match few images:** preserve the extracted archive hierarchy and check the warnings from `inspect_dataset.py`.
- **CUDA unavailable:** enable the Kaggle GPU accelerator. For a local functionality-only check, explicitly pass `--device cpu --allow-cpu`.
- **Out of memory:** reduce `--batch` (for example, to 8 or 4). Keep `--imgsz 640` for the comparable full run.

## References

- [MOCS Kaggle dataset](https://www.kaggle.com/datasets/xiaopan9802/mocs-dataset)
- [MOCS paper: Dataset and benchmark for detecting moving objects in construction sites](https://doi.org/10.1016/j.autcon.2020.103482)
- [Ultralytics documentation](https://docs.ultralytics.com/)
