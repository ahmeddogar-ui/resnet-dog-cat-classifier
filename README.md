# Dog vs Cat Classifier

Binary image classifier built with PyTorch. Uses a pretrained ResNet-18 with transfer learning to classify images as dog or cat. Served via a Flask REST API.

## How it works

ResNet-18 is pretrained on ImageNet. All layers are frozen except the final fully connected layer, which is replaced and trained on the Kaggle Dogs vs Cats dataset. Only the classifier head is trained — the convolutional layers act as a fixed feature extractor.

## Project structure

```
├── app.py                  # Flask app entry point
├── train_model.py          # Training script
├── image_organiser.py      # Organises Kaggle dataset into train/val splits
├── visualiser.py           # Visualises training batches
├── routes/
│   └── DOGCAT.py           # POST /predict endpoint
├── services/
│   └── predictor.py        # Model loading and inference logic
├── models/
│   └── dog_cat_model.pth   # Trained weights (not included in repo)
└── data/                   # Training data (not included in repo)
```

## Setup

```bash
pip install torch torchvision flask pillow
```

## Training

Download the [Kaggle Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) and place the images in `data/raw/`. Then:

```bash
python image_organiser.py
python train_model.py
```

Trained weights will be saved to `models/dog_cat_model.pth`.

## Running the API

```bash
python app.py
```

## Usage

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@/path/to/image.jpg"
```

Response:

```json
{"prediction": "cat"}
```

## Stack

- PyTorch
- ResNet-18 (pretrained on ImageNet)
- Flask
- Kaggle Dogs vs Cats dataset
