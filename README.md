# ASL Alphabet Image Classification

This project builds a convolutional neural network with TensorFlow and Keras to classify images of American Sign Language (ASL) alphabet signs.

## Overview

The notebooks cover dataset preparation, image preprocessing, augmentation, model training, and evaluation.

- [Explore sample images](notebooks/images.ipynb)
- [Review dataset preparation](notebooks/datacleaning.ipynb)
- [Review preprocessing and training](notebooks/load_preprocess.ipynb)

## Dataset

The training and test images are separate from this repository. The notebooks look for local datasets under `data/raw/` and `data/processed/`. Check the path variables in each notebook before running its cells.

## Setup

```bash
git clone https://github.com/ChahineDerbali/Sign-Language-Project.git
cd Sign-Language-Project
```

See [requirements.txt](requirements.txt) for the project's Python dependencies. The webcam, Flask, and Streamlit demos reference a model file at a machine-specific path and need additional setup before they can run from a fresh clone.

## License

See [LICENCE](LICENCE).
