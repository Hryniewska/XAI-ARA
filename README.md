# XAI-ARA

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run iNNvestigate:
```bash
jupyter notebook
```
and go to innvestigate and choose 'innvestigate_tested_solutions.ipynb'


To run LIME:
```bash
python3 lime_ara.py
```

To run contours detection:
```bash
cd data_augmentation
g++ thresholding_image_segmentation.cpp -o contours `pkg-config --cflags --libs opencv`
./contours <PATH_TO_IMAGE>
cd ..
```

