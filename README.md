# Image-Captioning

## Setup

1. Clone this repository

2. Download and extract MSCOCO14 Dataset with annotations

```
chmod +x download.sh
./download.sh
```

3. Install pytorch from https://pytorch.org/

4. Install package requirements
```pip install -r requirements.txt```

5. Install COCO Python API

```
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
```

6. Build vocab

```python build_vocab.py```

## Training the model

1. To run the train model script, `python3 train.py`

## Testing the model

1. Set the `checkpoint_path` variable in `test_model.py` to the path of the saved model in `./models/` folder

2. To run the test model script, `python3 bleu_score.py`

## Running the GUI

1. Install app package requirements `pip install -r app_requirements.txt`
2. To start the GUI interface, cd into flaskapp and run `flask run`
3. To predict a sample image, you can use the `sample_image.jpg` provided.
4. The uploaded image along with its predicted caption and attention plot will be shown.

## Implementation Details
Refer to report.pdf for an explanation of the code and implementation.

