# Face Recognition Example

This project contains small examples using the Python `face_recognition` library.

## Required image

Add an image named `test.jpg` to the `img` directory:

```text
img/test.jpg
```

The image should contain at least one clearly visible face. All scripts use this file as both the known and unknown image.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install face-recognition numpy
```

## Run

From the project directory, run one of the examples:

```bash
python main.py
python main2.py
python withTime.py
```

`main2.py` and `withTime.py` save or load the generated face encoding from `biden_encoding.npy` to avoid recalculating it every time.
