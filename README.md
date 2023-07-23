# Text Generation Nietzsche

This repository contains Python scripts for a text generation model trained on the works of Friedrich Nietzsche. The model is implemented using TensorFlow and Keras, leveraging Long Short-Term Memory (LSTM) networks. The model generates new text based on the writing style of Nietzsche.

## Setup and Requirements

The application requires the following software and libraries:

- Python 3.6 or later
- TensorFlow 2.x
- Keras
- CUDA (optional, for GPU support)

To install the necessary libraries, you can use pip:

```bash
pip install tensorflow keras
```

Please note: If you plan to use a CUDA-enabled GPU, make sure to install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn) according to the instructions provided by NVIDIA.

## Usage

After installing the dependencies, you can run the training script:

```bash
python textgenerationnietsche.py
```

After training, you can run the testing script:

```bash
python texttestniet.py
```

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.
