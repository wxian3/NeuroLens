# Neural Lens Modeling

This repository contains an implementation of the Neural Lens Modeling paper. The implementation includes keypoint detection, dataset generation, and training networks.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Generation](#dataset-generation)
- [NeuroLens Optimization](#training-networks)
- [License](#license)

## Introduction

The Neural Lens Modeling project aims to simulate realistic lens blur effects using neural networks. By training the networks on a dataset of blurred and sharp images, the models learn to predict lens blur parameters based on input images. The implementation provides tools for keypoint detection, dataset generation, and training the networks.

- Keypoint detection: The implementation includes our method to detect keypoints in images and to generate new keypoint patterns to be detected more accurately.
- Dataset generation: Tools are provided to generate a dataset of input images and corresponding ground truth lens parameters. This dataset can be used to train the neural networks.
- Training networks: The implementation includes network architectures and training scripts to train the neural networks using the capture or generated dataset.

## Requirements

To use the Neural Lens Modeling implementation, you need the following dependencies:

- Python (version 3.7 or higher)
- TensorFlow (version 2.4 or higher)
- OpenCV (version 4.2 or higher)
- NumPy (version 1.19 or higher)
- Matplotlib (version 3.3 or higher)

You can install the required Python packages using pip:

```
pip install tensorflow opencv-python numpy matplotlib
```

## Installation

First clone this repository to your local machine:

```
git clone https://github.com/wxian3/neuroLens.git
```

## Dataset Generation

To generate a dataset for training the neural networks, follow these steps:

1. Prepare a set of color images that will be used as input.
2. Apply lens distortion effects to the distort images using various lens parameters.
3. Save the distorted images along with the corresponding lens parameters (e.g., lens distortion, focal distance, aperture diameter) in a suitable format (e.g., CSV, JSON).

The dataset generation process requires the keypoint detection module to accurately estimate the blur parameters.
```
cd SynLens
python import_lenses.py
```

## Optimizing NeuroLens for Calibration

To train the neural networks using the generated dataset, follow these steps:

1. Load the dataset of blurred images and blur parameters (JSON files).
2. Split the dataset into training and validation sets.
3. Run calibration.

```
cd calibration/scripts
python calibrate.py --json_path path_to_input_json --output_path path_to_output
```
Refer to the provided training scripts and network architectures in the implementation for more details.

## Keypoint Detection and Marker Optimization
```
cd calibration
python keypoint_detection.py --input_path path_to_input_images --output_path path_to_output
python scripts/optimize_marker.py --output_path path_to_output
```
This command will process the input image and output keypoint detection results (correspondences in JSON). The resulting output will be saved at the specified output path.

## License

The Neural Lens Modeling implementation is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes.
