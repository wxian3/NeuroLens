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

We introduce NeuroLens, a new method that accurately optimizes for camera lens distortion using an invertible neural network. Compared to standard calibration packages, NeuroLens improves the accuracy and generality of pre-capture camera calibration and allows for refinement during 3D reconstruction.

[Paper]: https://arxiv.org/abs/2304.04848
[Project website]: https://neural-lens.github.io

This is a PyTorch implementation provides tools for keypoint detection, dataset generation, and training the networks.

## Requirements

To use the Neural Lens Modeling implementation, you need Python (version 3.7 or higher). Python environment including the required dependencies. This requires an NVIDIA GPU and a CUDA installation with version 11.1 - 11.3.

You can install the required Python packages using pip:

```
pip install -r requirements.txt
```

## Installation

First clone this repository to your local machine:

```
git clone https://github.com/wxian3/neuroLens.git
```

## Dataset Generation

To generate a dataset for training the neural networks, follow these steps:

1. Prepare a keypoint pattern as input and render it on each view.
2. Apply lens distortion effects to the target markers using various lens parameters.
3. Save the distorted images along with the corresponding lens parameters (e.g., lens distortion, focal distance, aperture diameter) in JSON.

```
cd SynLens
python import_lenses.py --input_path path_to_target
```

## Optimizing NeuroLens for Calibration

To train the neural networks using the generated or captured dataset, follow these steps:

1. Load the sequence of distorted images and keypoint correspondences from JSON files.
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
python scripts/optimize_marker.py

python keypoint_detection.py --input_path path_to_input_images --output_path path_to_output
```
This command will process the input image and output keypoint detection results (correspondences in JSON). The resulting output will be saved at the specified output path.

## License

The Neural Lens Modeling implementation is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes.
