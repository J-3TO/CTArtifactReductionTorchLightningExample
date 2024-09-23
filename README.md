# TorchLightningExample

Example training script to train the U-Net for artifact reduction using the torch [lightning framwork](https://lightning.ai/docs/pytorch/stable/).

## Overview

- networks.py/utils.py - The model and dataloader/utility functions are defined here.
- test/val/train.csv - Dataframes with the filenames of the different data splits
- test_pipeline.ipynb - Notebook to test the training pipeline and do some hyperparemeter tuning.
- train.py - Script which contains the actual training loop. Hyperparameters can be passed as command-line arguments
- run.sh - Executes train.py with command-line arguments; Makes it easy to test several hyperparameters.

## How to use

- Modify networks.py/utils.py until it fits the requirements of your project.
- Test the pipeline with test_pipeline.ipynb
- adjust train.py
- Execute train.py directly or via run.sh
- Monitor training via csv log or tensorboard
