# CPFANet: Contextual Perception Feature Aggregation-Based Transformer for Polyp Segmentation

## Step 1: Dataset Preparation
Ensure that the dataset is stored under `./data/polypSegDataset` and follows the dataset division as per the PraNet standard. The training dataset should be placed under `./TrainDataset`, with image paths in `./TrainDataset/images` and label paths in `./TrainDataset/masks`. The testing dataset should be placed under `./TestDataset`, containing five test subfolders, each of which should have `/images` and `/masks` folders.

## Step 2: Pretrained Weights
Check that the `pvt_v2_b2.pth` file is placed in the `./pretrained_weights/` directory.

## Step 3: Training and Testing
Once the above conditions are met, run the `train_and_test.py` script to train and test CPFANet located in `./lib` (please note some code may not be uploaded yet, as it will be uploaded after the peer review process). Training logs will be saved in the `./logs` directory, and the trained model weights will be saved in the `./snapshots` directory. During testing, performance metrics will be calculated and output using `metrics.py` located in `./utils`. The predicted segmentation maps will be saved in the `./result_maps/MODEL_NAME` directory.

## Step 4: Custom Model Evaluation
If you have your own model code and weights, place the code in the `./lib` directory and the weights in `./snapshots/MODEL_NAME`. After debugging the code in `model_evaluate.py`, run it to obtain performance metrics and result maps on the test set.

## Citation
If you use the code in your work, please kindly cite it as follows:

```bibtex
@misc{N/A,  
  authors = {Li et al.},  
  title = {CPFANet: Contextual Perception Feature Aggregation-Based Transformer for Polyp Segmentation},  
  year = {2025},  
  journal = {N/A (Work not yet published, please check back for updates)}  
}
