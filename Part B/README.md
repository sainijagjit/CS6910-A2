# Pretrained Model on iNaturalist Dataset

## Submitted By: Jagjit Singh NS24Z060

### Link to the project report:

[https://wandb.ai/ns24z060/CS6910%20-%20Assignment%201/reports/CS6910-Assignment-1--Vmlldzo3MjE3NzEy?accessToken=5izqb3kfaj6wygetx2tpseeeofi05yn6nwipp6n1f06ubcuh910iiffc722uywaf](https://api.wandb.ai/links/ns24z060/mptxog54)

This detailed guide provides instructions for setting up, training, and evaluating a Pretained model on the iNaturalist dataset, focusing on classifying images into categories of natural species.

## Prerequisites

Before you begin, ensure you have Python 3.6 or later installed, along with PyTorch, PyTorch Lightning, and other required libraries:

```bash
pip install torch torchvision torchmetrics pytorch-lightning
```

## Installation Steps

1. **Clone the repository:** Clone the project repository to your local machine using the command below. Replace `<repository-url>` with the actual URL of the repository.
   ```shell
   git clone https://github.com/sainijagjit/CS6910-A2.git
   ```
2. **Navigate to the project directory:** Change to the project directory using the command line.
   ```shell
   cd Part B
   ```
3. **Wandb Key:** Add wandb api key in collab secret with field name wandb_key.

## Dataset Structure

Your dataset should be organized in the following structure, with a separate folder for training and validation images. Each of these folders should contain subdirectories for each class of images:

```
path/to/your/dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   │   ...
└── val/
    ├── class1/
    ├── class2/
    │   ...
```

## Training the Model

To train the model, use the following command, specifying the path to your dataset and other training parameters as needed:

```bash
python train_evaluate.py --mode train --data_dir path/to/your/dataset --batch_size 64 --learning_rate 0.001 --epochs 10 --augment_data --batch_norm
```

### Training Arguments:

- `--mode` specifies the operation mode. Set to `train` for training.
- `--data_dir` sets the path to the dataset directory.
- `--batch_size` determines the batch size for training.
- `--learning_rate` sets the learning rate for the optimizer.
- `--epochs` specifies the number of epochs to train for.
- `--augment_data` enables data augmentation (flag).

## Evaluating the Model

For model evaluation, use the following command, providing the path to your saved model weights:

```bash
python train_evaluate.py --mode evaluate --data_dir path/to/your/dataset --weights_path path/to/your/model_weights.ckpt
```

### Evaluation Arguments:

- `--mode` should be set to `evaluate` for evaluation.
- `--data_dir` specifies the path to the dataset directory for evaluation.
- `--weights_path` sets the path to the saved model weights.

This flexible approach allows you to easily train and evaluate the CNN model with custom configurations.

## Dataset Preparation

Download and prepare the iNaturalist dataset by following these steps:

1. **Download the dataset** from [the iNaturalist competition link](https://github.com/visipedia/inat_comp) or the dataset's official page.
2. **Copy dataset** to a directory in google drive, ensuring there are `train` and `val` folders for training and validation images, respectively.
3. **Replace zip path** with the link to your dataset

## Training the Model (With WanDB)

To train the model, follow these steps:

1. **Configure the model and training parameters:** Open `Assignment_2_iNaturalist-Part B.ipynb` in a Jupyter Notebook or Google Colab. The configuration section allows you to set various hyperparameters such as `Pretrained Model`, `learning_rate`, etc.
2. **Integrate with Weights & Biases (Optional):** For tracking experiments, sign up for a Weights & Biases account and log in. Update the `wandb` configuration within the notebook to set your project and entity name. This step is optional but recommended for better experiment tracking.
3. **Adjust the `sweep_config`:** If you're planning to use Weights & Biases for hyperparameter tuning, adjust the `sweep_config` dictionary in the notebook to define the hyperparameter search space.
4. **Start the training process:** Run all cells in the notebook to start the training process. The training script will automatically split the data, train the model, and validate it on the validation dataset. The best model will be saved based on validation accuracy.

## Evaluation

After training, the model is evaluated on the test set. Follow these steps to perform evaluation:

1. **Load the best model:** The training script saves the best model based on validation performance. Load this model using PyTorch's `load_from_checkpoint` function.
2. **Prepare the test data:** Ensure the test data is loaded and preprocessed similarly to the training data. Use the `DataModule` class provided in the notebook for consistent data loading and preprocessing.
3. **Run the evaluation:** Use the `trainer.test` method to evaluate the loaded model on the test dataset. The script logs test accuracy and other relevant metrics for your analysis.

## Advanced Usage

- **Hyperparameter Tuning with Weights & Biases:** To perform hyperparameter tuning, use the Weights & Biases sweep functionality. Define your sweep configuration in the `sweep_config` section and start the sweep process by executing the relevant cell in the notebook. This process will automatically explore the hyperparameter space and log the results to your Weights & Biases project.
- **Data Augmentation:** Experiment with different data augmentation techniques by adjusting the `augmentation` section within the `DataModule` class. This can potentially improve model robustness and accuracy.
