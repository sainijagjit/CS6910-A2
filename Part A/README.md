# CNN on iNaturalist Dataset

This detailed guide provides instructions for setting up, training, and evaluating a Convolutional Neural Network (CNN) on the iNaturalist dataset, focusing on classifying images into categories of natural species.

## Prerequisites

Ensure the following requirements are met before starting:

- Python 3.6 or later.
- GPU access (recommended for faster training).
- Internet connection for dataset download and Python package installation.

## Installation Steps

1. **Clone the repository:** Clone the project repository to your local machine using the command below. Replace `<repository-url>` with the actual URL of the repository.
   ```shell
   git clone https://github.com/sainijagjit/CS6910-A2.git
   ```
2. **Navigate to the project directory:** Change to the project directory using the command line.
   ```shell
   cd Part A
   ```

## Dataset Preparation

Download and prepare the iNaturalist dataset by following these steps:

1. **Download the dataset** from [the iNaturalist competition link](https://github.com/visipedia/inat_comp) or the dataset's official page.
2. **Copy dataset** to a directory in google drive, ensuring there are `train` and `val` folders for training and validation images, respectively.
3. **Replace zip path** with the link to your dataset

## Training the Model

To train the model, follow these steps:

1. **Configure the model and training parameters:** Open `Assignment_2_iNaturalist.ipynb` in a Jupyter Notebook or Google Colab. The configuration section allows you to set various hyperparameters such as `num_filters`, `batch_size`, `learning_rate`, etc.
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
- **Customizing Model Architecture:** You can customize the CNN architecture by modifying the `Net` class in the notebook. This allows for experimentation with different layer configurations, activation functions, and other architectural changes.
- **Data Augmentation:** Experiment with different data augmentation techniques by adjusting the `augmentation` section within the `DataModule` class. This can potentially improve model robustness and accuracy.
