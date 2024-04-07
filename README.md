
# CS6910 Assignment 1
## Submitted By: Jagjit Singh NS24Z060

### Link to the project report:

[[https://wandb.ai/ns24z060/CS6910%20-%20Assignment%201/reports/CS6910-Assignment-1--Vmlldzo3MjE3NzEy]([https://api.wandb.ai/links/ns24z060/mptxog54](https://api.wandb.ai/links/ns24z060/mptxog54))]([https://wandb.ai/ns24z060/CS6910%20-%20Assignment%201/reports/CS6910-Assignment-1--Vmlldzo3MjE3NzEy?accessToken=5izqb3kfaj6wygetx2tpseeeofi05yn6nwipp6n1f06ubcuh910iiffc722uywaf](https://api.wandb.ai/links/ns24z060/mptxog54))


### Instructions to train and evaluate the models:

1. To train a neural network model for image classification on the Fashion-MNIST dataset using categorical cross-entropy loss, you can utilize the notebook named : **Assignment_1_MNIST_wandb.ipynb**.

   a.  In this notebook, you can train the model using the best values for hyperparameters obtained from the wandb sweeps functionality by avoiding the execution of cells in the section titled "Model Training". Simply run all the other cells of the notebook to train the model. The final model will be trained on the full training set, and evaluation will be performed on the test set.
   b. In order to run the hyperparameter search, run the full notebook.
   

2. Alternatively, you can train the model with your custom hyperparameters using the **train.py** script by passing appropriate arguments to update the model. Evaluation of the model will be calculated at the end of training.

| Argument             | Description                                                                                         |
|----------------------|-----------------------------------------------------------------------------------------------------|
| -wp, --wandb_project| Project name used to track experiments in Weights & Biases dashboard                                 |
| -we, --wandb_entity | Wandb Entity used to track experiments in the Weights & Biases dashboard.                            |
| -d, --dataset        | Choices: ["mnist", "fashion_mnist"]                                                                |
| -e, --epochs         | Number of epochs to train neural network.                                                          |
| -b, --batch_size     | Batch size used to train neural network.                                                           |
| -l, --loss           | Choices: ["mean_squared_error", "cross_entropy"]                                                   |
| -o, --optimizer      | Choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]                                    |
| -lr, --learning_rate | Learning rate used to optimize model parameters.                                                   |
| -m, --momentum       | Momentum used by momentum and nag optimizers.                                                      |
| -beta, --beta        | Beta used by rmsprop optimizer                                                                     |
| -beta1, --beta1      | Beta1 used by adam and nadam optimizers.                                                           |
| -beta2, --beta2      | Beta2 used by adam and nadam optimizers.                                                           |
| -eps, --epsilon      | Epsilon used by optimizers.                                                                        |
| -w_d, --weight_decay | Weight decay used by optimizers.                                                                   |
| -w_i, --weight_init  | Choices: ["random", "Xavier"]                                                                      |
| -nhl, --num_layers   | Number of hidden layers used in feedforward neural network.                                         |
| -sz, --hidden_size   | Number of hidden neurons in a feedforward layer.                                                   |
| -a, --activation     | Choices: ["identity", "sigmoid", "tanh", "ReLU"]                                                    |




```python

```
