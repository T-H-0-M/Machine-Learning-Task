# Spiral-Classification

This program demonstrates the training and evaluation of Support Vector Machine (SVM) and Neural Network models for classification tasks. It includes modules for dataset generation, SVM training, and Neural Network training.

## Usage

1. Install the required dependencies by running `pip install -r requirements.txt`.
2. Run the `main.py` file to execute the program.

## Modules

The program consists of the following modules:

1. `dataset_generation.py`: Defines the `Data_Gen` class for generating and retrieving datasets from CSV files.

2. `neural_network.py`: Defines the `ANN` class, a multi-layered neural network implemented using the `torch.nn.Module` package.

3. `nn_training.py`: Defines the `Train` class for training and evaluating a neural network model. It uses the PyTorch library for training and evaluation.

4. `svm.py`: Defines the `SVM` class for training and evaluating a Support Vector Machine (SVM) model. It utilizes the scikit-learn library for SVM training and evaluation.

5. `main.py`: The main script that demonstrates the usage of the SVM and Neural Network models. It imports the necessary modules, creates instances of the models, and performs training and evaluation on the provided datasets.

## References

The following references were used in the code comments and adaptations:

- UON COMP3330 Labs, Week 3: The code for plotting the dataset in SVM training and generating a generalization graph in SVM evaluation was adapted from the UON COMP3330 Lab materials.

- UON COMP3330 Labs, Week 4: The code for training and evaluating the neural network model, as well as generating a learning curve graph, was adapted from the UON COMP3330 Lab materials.

- scikit-learn documentation: The scikit-learn library was used for splitting the data, calculating performance metrics (e.g., accuracy, confusion matrix), and generating classification reports. The official scikit-learn documentation was referenced for usage instructions and parameter details.

- PyTorch documentation: The PyTorch library was used for implementing and training the neural network model. The official PyTorch documentation was referenced for usage instructions and API details.

## Author

This program was developed by Thomas Bandy.

