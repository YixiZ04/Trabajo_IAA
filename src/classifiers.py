"""
    All the classfiers are found in this Script. Here 4 classfiers are coded as objects:
        1. RandomClassifier
        2. NaiveBayesClassifier
        3. BaggingNaiveBayesClassifier
        4. TorchCNNClassifier
"""

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from src.utils import MedMNISTDataFetcher



## Parameters

RANDOM_SEED = 42

## Objects

class RandomClassifier:
    """
        Assumes the distribution of each class is equal, so the prediction is random only depending on the number of classes.
        Inputs: 2D array (Actually anything)
        Outputs: Predicted array.
    """
    def __init__(self,n_classes,random_seed=RANDOM_SEED):
        self.random_seed = random_seed

        self.n_classes = n_classes
        self.prediction = []

    def predict (self, X_test) -> np.ndarray:
        np.random.seed(self.random_seed)
        prediction_length = len(X_test)
        self.prediction = np.array([ np.random.randint(0,self.n_classes) for _ in range( prediction_length) ])

        return self.prediction

    def accuracy(self, y_test) -> float:
        acc = len(y_test[self.prediction==y_test]) / len(y_test)
        return acc
class NaiveBayesClassifier:
    """
        To implement Naive-Bayes Algorithm based on frequencies. Assumed the independency of pixels-classes (Not True at all, but ok).
        Based on log-likelihood, as based on PDF function would need 3 nested loops, which is too expensive and slow.
    """
    def __init__ (self, random_seed=RANDOM_SEED, smoothing=1e-9):
        """
            The smoothing parameter is used to avoid any dividing by 0 or log(0).
        """

        self.random_seed = random_seed
        self.smoothing = smoothing
        self.classes = []
        self.n_classes = 0

        self.mean_vector = []
        self.std_vector = []
        self.frequencies = []
        self.accuracy = 0.0

    def fit(self, scaled_X_train,y_train) -> None:
        """
            Fit the training data to get the mean and standard deviation of the training data.
        """
        self.classes = np.unique(y_train)
        self.n_classes = np.int64(len(np.unique(y_train)))
        for class_ in self.classes:
            temp_X = scaled_X_train[y_train == class_]
            self.mean_vector.append(temp_X.mean(axis=0))
            self.std_vector.append(temp_X.std(axis=0) + self.smoothing)
            self.frequencies.append(len(temp_X) / len(scaled_X_train))

        self.mean_vector = np.array(self.mean_vector)
        self.std_vector = np.array(self.std_vector)
        self.frequencies = np.array(self.frequencies)
        self.prediction = []

    def _log_gaussian_lh (self, image, mean_vector, std_vector) -> np.float64:
        """
            Computes the likelihood in vectorized form
        """
        log_lh = -0.5 * np.log(2.0 * np.pi * std_vector**2) - ((image - mean_vector) ** 2) / (2.0 * std_vector**2)
        return np.float64(log_lh)

    def _predict_single_entree (self, image) -> np.int64:
        """
            This internal function predicts one sole entree.
            The implementation is based on log-likelihood.
        """
        log_likelihood_array = np.zeros(self.n_classes)
        for class_index, class_ in enumerate(self.classes):
            log_prior = np.log(self.frequencies[class_index] + self.smoothing)
            log_likelihood = np.sum(self._log_gaussian_lh(image, self.mean_vector[class_index], self.std_vector[class_index]))
            log_likelihood_array[class_index] = log_likelihood + log_prior

        most_likely_class_index = np.argmax(log_likelihood_array)
        return np.int64(self.classes[most_likely_class_index])

    def predict (self, scaled_X_test, y_test) -> np.ndarray:
        """
            Makes the prediction on test set.
        """
        for image in scaled_X_test:
            prediction = self._predict_single_entree(image)
            self.prediction.append(prediction)
        self.accuracy = len(y_test[self.prediction == y_test]) / len(y_test)
        return np.array(self.prediction)

    def fit_predict (self,scaled_X_train, y_train, scaled_X_test, y_test) -> np.ndarray:
        """
            Fit the training data and makes the prediction.
        """
        self.fit(scaled_X_train, y_train)
        return self.predict(scaled_X_test, y_test)




class BaggingNaiveBayesClassifier:
    """
        Builds a bagging classifier using the NaiveBayesClassifier as the base classifier.
    """

    def __init__ (self,
                  n_estimators = 50,
                  random_seed=RANDOM_SEED,
                  ):
        self.random_seed = random_seed
        self.n_estimators = n_estimators
        self.estimators = []
        self.classes = []
        self.n_classes = 0
        self.accuracy = 0.0
        self.prediction = []
        self.rng = np.random.default_rng(self.random_seed)

    def fit (self,scaled_X_train, y_train) -> None:
        """
            Random sampling + get mean and standard deviation of the training data.
        """
        training_indices = np.arange(len(scaled_X_train))
        for _ in range(self.n_estimators):
            random_sampled_indices = self.rng.choice (training_indices,
                                             size = np.int64(np.ceil(0.5*len(scaled_X_train))),
                                             replace=True,
                                             shuffle=True)
            X_sampled = scaled_X_train[random_sampled_indices]
            y_sampled = y_train[random_sampled_indices]

            temp_model = NaiveBayesClassifier()
            temp_model.fit(X_sampled, y_sampled)
            self.estimators.append(temp_model)


    def predict (self, scaled_X_test, y_test) -> np.ndarray:
        """
            Makes prediction on test set.
        """
        prediction_array = []
        for estimator in self.estimators:
            temp_preds = estimator.predict(scaled_X_test, y_test)
            prediction_array.append(temp_preds)

        prediction_array=np.array(prediction_array)
        for image_prediction in prediction_array.T:
            prediction = np.argmax(np.bincount(image_prediction))
            self.prediction.append(prediction)
        self.accuracy = len(y_test[self.prediction == y_test]) / len(y_test)
        return np.array(self.prediction)

    def fit_predict (self, scaled_X_train, y_train, scaled_X_test, y_test) -> np.ndarray:
        """
            Fit the training data and makes the prediction.
        """
        self.fit(scaled_X_train, y_train)
        return self.predict(scaled_X_test, y_test)

class _Net(nn.Module):
    """
        Initiates a CNN architecture:
            4 Convolutional layer:
                 padding = 1 and kernel size = 3.
                 BatchNormalization
                 ReLU as the activation function
            MLP:
                Flatten
                Input layer
                2 hidden layers
                Output layer
        Note: Only works if input size = 28x28; else the numbers defined inside the _Net should be changed.
    """
    def __init__ (self,in_channels, n_classes):
        super(_Net,self).__init__()
        self.network = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Second convolutional layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # Third convolutional layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            # MLP
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=n_classes),
        )
    def forward(self, x):
        return self.network(x)


class TorchCNNClassifier:
    """
        Builds and trains a CNN for MedMNIST classfier.
        The architecture is specified in the previous object.
        The optimizer is Adam and the loss function is CrossEntropyLoss.
        The device depends on if CUDA is available.
        The lr can be specified, but the default = 1e-4
        The num_epochs set to 40 by default, no EarlyStopping is implemented.
    """
    def __init__ (self, train_data, test_data, in_channels, n_classes, lr=1e-4, num_epochs=40):
        self.train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True)
        self.test_loader = data.DataLoader(test_data, batch_size=64, shuffle=False)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.loss_function = nn.CrossEntropyLoss()
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

        #Build the model
        self.model = _Net(self.in_channels, self.n_classes)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.prediction = []
        self.accuracy = 0.0

    def _training_model (self)-> None:
        """
            Contains the training process.
        """
        for epoch in range(self.num_epochs):
            self.model.train()
            pbar = tqdm(self.train_loader)
            pbar.set_description(f"Epoch {epoch + 1}")
            for image, label in pbar:
                self.optimizer.zero_grad()
                inputs = image.to(self.device)
                targets = label.to(self.device)
                outputs = self.model(inputs)
                targets = targets.to(self.device)
                targets = targets.squeeze().long()
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()

    def _predict_test (self) -> np.ndarray:
        """
            Contains the predicting process for test loader.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            pbar=tqdm(self.test_loader)
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                prediction = torch.argmax(outputs, dim=1)
                for pred, label in zip(prediction, labels):
                    self.prediction.append(pred.cpu().numpy())
                    total += 1
                    if pred == label:
                        correct += 1

        self.accuracy = correct / total

        return np.array(self.prediction)

    def fit_predict(self) -> np.ndarray:
        self._training_model()
        prediction = self._predict_test()
        return prediction





if __name__ == "__main__":
    data_fetcher = MedMNISTDataFetcher("pathmnist")
    # X_train, X_test, y_train, y_test = data_fetcher.to_array()
    # random_classifier = RandomClassifier (X_train, X_test, y_train)
    # randon_prediction = random_classifier.predict()
    # data_mapper = Number2LabelMapper (randon_prediction, "pathmnist").map_label()
    # naive_classifier = NaiveBayesClassifier()
    # naive_prediction2 = naive_classifier.fit_predict(X_train, y_train,X_test)
    # bagging_classifier = BaggingNaiveBayesClassifier()
    # bagging_prediction =bagging_classifier.fit_predict(X_train, y_train,X_test)
    #
    # torch_train_input, torch_test_input = data_fetcher.to_torch_dataset()
    # in_channels = data_fetcher.channels
    # n_classes = 9
    #
    # torch_cnn = TorchCNNClassifier(torch_train_input, torch_test_input, in_channels, n_classes)
    # torch_prediction = torch_cnn.fit_predict()
    #





