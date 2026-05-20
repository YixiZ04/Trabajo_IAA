"""
    The datafetcher from MNIST is found here, and some util objects as well.
"""




from medmnist import INFO
import medmnist

import numpy as np

import torchvision.transforms as transforms


class MedMNISTDataFetcher:
    """
        Fetches a medMNIST dataset and convets it to a suitable input format.
        Inputs:
            label: e.g. "pathmnist".
            size: the image pixels (size x size). Default is 28.
            download: whether to download the dataset. Default is True
    """
    def __init__ (self, label, size=28, download=True) -> None:
        self.possible_sizes = (28, 64, 184, 224)

        if label not in INFO.keys():
            raise ValueError(f"Expected values: {list(INFO.keys())}. Got `{label}` instead.")


        if size not in self.possible_sizes:
            raise ValueError(f"Expected sizes: {self.possible_sizes}, but got {size} instead.")
        self.label = label
        self.label_dict = INFO[self.label] ["label"]
        self.size = size

        self.info = INFO[self.label]
        self.DataClass = getattr(medmnist, self.info["python_class"])

        self.channels = INFO[self.label]["n_channels"]
        self.download = download

    def to_array (self) -> tuple [np.ndarray, np.ndarray, np.ndarray,np.ndarray]:
        """
            Converts images to 2D arrays and labels to 1D array.
        """
        print (f"Fetching data for {self.label}...")
        train_data = self.DataClass(split="train",
                                    download=self.download,
                                    size=self.size)
        test_data = self.DataClass(split="test",
                                   download=self.download,
                                   size=self.size)

        train_data_count = len(train_data)
        test_data_count = len(test_data)
        # Images 2D array
        train_image_array = np.array([np.array(train_data[i][0]) for i in range(train_data_count)])
        test_image_array = np.array([np.array(test_data[i][0]) for i in range(test_data_count)])
        flat_train_image_array = np.array([ image.flatten() for image in train_image_array ])
        flat_test_image_array = np.array([ image.flatten() for image in test_image_array ])

        #Labels 1D array
        train_label_array = np.array([ np.array(train_data[i][1]) for i in range(len(train_data))])
        test_label_array = np.array([ np.array(test_data[i][1])for i in range(len(test_data))])

        print ("Done")
        return flat_train_image_array, flat_test_image_array, train_label_array.flatten(), test_label_array.flatten()

    def to_torch_dataset(self) -> tuple:
        """
            Converts the inputs to a PathMNIST object. This can directly be converted to a torch DataLoader.
        """
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_dataset = self.DataClass(split="train",
                                       transform=data_transform,
                                       download=self.download,
                                       size=self.size)
        test_dataset = self.DataClass (split="test",
                                       transform=data_transform,
                                       download=self.download,
                                       size=self.size)
        print(train_dataset)
        print("===================")
        print(test_dataset)

        return train_dataset, test_dataset

class Number2LabelMapper:
    """
        Maps the numbers to the corresponding label of a dataset, e.g., PathMNIST.
        Inputs:
                label_array: the predicted or the original array.
                label: the label of the dataset, an error message will be raised if wrong one is given (e.g., "pathmnist").
    """
    def __init__(self,  label) -> None:
        if label not in INFO.keys():
            raise ValueError(f"Expected values: {list(INFO.keys())}. Got `{label}` instead.")
        self.label = label
        self.label_dict = INFO[self.label] ["label"]

    def map_label (self, label_array) -> np.ndarray:
        res_array = np.array([ self.label_dict[str(num_label)] for num_label in label_array ])
        return res_array

class PredictionInfoExtracter:
    """
        This extracts the information of the prediction.

    """
    def __init__ (self) -> None:
        pass

    def extract (self, X_test, y_test, y_pred) -> tuple:
        """
            Extracts 5 arrays:
                - The wrongly predicted inputs.
                - The true labels of those inputs.
                - The correctly predicted inputs.
                - The labels of those inputs.
            These are needed for plotting the SOM correctly.

                - The labels fo those wrongly predicted.
            This is for image plotting.
        """
        wrong_pred_X = X_test [y_test != y_pred]
        true_label_wrong_pred_y = y_test [y_test != y_pred]
        true_pred_X = X_test [y_test == y_pred]
        true_pred_y = y_test [y_test == y_pred]
        pred_label_wrong_pred_y = y_pred [y_test != y_pred]

        return (np.array(wrong_pred_X),
                np.array(true_label_wrong_pred_y),
                np.array(true_pred_X),
                np.array(true_pred_y),
                np.array(pred_label_wrong_pred_y),)










if __name__ == "__main__":
    data_fetcher = MedMNISTDataFetcher("pathmnist")
    a,b,c,d = data_fetcher.to_array()
    label_mapper = Number2LabelMapper(d, "pathmnist")
    string_label_array = label_mapper.map_label()







