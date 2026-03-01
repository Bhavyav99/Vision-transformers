import os
import cv2 as cv
import numpy as np


class MBH_dataset:
    """
    Custom dataset class for loading MBH dataset images.

    Args:
        data_dir (str): Path to the directory containing image data.
        label_file (str): Path to the file containing image labels.
        transform (callable): Optional transform to be applied to the image.

    Attributes:
        data_dir (str): Path to the directory containing image data.
        label_file (str): Path to the file containing image labels.
        text (list): List containing lines from the label file.
        image_paths (list): List containing paths to images.
        labels (list): List containing image labels.
        transform (callable): Optional transform to be applied to the image.
    """
    def __init__(self, data_dir, label_file, transform):
        self.data_dir = data_dir
        self.label_file = label_file
        with open(self.label_file) as f:
            self.text = [line for line in f]
        self.image_paths = []
        self.labels = []
        self.transform = transform

    def load_data(self):
        """
        Load image paths and labels from the label file.
        """
        for lines in self.text:
            temp = str(lines).split()
            self.img_name = temp[0]
            self.img_folder = temp[1]
            self.label = temp[2]
            image_path = self.data_dir + "total_data/" + str(self.img_folder) + "/" + self.img_name
            self.image_paths.append(image_path)
            self.labels.append(int(self.label))

    def __len__(self):
        """
        Get the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.text)

    def __getitem__(self, index):
        """
        Get image and label at the specified index.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        image_path = self.image_paths[index]
        image = cv.imread(image_path)
        label = self.labels[index]
        # Convert BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        #image = np.transpose(image, (2, 0, 1))
        # print(image.shape)

        if self.transform:
            # print("error")
            # print(image.shape)
            # image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)

        return image, label