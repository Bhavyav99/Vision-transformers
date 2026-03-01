import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import json
import glob
from tqdm import tqdm
from sklearn import svm
import supervision as sv

class MBH_dataset:
    """Custom dataset class for loading MBH dataset."""
    def __init__(self, data_dir, label_file, transform):
        """
        Initialize MBH_dataset class.

        Args:
            data_dir (str): Directory containing the dataset.
            label_file (str): Path to the label file.
            transform: Transformations to be applied to the data.
        """

        self.data_dir = data_dir
        self.label_file = label_file
        with open(self.label_file) as f:
            self.text = [line for line in f]
        self.image_paths = []
        self.labels = []
        self.transform = None
        self.ok_image_files = []  # Initialize as instance attributes
        self.nok_image_files = []  # Initialize as instance attributes

    def load_data(self):
        """Load data from the dataset."""
        for lines in self.text:
            temp = str(lines).split()
            self.img_name = temp[0]
            self.img_folder = temp[1]
            self.label = temp[2]
            image_path = self.data_dir + "total_data/" + str(self.img_folder) + "/" + self.img_name
            self.image_paths.append(image_path)
            self.labels.append(int(self.label))

            if int(self.label) == 0:
                self.ok_image_files.append(image_path)
            else:
                self.nok_image_files.append(image_path)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        """
        Get item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        label = self.labels[index]


        if self.transform:
            image = self.transform(image)


        return self.ok_image_files, self.nok_image_files



dataset = MBH_dataset(data_dir = '/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                     label_file = "/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - H/train.txt", transform = None)

dataset.load_data()  # Load the data
#print(dataset.ok_image_files)  # Access ok_image_files
ok_dict = {element: 'okay' for element in dataset.ok_image_files}
nok_dict = {element: 'not okay' for element in dataset.nok_image_files}


## adding both the images to one dictionary
ok_dict.update(nok_dict)
labels = ok_dict

files = labels.keys()

# Load DINOv2 model
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14.to(device)

transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])


def load_image(img: str) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.

    Args:
        img (str): Path to the image file.

    Returns:
        torch.Tensor: Tensor representing the image.
    """
    img = Image.open(img)

    transformed_img = transform_image(img)[:3].unsqueeze(0)

    return transformed_img



def compute_embeddings(files: list) -> dict:
    """
    Create an index that contains all of the images in the specified list of files.

    Args:
        files (list): List of file paths.

    Returns:
        dict: Dictionary containing embeddings for each image.
    """
    all_embeddings = {}

    with torch.no_grad():
        for i, file in enumerate(tqdm(files)):
            embeddings = dinov2_vits14(load_image(file).to(device))

            all_embeddings[file] = np.array(embeddings[0].cpu().numpy()).reshape(1, -1).tolist()

    with open("part_b.json", "w") as f:
        f.write(json.dumps(all_embeddings))

    return all_embeddings


embeddings = compute_embeddings(files)


# Train SVM classifier
clf = svm.SVC(gamma='scale')
y = [labels[file] for file in files]
embedding_list = list(embeddings.values())
clf.fit(np.array(embedding_list).reshape(-1, 384), y)


test_dataset = MBH_dataset(data_dir = '/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                     label_file = "/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - H/model_test.txt", transform = None)



test_dataset.load_data()

# Define labels for the test dataset
test_ok_dict = {element: 'okay' for element in test_dataset.ok_image_files}
test_nok_dict = {element: 'not okay' for element in test_dataset.nok_image_files}


test_ok_dict.update(test_nok_dict)
labels_test = test_ok_dict




labels_test_key = list(labels_test.keys())

predicted_test_key = []


# Make predictions for the test dataset
for i in range(len(labels_test_key)):
    input_file = labels_test_key[i]

    new_image = load_image(input_file)


    with torch.no_grad():
        embedding = dinov2_vits14(new_image.to(device))

        prediction = clf.predict(np.array(embedding[0].cpu()).reshape(1, -1))
        #print("Predicted class: " + prediction[0])
        predicted_test_key.append(prediction[0])

TP = 0
TN = 0
FP = 0
FN = 0

labels_test_val = list(labels_test.values())


for i in range(len(labels_test_val)):
    if (labels_test_val[i] == "not okay" and predicted_test_key[i] == "not okay"):
        TP+=1
    elif labels_test_val[i] == "okay" and predicted_test_key[i]=="okay":
        TN+=1
    elif labels_test_val[i]=="okay" and predicted_test_key[i]=="not okay":
        FP+=1
    elif labels_test_val[i]=="not okay" and predicted_test_key[i]=="okay":
        FN+=1


print("TP:", TP, "\tFP:", FP, "\nFN:",FN , "\tTN:", TN)
