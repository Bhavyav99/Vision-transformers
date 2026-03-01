import itertools
from torchvision import models
import torch.nn as nn
import torch
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
import os
import math
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_labels = ['Class 0', 'Class 1']



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
        #image = cv.resize(image, (224, 224))
        image = np.transpose(image, (2, 0, 1))
        # print(image.shape)

        if self.transform:
            # print("error")
            # print(image.shape)
            image = np.transpose(image, (1, 2, 0))
            image = self.transform(image)

        return image, label


class CustomModel(pl.LightningModule):
    """
    Custom PyTorch Lightning module for training a model.

    Args:
        model (torch.nn.Module): The neural network model.
        criterion: Loss criterion for training.
        optimizer: Optimizer for updating model parameters.
        num_epochs (int): Number of epochs for training.
        early_stop_patience (int): Patience for early stopping.
        transfer_learning (str): Strategy for transfer learning.
        train_transforms (torchvision.transforms.Compose): Transformations for training data.
        val_transforms (torchvision.transforms.Compose): Transformations for validation data.
        writer (torch.utils.tsensorboard.SummaryWriter): TensorBoard writer for logging.
    """
    def __init__(self, model,  criterion, optimizer, num_epochs, early_stop_patience, transfer_learning, train_transforms, val_transforms, writer):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.best_val_loss = float('inf')
        self.current_patience = 0
        self.unfreeze_layers = transfer_learning
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.writer = writer

        # Define your custom dataset and dataloaders
        # Assume you have a dataset class named CustomDataset
        # Prepare your datasets_in and data loaders
        train_dataset = MBH_dataset(data_dir='/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                                    label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - F/model_train.txt",
                                    transform=self.train_transforms)
        train_dataset.load_data()

        val_dataset = MBH_dataset(data_dir='/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                                  label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - F/model_val.txt",
                                  transform=self.val_transforms)
        val_dataset.load_data()

        test_dataset = MBH_dataset(data_dir='/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                                   label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - F/model_test.txt",
                                   transform=None)
        test_dataset.load_data()

        self.dataloaders = {
            'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
            'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4),
            'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        }

        if self.unfreeze_layers == "last":
            # Unfreeze only the final classification layer(s)
            for param in model.fc.parameters():
                param.requires_grad = True

        elif self.unfreeze_layers == "all":
            for param in model.parameters():
                param.requires_grad = True

        elif self.unfreeze_layers == "from_layer4":
            for param in model.layer4.parameters():  # Assuming "layer4" is the second-to-last layer
                param.requires_grad = True

            for param in model.fc.parameters():  # Last layer
                param.requires_grad = True

        elif self.unfreeze_layers == "from_layer3":
            for param in model.layer3.parameters():  # Third-to-last layer
                param.requires_grad = True

            for param in model.layer4.parameters():  # Second-to-last layer
                param.requires_grad = True

            for param in model.fc.parameters():  # Last layer
                param.requires_grad = True




    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (tuple): Input batch.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss for logging.
        """
        inputs, labels = batch
        inputs = inputs.float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.long())
        if loss is None:
            loss = 1.0
        self.log('train_loss', loss)
        # print("training loss", loss)
        self.writer.add_scalar("Training_Loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (tuple): Input batch.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss for logging.
        """
        inputs, labels = batch
        inputs = inputs.float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.long())
        if loss is None:
            loss = 1.0
        self.log('val_loss', loss)
        # print("validation step loss", loss)
        self.writer.add_scalar("Validation_Loss", loss)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        return self.optimizer

    def train_dataloader(self):
        """
        Create the training dataloader.

        Returns:
            torch.utils.data.DataLoader: Training dataloader.
        """
        return self.dataloaders['train']

    def val_dataloader(self):
        """
       Create the validation dataloader.

       Returns:
           torch.utils.data.DataLoader: Validation dataloader.
       """
        return self.dataloaders['val']

    def on_train_epoch_end(self):
        """
        Perform actions at the end of each training epoch.
        """
        # Process validation outputs and compute metrics
        train_outputs = self.trainer.logged_metrics['train_loss']
        # print("training epochend",train_outputs)
        train_preds, train_labels = self.get_predictions_and_labels('train')
        # print("Train_Preds:", train_preds)
        # print("Train_Labels:", train_labels)

        try:
            precision_train_ok = precision_score(train_labels, train_preds, pos_label=0)
        except ZeroDivisionError:
            precision_train_ok = 0.0
        try:
            precision_train_nok = precision_score(train_labels, train_preds, pos_label=1)
        except ZeroDivisionError:
            precision_train_nok = 0.0

        #self.log('precision_train_ok', precision_train_ok)
        #self.log('precision_train_nok', precision_train_nok)

        self.writer.add_scalar("precision_train_ok", precision_train_ok)
        self.writer.add_scalar("precision_train_Nok", precision_train_nok)

        #print('precision_train_ok', precision_train_ok)
        #print('precision_train_nok', precision_train_nok)

        try:
            recall_train_ok = recall_score(train_labels, train_preds, pos_label=0)
        except ZeroDivisionError:
            recall_train_ok = 0.0
        try:
            recall_train_nok = recall_score(train_labels, train_preds, pos_label=1)
        except ZeroDivisionError:
            recall_train_nok = 0.0

        #self.log('recall_train_ok', recall_train_ok)
        #self.log('recall_train_nok', recall_train_nok)

        self.writer.add_scalar("recall_train_ok", recall_train_ok)
        self.writer.add_scalar("recall_train_Nok", recall_train_nok)
        #print('recall_train_ok', recall_train_ok)
        #print('recall_train_nok', recall_train_nok)

        try:
            fbeta_score_train_ok = fbeta_score(train_labels, train_preds, pos_label=0, beta=0.5)
        except ZeroDivisionError:
            fbeta_score_train_ok = 0.0
        try:
            fbeta_score_train_nok = fbeta_score(train_labels, train_preds, pos_label=1, beta=1.5)
        except ZeroDivisionError:
            fbeta_score_train_nok = 0.0

        #self.log("fbeta_score_train_ok", fbeta_score_train_ok)
        #self.log("fbeta_score_train_nok", fbeta_score_train_nok)

        self.writer.add_scalar("F_train_ok", fbeta_score_train_ok)
        self.writer.add_scalar("F_train_Nok", fbeta_score_train_nok)

        #print("fbeta_score_train_ok", fbeta_score_train_ok)
        #print("fbeta_score_train_nok", fbeta_score_train_nok)

        cm = confusion_matrix(train_labels, train_preds)
        confusion_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        #print(confusion_df)
        # Extract individual values from the confusion matrix
        # print(cm)
        tn, fp, fn, tp = cm.ravel()
        # print(f'Confusion Matrix:')
        # print(f'TP: {tp}  FP: {fp}')
        # print(f'FN: {fn}  TN: {tn}')

        # self.log("train_confusion_df",cm)
        #self.log("train_TN", tn)
        #self.log("train_TP", tp)
        #self.log("train_FP", fp)
        #self.log("train_FN", fn)

        try:
            if fn == 0:
                omission_rate_train_nok = 1.0
            else:
                omission_rate_train_nok = fn / (tn + fn)
        except:
            omission_rate_train_nok = 1.0

        self.writer.add_scalar("Omission_train_Nok", omission_rate_train_nok)
        #print("omission_rate_train_ok", omission_rate_train_ok)
        #print("omission_rate_train_nok", omisison_rate_train_nok)

        # trainc_p_nok = tp / (fp + tp)
        # trainc_p_ok = tn / (tn + fn)
        #
        # self.log("trainc_p_nok", trainc_p_nok)
        # self.log("trainc_p_ok", trainc_p_ok)
        #
        # trainc_r_ok = tn / (tn + fp)
        # trainc_r_nok = tp / (fn + tp)
        #
        # self.log("trainc_r_nok", trainc_r_nok)
        # self.log("trainc_r_ok", trainc_r_ok)
        #
        # trainc_f_ok = ((1 + (0.5 * 0.5)) * trainc_p_ok * trainc_r_ok) / (((0.5 * 0.5) * trainc_p_ok) + trainc_r_ok)
        # trainc_f_nok = ((1 + (1.5 * 1.5)) * trainc_p_nok * trainc_r_nok) / (((1.5 * 1.5) * trainc_p_nok) + trainc_r_nok)
        # # print("check:", p_ok, p_nok)
        #
        # self.log("trainc_f_nok", trainc_f_nok)
        # self.log("trainc_f_ok", trainc_f_ok)

    def on_validation_epoch_end(self):
        """
        Perform actions at the end of each validation epoch.
        """
        # Process validation outputs and compute metrics
        val_outputs = self.trainer.logged_metrics['val_loss']
        # print("Valepoch end", val_outputs)
        # print("test2")

        val_preds, val_labels = self.get_predictions_and_labels('val')
        # print("Val_Preds:" ,val_preds)
        # print("Val_Labels:", val_labels)

        try:
            precision_val_ok = precision_score(val_labels, val_preds, pos_label=0, zero_division=0)
        except ZeroDivisionError:
            precision_val_ok = 0.0
        try:
            precision_val_nok = precision_score(val_labels, val_preds, pos_label=1, zero_division=0)
        except ZeroDivisionError:
            precision_val_nok =0.0



        #self.log("precision_val_ok", precision_val_ok)
        #self.log("precision_val_nok", precision_val_nok)

        self.writer.add_scalar("Precision_val_ok", precision_val_ok)
        self.writer.add_scalar("Precision_val_Nok", precision_val_nok)
        #print("precision_val_ok", precision_val_ok)
        #print("precision_val_nok", precision_val_nok)

        try:
            recall_val_ok = recall_score(val_labels, val_preds, pos_label=0, zero_division=0)
        except:
            recall_val_ok = 0.0
        try:
            recall_val_nok = recall_score(val_labels, val_preds, pos_label=1, zero_division=0)
        except ZeroDivisionError:
            recall_val_nok = 0.0
        #self.log("recall_val_ok", recall_val_ok)
        #self.log("recall_val_nok", recall_val_nok)

        self.writer.add_scalar("recall_val_ok", recall_val_ok)
        self.writer.add_scalar("recall_val_Nok", recall_val_nok)
        #print("recall_val_ok", recall_val_ok)
        #print("recall_val_nok", recall_val_nok)

        try:
            fbeta_score_val_ok = fbeta_score(val_labels, val_preds, pos_label=0, beta=0.5, zero_division=0)
        except ZeroDivisionError:
            fbeta_score_val_ok = 0.0
        try:
            fbeta_score_val_nok = fbeta_score(val_labels, val_preds, pos_label=1, beta=1.5, zero_division=0)
        except:
            fbeta_score_val_nok = 0.0

        #self.log("fbeta_score_val_ok", fbeta_score_val_ok)
        #self.log("fbeta_score_val_nok", fbeta_score_val_nok)

        self.writer.add_scalar("F_val_ok", fbeta_score_val_ok)
        self.writer.add_scalar("F_val_Nok", fbeta_score_val_nok)

        #print("fbeta_score_val_ok", fbeta_score_val_ok)
        #print("fbeta_score_val_nok", fbeta_score_val_nok)

        cm = confusion_matrix(val_labels, val_preds)
        confusion_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        #print(confusion_df)
        # Extract individual values from the confusion matrix
        # print(cm)
        tn, fp, fn, tp = cm.ravel()
        # print(f'Confusion Matrix:')
        # print(f'TP: {tp}  FP: {fp}')
        # print(f'FN: {fn}  TN: {tn}')

        # self.log("val_confusion_df", cm)
        # self.log("val_TN", tn)
        # self.log("val_TP", tp)
        # self.log("val_FP", fp)
        # self.log("val_FN", fn)


        try:
            if fn == 0:
                omission_rate_val_nok = 1.0
            else:
                omission_rate_val_nok = fn / (tn + fn)
        except:
            omission_rate_val_nok = 1.0

        self.writer.add_scalar("omission_val_Nok", omission_rate_val_nok)

        #print("omission_rate_val_ok", omission_rate_val_ok)
        #print("omission_rate_val_nok", omisison_rate_val_nok)

        #valc_p_nok = tp / (fp + tp)
        #valc_p_ok = tn / (tn + fn)

        #self.log("valc_p_nok", valc_p_nok)
        #self.log("valc_p_ok", valc_p_ok)
        # print("check:", p_ok, p_nok)

        #valc_r_ok = tn / (tn + fp)
        #valc_r_nok = tp / (fn + tp)

        #self.log("valc_r_ok", valc_r_ok)
        #self.log("valc_r_nok", valc_r_nok)

        #valc_f_ok = ((1 + (0.5 * 0.5)) * valc_p_ok * valc_r_ok) / (((0.5 * 0.5) * valc_p_ok) + valc_r_ok)
        #valc_f_nok = ((1 + (1.5 * 1.5)) * valc_p_nok * valc_r_nok) / (((1.5 * 1.5) * valc_p_nok) + valc_r_nok)

        #self.log("valc_f_ok", valc_f_ok)
        #self.log("valc_f_nok", valc_f_nok)

        # Early stopping check
        if val_outputs < self.best_val_loss:
            self.best_val_loss = val_outputs
            self.current_patience = 0
        else:
            self.current_patience += 1
            if self.current_patience >= self.early_stop_patience:
                self.trainer.should_stop = True  # Trigger early stopping

    def get_predictions_and_labels(self, phase):
        """
        Get predictions and true labels for a given phase.

        Args:
            phase (str): Phase for which to get predictions and labels ('train', 'val', or 'test').

        Returns:
            tuple: Tuple containing lists of true labels and predicted labels.
        """
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in self.dataloaders[phase]:
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device).float()
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        return all_labels, all_predictions


def get_transforms_from_strings(transformations):
    composed_transforms = []

    for transformation in transformations:
        #print(transformation)
        if "Noise Scale" in transformation:
            noise_scale_range = float(transformation.split(": ")[1])
            composed_transforms.append(transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_scale_range))
        elif "Gray Scale" in transformation:
            composed_transforms.append(transforms.Grayscale())
        elif "Gaussian Blur" in transformation:
            blur_sigma = float(transformation.split("(Sigma): ")[1])
            composed_transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=blur_sigma))
        elif "Color Jitter" in transformation:
            values = transformation.split("(")[1].split(")")[0].split(", ")
            brightness = float(values[0].split(": ")[1])
            contrast = float(values[1].split(": ")[1])
            saturation = float(values[2].split(": ")[1])
            hue = float(values[3].split(": ")[1])
            composed_transforms.append(transforms.ColorJitter(brightness=brightness,
                                                             contrast=contrast,
                                                             saturation=saturation,
                                                             hue=hue))
        elif "Rotation Angle" in transformation:
            rotation_angle = int(transformation.split(": ")[1])
            composed_transforms.append(transforms.RandomRotation(degrees=rotation_angle))
        elif "Horizontal Flip" in transformation:
            composed_transforms.append(transforms.RandomHorizontalFlip())
        elif "Vertical Flip" in transformation:
            composed_transforms.append(transforms.RandomVerticalFlip())
        else:
            # Handle unrecognized transformations or other specific cases
            pass

        composed_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        composed_transforms.append(transforms.ToTensor())

    return transforms.Compose(composed_transforms)





def objective(trial):
    """
    Objective function for Optuna optimization.

    Args:
        trial (optuna.trial.Trial): Optuna's trial object.

    Returns:
        float: Validation loss.
    """
    # Hyperparameters we want optimize

    transfomation_list = ['noise_scale', 'gray_scale', 'gaussian_blur', 'color_jitter', 'rotation', 'horizontal_flip','vertical_flip']
    #transfomation_list = ['color_jitter']
    transfomation_combinations = []
    for r in range(1, len(transfomation_list) + 1):
        transfomation_combinations.extend([list(x) for x in itertools.combinations(iterable=transfomation_list, r=r)])
    #print(transfomation_combinations)

    params = {
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2),
        'optimizer_name': trial.suggest_categorical('optimizer', ['SGD', 'Adam']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
        'unfreeze_layers': trial.suggest_categorical("unfreeze_layers",
                                                     ["all", "last", "from_layer4", "from_layer3"]),
        #'transformations': trial.suggest_categorical('transformations', choices=transfomation_combinations),
        #'perform_transform': trial.suggest_categorical("perform_transform", choices = ["yes", "no"])
        'noise_scale' : trial.suggest_categorical('noise_scale', [True, False]),
        'gray_scale' : trial.suggest_categorical('gray_scale', [True, False]),
        'gaussian_blur' : trial.suggest_categorical('gaussian_blur', [True, False]),
        'color_jitter' : trial.suggest_categorical('color_jitter',[True, False]),
        'rotation' : trial.suggest_categorical('rotation', [True, False]),
        'horizontal_flip' : trial.suggest_categorical('horizontal_flip', [True, False]),
        'vertical_flip' : trial.suggest_categorical('vertical_flip', [True, False])
    }



    temp_transformation_list = []
    temp_transformation_list.append(transforms.ToTensor())
    if params["noise_scale"] :
        # Define the range for noise scale transformation
        noise_scale_range = round(random.uniform(0.0, 1.8), 1)
        #noise_scale_range = trial.suggest_float('noise_scale_range', 0.0, 1.8, step = 0.1)
        temp_transformation_list.append(transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_scale_range))
    if params["gray_scale"] :
        probability = round(random.uniform(0.0, 1.0), 1)
        #probability = trial.suggest_float("probability", 0.0, 1.0, step=0.1)
        temp_transformation_list.append(transforms.RandomGrayscale(p=probability))
    if params["gaussian_blur"]:

        kernel_size = random.randrange(1,51,2)
        #kernel_size = trial.suggest_int("kernal_size", 1,51, step = 2)
        temp_transformation_list.append(transforms.GaussianBlur((kernel_size, kernel_size+2), sigma=(0.1, 5.)))
    if params["color_jitter"] :
        brightness_range = [0.5, 1.5]  # Make sure it's a list or tuple with two elements
        contrast = round(random.uniform(0,1), 1)
        #contrast = trial.suggest_float("contrast", 0, 1, step=0.1)
        saturation = round(random.uniform(0,1), 1)
        #saturation = trial.suggest_float("saturation", 0, 1, step=0.1)
        hue = round(random.uniform(0,0.5), 1)
        #hue = trial.suggest_float("hue", 0, 0.5, step=0.1)

        temp_transformation_list.append(transforms.ColorJitter(
            brightness=brightness_range,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ))

    if params["rotation"]:
        temp_transformation_list.append(transforms.RandomRotation(degrees=(0,180)))

    if params["horizontal_flip"]:
        temp_transformation_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if params["vertical_flip"]:
        temp_transformation_list.append(transforms.RandomVerticalFlip(p=0.5))


    train_transforms = transforms.Compose(temp_transformation_list)


    # noise_scale = params["noise_scale"]
    # gray_scale = params["gray_scale"]
    # gaussian_blur = params["gaussian_blur"]
    # color_jitter = params["color_jitter"]
    # rotation = params["rotation"]
    # horizontal_flip = params["horizontal_flip"]
    # vertical_flip = params["vertical_flip"]
    #
    # #print(noise_scale)
    #
    # transformations = []
    # # Check which transformations are selected
    # if noise_scale:
    #     print("entered loop")
    #     # Define the range for noise scale transformation
    #     noise_scale_range = trial.suggest_float('noise_scale_range', 0.1, 1.0)
    #     transformations.append(f"Noise Scale: {noise_scale_range}")
    #
    # if gray_scale:
    #     transformations.append("Gray Scale")
    #
    # if gaussian_blur:
    #     # Define the range for Gaussian blur
    #     blur_sigma = trial.suggest_float('blur_sigma', 0.1, 1.0)
    #     transformations.append(f"Gaussian Blur (Sigma): {blur_sigma}")
    #
    # if color_jitter:
    #     # Define the range for color jitter transformations
    #     brightness = trial.suggest_float('brightness', 0.1, 1.0)
    #     contrast = trial.suggest_float('contrast', 0.1, 1.0)
    #     saturation = trial.suggest_float('saturation', 0.1, 1.0)
    #     hue = trial.suggest_float('hue', 0, 0.5)
    #     transformations.append(
    #         f"Color Jitter (Brightness: {brightness}, Contrast: {contrast}, Saturation: {saturation}, Hue: {hue})")
    #
    # if rotation:
    #     # Define the range for rotation angle
    #     rotation_angle = trial.suggest_int('rotation_angle', 0, 360)
    #     transformations.append(f"Rotation Angle: {rotation_angle}")
    #
    # if horizontal_flip:
    #     transformations.append("Horizontal Flip")
    #
    # if vertical_flip:
    #     transformations.append("Vertical Flip")
    #
    # #transformations.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    #
    #
    #
    # #print(transformations)
    # train_transforms = get_transforms_from_strings(transformations)
    # print("train_transforms", train_transforms)

    val_transforms = transforms.Compose([
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.ToTensor()
    ])



    # Define your ResNet-50 model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(params['dropout_rate']), nn.Linear(num_ftrs, 2))
    model.to(device)

    # Define criterion
    criterion = nn.CrossEntropyLoss()

    optimizer = getattr(
        torch.optim, params["optimizer_name"]
    )(model.parameters(), lr=params["lr"])

    writer = SummaryWriter()
    transfer_learning = params['unfreeze_layers']
    custom_model = CustomModel(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        early_stop_patience=5,
        transfer_learning = transfer_learning,
        train_transforms = train_transforms,
        val_transforms = val_transforms,
        writer=writer
    )
    #csv_logger = CSVLogger(save_dir="/home/hpc/iwfa/iwfa011h/Code/Code/ResNet50/logs/train", name="optuna_50")


    trainer = pl.Trainer(
        accelerator="auto",  # Use GPU if available
        max_epochs=100,
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')]
    )

    trainer.fit(custom_model)
    return trainer.callback_metrics["val_loss"].item()



if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(), storage="sqlite:///part_f_resnet.sqlite3", study_name ="part_f_resnet", load_if_exists=True)
    # Minimize validation loss
    study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed

    # Get the best hyperparameters found by Optuna
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)