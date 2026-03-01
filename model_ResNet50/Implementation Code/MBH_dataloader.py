from torchvision import transforms
from torch.utils.data import DataLoader
from MBH_dataset import MBH_dataset
import random
import torch



'''

## PART I --> START
brightness_range = [0.5, 1.5]  # Make sure it's a list or tuple with two elements
contrast = round(random.uniform(0,1), 1)
#contrast = trial.suggest_float("contrast", 0, 1, step=0.1)
saturation = round(random.uniform(0,1), 1)
#saturation = trial.suggest_float("saturation", 0, 1, step=0.1)
hue = round(random.uniform(0,0.5), 1)
#hue = trial.suggest_float("hue", 0, 0.5, step=0.1)

train_transform = transforms.Compose([
    transforms.ToTensor(),
transforms.ColorJitter(
        brightness=brightness_range,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])




# Define your custom dataset and dataloaders
# Assume you have a dataset class named CustomDataset
# Prepare your datasets_in and data loaders

train_dataset = MBH_dataset(data_dir='/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                                    label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - I/model_train.txt",
                                    transform=train_transform)
train_dataset.load_data()

val_dataset = MBH_dataset(data_dir='/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                          label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - I/model_val.txt",
                          transform=val_transform)
val_dataset.load_data()

test_dataset = MBH_dataset(data_dir='/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                          label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - I/model_test.txt",
                          transform=test_transform)
test_dataset.load_data()





dataloaders = {
    'train': DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
}

##PART I --> END


'''

