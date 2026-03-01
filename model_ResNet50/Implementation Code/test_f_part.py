from torchvision import models
import torch.nn as nn
import torch
from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix,classification_report
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from MBH_dataset import MBH_dataset
import random
import torch
import csv
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_labels = ['Class 0', 'Class 1']

class CustomModel(pl.LightningModule):
    def __init__(self, model, dataloaders, criterion, optimizer, num_epochs, early_stop_patience, transfer_learning, writer ):
        super().__init__()
        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.best_val_loss = float('inf')
        self.current_patience = 0
        self.unfreeze_layers = transfer_learning
        self.writer = writer

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
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.float()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels.long())
        if loss is None:
            loss = 1.0
        self.log('train_loss', loss)
        self.writer.add_scalar("Training_Loss", loss)
        # print("training loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
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
        return self.optimizer

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def on_train_epoch_end(self):
        # Process validation outputs and compute metrics
        train_outputs = self.trainer.logged_metrics['train_loss']
        # print("training epochend",train_outputs)
        train_preds, train_labels = self.get_predictions_and_labels('train')
        # print("Train_Preds:", train_preds)
        # print("Train_Labels:", train_labels)

        precision_train_ok = precision_score(train_labels, train_preds, pos_label=0)
        precision_train_nok = precision_score(train_labels, train_preds, pos_label=1)

        self.log('precision_train_ok', precision_train_ok)
        self.log('precision_train_nok', precision_train_nok)

        print('precision_train_ok', precision_train_ok)
        print('precision_train_nok', precision_train_nok)

        self.writer.add_scalar("precision_train_ok", precision_train_ok)
        self.writer.add_scalar("precision_train_Nok", precision_train_nok)

        recall_train_ok = recall_score(train_labels, train_preds, pos_label=0)
        recall_train_nok = recall_score(train_labels, train_preds, pos_label=1)

        self.log('recall_train_ok', recall_train_ok)
        self.log('recall_train_nok', recall_train_nok)

        print('recall_train_ok', recall_train_ok)
        print('recall_train_nok', recall_train_nok)

        self.writer.add_scalar("recall_train_ok", recall_train_ok)
        self.writer.add_scalar("recall_train_Nok", recall_train_nok)

        fbeta_score_train_ok = fbeta_score(train_labels, train_preds, pos_label=0, beta=0.5)
        fbeta_score_train_nok = fbeta_score(train_labels, train_preds, pos_label=1, beta=1.5)

        self.log("fbeta_score_train_ok", fbeta_score_train_ok)
        self.log("fbeta_score_train_nok", fbeta_score_train_nok)

        print("fbeta_score_train_ok", fbeta_score_train_ok)
        print("fbeta_score_train_nok", fbeta_score_train_nok)

        cm = confusion_matrix(train_labels, train_preds)
        confusion_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        print(confusion_df)
        # Extract individual values from the confusion matrix
        # print(cm)
        tn, fp, fn, tp = cm.ravel()
        # print(f'Confusion Matrix:')
        # print(f'TP: {tp}  FP: {fp}')
        # print(f'FN: {fn}  TN: {tn}')
        self.writer.add_scalar("F_train_ok", fbeta_score_train_ok)
        self.writer.add_scalar("F_train_Nok", fbeta_score_train_nok)

        # self.log("train_confusion_df",cm)
        self.log("train_TN", tn)
        self.log("train_TP", tp)
        self.log("train_FP", fp)
        self.log("train_FN", fn)

        omission_rate_train_ok = fn / (tp + fn)
        omisison_rate_train_nok = fp / (tn + fp)

        self.log("omission_rate_train_ok", omission_rate_train_ok)
        self.log("omission_rate_train_nok", omisison_rate_train_nok)

        print("omission_rate_train_ok", omission_rate_train_ok)
        print("omission_rate_train_nok", omisison_rate_train_nok)

        self.writer.add_scalar("Omission_train_ok", omission_rate_train_ok)
        self.writer.add_scalar("Omission_train_Nok", omisison_rate_train_nok)

        trainc_p_nok = tp / (fp + tp)
        trainc_p_ok = tn / (tn + fn)

        self.log("trainc_p_nok", trainc_p_nok)
        self.log("trainc_p_ok", trainc_p_ok)

        trainc_r_ok = tn / (tn + fp)
        trainc_r_nok = tp / (fn + tp)

        self.log("trainc_r_nok", trainc_r_nok)
        self.log("trainc_r_ok", trainc_r_ok)

        trainc_f_ok = ((1 + (0.5 * 0.5)) * trainc_p_ok * trainc_r_ok) / (((0.5 * 0.5) * trainc_p_ok) + trainc_r_ok)
        trainc_f_nok = ((1 + (1.5 * 1.5)) * trainc_p_nok * trainc_r_nok) / (((1.5 * 1.5) * trainc_p_nok) + trainc_r_nok)
        # print("check:", p_ok, p_nok)

        self.log("trainc_f_nok", trainc_f_nok)
        self.log("trainc_f_ok", trainc_f_ok)

    def on_validation_epoch_end(self):
        # Process validation outputs and compute metrics
        val_outputs = self.trainer.logged_metrics['val_loss']
        # print("Valepoch end", val_outputs)
        # print("test2")

        val_preds, val_labels = self.get_predictions_and_labels('val')
        # print("Val_Preds:" ,val_preds)
        # print("Val_Labels:", val_labels)
        precision_val_ok = precision_score(val_labels, val_preds, pos_label=0, zero_division=0)
        precision_val_nok = precision_score(val_labels, val_preds, pos_label=1, zero_division=0)

        self.log("precision_val_ok", precision_val_ok)
        self.log("precision_val_nok", precision_val_nok)

        print("precision_val_ok", precision_val_ok)
        print("precision_val_nok", precision_val_nok)

        self.writer.add_scalar("Precision_val_ok", precision_val_ok)
        self.writer.add_scalar("Precision_val_Nok", precision_val_nok)

        recall_val_ok = recall_score(val_labels, val_preds, pos_label=0, zero_division=0)
        recall_val_nok = recall_score(val_labels, val_preds, pos_label=1, zero_division=0)

        self.log("recall_val_ok", recall_val_ok)
        self.log("recall_val_nok", recall_val_nok)

        print("recall_val_ok", recall_val_ok)
        print("recall_val_nok", recall_val_nok)

        self.writer.add_scalar("recall_val_ok", recall_val_ok)
        self.writer.add_scalar("recall_val_Nok", recall_val_nok)

        fbeta_score_val_ok = fbeta_score(val_labels, val_preds, pos_label=0, beta=0.5, zero_division=0)
        fbeta_score_val_nok = fbeta_score(val_labels, val_preds, pos_label=1, beta=1.5, zero_division=0)

        self.log("fbeta_score_val_ok", fbeta_score_val_ok)
        self.log("fbeta_score_val_nok", fbeta_score_val_nok)

        print("fbeta_score_val_ok", fbeta_score_val_ok)
        print("fbeta_score_val_nok", fbeta_score_val_nok)

        self.writer.add_scalar("F_val_ok", fbeta_score_val_ok)
        self.writer.add_scalar("F_val_Nok", fbeta_score_val_nok)

        cm = confusion_matrix(val_labels, val_preds)
        confusion_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        print(confusion_df)
        # Extract individual values from the confusion matrix
        # print(cm)
        tp, fn, fp, tn = cm.ravel()
        # print(f'Confusion Matrix:')
        # print(f'TP: {tp}  FP: {fp}')
        # print(f'FN: {fn}  TN: {tn}')

        # self.log("val_confusion_df", cm)
        self.log("val_TN", tn)
        self.log("val_TP", tp)
        self.log("val_FP", fp)
        self.log("val_FN", fn)

        omission_rate_val_ok = fn / (tp + fn)
        omisison_rate_val_nok = fp / (tn + fp)

        self.log("omission_rate_val_ok", omission_rate_val_ok)
        self.log("omission_rate_val_nok", omisison_rate_val_nok)

        print("omission_rate_val_ok", omission_rate_val_ok)
        print("omission_rate_val_nok", omisison_rate_val_nok)

        self.writer.add_scalar("omission_val_ok", omission_rate_val_ok)
        self.writer.add_scalar("omission_val_Nok", omisison_rate_val_nok)

        valc_p_nok = tp / (fp + tp)
        valc_p_ok = tn / (tn + fn)

        self.log("valc_p_nok", valc_p_nok)
        self.log("valc_p_ok", valc_p_ok)
        # print("check:", p_ok, p_nok)

        valc_r_ok = tn / (tn + fp)
        valc_r_nok = tp / (fn + tp)

        self.log("valc_r_ok", valc_r_ok)
        self.log("valc_r_nok", valc_r_nok)

        valc_f_ok = ((1 + (0.5 * 0.5)) * valc_p_ok * valc_r_ok) / (((0.5 * 0.5) * valc_p_ok) + valc_r_ok)
        valc_f_nok = ((1 + (1.5 * 1.5)) * valc_p_nok * valc_r_nok) / (((1.5 * 1.5) * valc_p_nok) + valc_r_nok)

        self.log("valc_f_ok", valc_f_ok)
        self.log("valc_f_nok", valc_f_nok)

        # Early stopping check
        if val_outputs < self.best_val_loss:
            self.best_val_loss = val_outputs
            self.current_patience = 0
        else:
            self.current_patience += 1
            if self.current_patience >= self.early_stop_patience:
                self.trainer.should_stop = True  # Trigger early stopping

    def get_predictions_and_labels(self, phase):
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



params = {
    'batch_size': 16,
    'lr': 0.004411608055756486,
    'optimizer_name': 'SGD',
    'dropout_rate': 0.365442517469485,
    'unfreeze_layers': 'last'
}


## PART F --> START

brightness_range = [0.5, 1.5]  # Make sure it's a list or tuple with two elements
contrast = round(random.uniform(0,1), 1)
#contrast = trial.suggest_float("contrast", 0, 1, step=0.1)
saturation = round(random.uniform(0,1), 1)
#saturation = trial.suggest_float("saturation", 0, 1, step=0.1)
hue = round(random.uniform(0,0.5), 1)
#hue = trial.suggest_float("hue", 0, 0.5, step=0.1)
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
transforms.ColorJitter(
        brightness=brightness_range,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    ),
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
                                    label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - F/model_train.txt",
                                    transform=train_transform)
train_dataset.load_data()

val_dataset = MBH_dataset(data_dir='/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                          label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - F/model_val.txt",
                          transform=val_transform)
val_dataset.load_data()

test_dataset = MBH_dataset(data_dir='/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/',
                          label_file="/home/woody/iwfa/iwfa011h/Fangjun_Wang_2021_MBH-dataset/MBH-dataset/Labels/Part - F/model_test.txt",
                          transform=test_transform)
test_dataset.load_data()





dataloaders = {
    'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
}

##PART F --> END


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

writer = SummaryWriter(log_dir="tensorboard_PART_F/")
transfer_learning = params['unfreeze_layers']
custom_model = CustomModel(
    model=model,
    dataloaders=dataloaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=100,
    early_stop_patience=5,
    transfer_learning = transfer_learning,
    writer = writer,



)

csv_logger = CSVLogger(save_dir="/home/hpc/iwfa/iwfa011h/Code/Code/Code/Code/Code/Code/ResNet50/logs/train", name="metrics")

trainer = pl.Trainer(
    accelerator="auto",  # Use GPU if available
    max_epochs=100,
    callbacks=[pl.callbacks.EarlyStopping(monitor='valc_f_nok', patience=12, mode='max', verbose=True)], logger=csv_logger
)

trainer.fit(custom_model)
torch.save(model.state_dict(), 'resnet_part_f.pth')


# Testing loop
def test_model(model, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set the model to evaluation mode
    model = model.to(device)

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Disable gradient computation for testing
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels.long())  # Convert labels to 'long'

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss = running_loss / len(dataloader)
    test_acc = correct / total

    precision_ok = precision_score(all_labels, all_predictions, pos_label=0)
    precision_nok = precision_score(all_labels, all_predictions, pos_label=1)

    recall_ok = recall_score(all_labels, all_predictions, pos_label=0)
    recall_not_ok = recall_score(all_labels, all_predictions, pos_label=1)

    fbeta_score_ok = fbeta_score(all_labels, all_predictions, pos_label=0, beta=0.5)
    fbeta_score_not_ok = fbeta_score(all_labels, all_predictions, pos_label=1, beta=1.5)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} ')
    print(f'Precision (ok): {precision_ok}')
    print(f'Precision (not ok): {precision_nok}')
    print(f'Recall (ok): {recall_ok}')
    print(f'Recall (not ok): {recall_not_ok}')
    print(f'F-beta Score (ok): {fbeta_score_ok}')
    print(f'F-beta Score (not ok): {fbeta_score_not_ok}')

    # Generate the classification report
    report = classification_report(all_labels, all_predictions)
    print(report)
    cm = confusion_matrix(all_labels, all_predictions)
    confusion_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    print(confusion_df)
    # Extract individual values from the confusion matrix
    print(cm)
    tp, fn, fp, tn = cm.ravel()
    print(f'Confusion Matrix:')
    print(f'TN: {tn}  FP: {fp}')
    print(f'FN: {fn}  TP: {tp}')

    omission_rate_ok = fn / (tp + fn)
    omisison_rate_nok = fp / (tn + fp)

    p_nok = tp / (fp + tp)
    p_ok = tn / (tn + fn)

    r_ok = tn / (tn + fp)
    r_nok = tp / (fn + tp)

    f_ok = ((1 + (0.5 * 0.5)) * p_ok * r_ok) / (((0.5 * 0.5) * p_ok) + r_ok)
    f_nok = ((1 + (1.5 * 1.5)) * p_nok * r_nok) / (((1.5 * 1.5) * p_nok) + r_nok)

    print("omission_rate_ok: ", omission_rate_ok, "omission_rate_nok", omisison_rate_nok)
    # print("p_ok:", p_ok, "p_nok:", p_nok)
    # print("r_ok:", r_ok, "r_nok:", r_nok)
    # print("f_ok:", f_ok, "f_nok", f_nok)
    header = ['Test Loss', "Test Acc", "Test Ok precision", "Test NOK Precision", "Test Ok Recall", "Test NOK Recall",
              "Fß Ok Score", "Fß NOK Score", "TP", "TN", "FP", "FN", "Omission Rate OK", "Omission NOK"]
    data = [test_loss, test_acc, precision_ok, precision_nok, recall_ok,recall_not_ok,
            fbeta_score_ok, fbeta_score_not_ok,tp,tn,fp,fn, omission_rate_ok, omisison_rate_nok ]
    file_path = "resnet_part_f_test.csv"

    with open(file_path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(data)


# Test the model on the validation or test dataset
test_model(model, dataloaders['test'])