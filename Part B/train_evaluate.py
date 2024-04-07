import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torchvision.models as models



class DataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, augment_data= True ,data_dir = './',num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.augment_data = augment_data
        self.num_workers=num_workers


        self.augmentation = transforms.Compose([
              transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
              transforms.RandomRotation(degrees=15),
              transforms.RandomHorizontalFlip(),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform = transforms.Compose([
              transforms.Resize(size=256),
              transforms.CenterCrop(size=224),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])



    def prepare_data(self):
        self.train_dataset = ImageFolder(root=f'{self.data_dir}/train')
        self.test_dataset = ImageFolder(root=f'{self.data_dir}/val')
        print(f'Classes {self.train_dataset.classes}')
        print(f'Test Samples {len(self.test_dataset)}')


    def setup(self, stage=None):
        self.test_dataset.transform = self.transform
        labels = self.train_dataset.targets
        train_indices, val_indices = train_test_split(
            range(len(self.train_dataset)),
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        if stage=='fit':
          class_names = self.train_dataset.classes
          class_counts = {class_name: 0 for class_name in class_names}
          for val_indice in val_indices:
              _, class_idx = self.train_dataset[val_indice]
              class_name = class_names[class_idx]
              class_counts[class_name] += 1
          print('Validation Datased Class Distribution')
          for class_name, count in class_counts.items():
              print(f"Class: {class_name}, No. of samples: {count}")
        self.val_dataset = Subset(self.train_dataset, val_indices)
        self.train_dataset = Subset(self.train_dataset, train_indices)
        if stage=='fit':
          print(f'Train Samples {len(self.train_dataset)}')
          print(f'Val Samples {len(self.val_dataset)}')

        if self.augment_data:
          self.train_dataset.dataset.transform = self.augmentation
        self.val_dataset.dataset.transform = self.transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


class Net(nn.Module):
  def __init__(self):
      super().__init__()
      self.model = models.resnet50(weights='IMAGENET1K_V1')
      num_features = self.model.fc.in_features
      self.model.fc = nn.Linear(num_features, 10)

  def forward(self, x):
        x =  self.model(x)
        return x

class MyCnnModel(L.LightningModule):
    def __init__(self,
                 learning_rate=0.01,
                 loss_function='cross_entropy',
                 optimizer='Adam'):
        super().__init__()
        self.save_hyperparameters()
        self.model = Net()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.loss_function = getattr(F, loss_function)
        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
      loss,acc = self._common_step(batch,batch_idx)
      self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
      self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)
      return loss

    def validation_step(self, batch, batch_idx):
      loss,acc =  self._common_step(batch,batch_idx)
      self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
      self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
      return loss


    def test_step(self,batch,batch_idx):
      loss,acc =  self._common_step(batch,batch_idx)
      self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
      self.log('test_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
      return loss

    def _common_step(self, batch, batch_idx):
      inputs, target = batch
      output = self.forward(inputs)
      loss = self.loss_function(output, target)
      acc = self.accuracy(output, target)
      return loss,acc

    def configure_optimizers(self):
      return getattr(torch.optim,self.optimizer)(self.parameters(), lr=self.learning_rate)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate the CNN Model on the iNaturalist Dataset")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'],
                        help='Operation mode: train or evaluate')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train for')
    parser.add_argument('--augment_data', action='store_true', 
                        help='Flag to augment training data')
    parser.add_argument('--weights_path', type=str, default='', 
                        help='Path to the model weights for evaluation')

    return parser.parse_args()


def train(args):
    dm = DataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    model = MyCnnModel(args.activation_function,args.num_filters,args.dense_neurons,args.learning_rate,args.batch_size,args.dropout,args.batch_norm)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='max',save_top_k=1,filename='best_model')
    early_stop_callback = EarlyStopping(monitor="val_loss",patience=3)
    trainer = L.Trainer(
      max_epochs=args.epochs,
      check_val_every_n_epoch=1,
      log_every_n_steps=1,
      callbacks=[checkpoint_callback,early_stop_callback])
    trainer.fit(model, dm)

def evaluate(args):
    dm = DataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    model = MyCnnModel.load_from_checkpoint(checkpoint_path=args.weights_path) 
    trainer = L.Trainer()
    trainer.test(model, dm)

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)