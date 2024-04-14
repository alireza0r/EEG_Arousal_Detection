from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import os
from tqdm import tqdm
from datetime import datetime

class Trainer(nn.Module):
    def __init__(self, model, criterion, optimizer, save_path='./', seed=42):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.save_path = save_path
        self.writer = SummaryWriter(os.path.join(self.save_path, 'Logs'))
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, inputs):
        outputs = self.model(inputs)
        loss = self.criterion(outputs, inputs)
        # loss = loss.mean()
        return loss.mean(), outputs

    def train_step(self, inputs):
        self.optimizer.zero_grad()
        loss, outputs = self(inputs)
        loss.backward()
        self.optimizer.step()
        return loss.item(), outputs

    def train(self, train_loader, val_loader, num_epochs, save_weight_mode=None, save_weight_acc=None):
        for epoch in range(num_epochs):
            # Training phase
            s_time = time()
            self.model.train()
            total_train_loss = 0.0
            with tqdm(total=len(train_loader), desc="Train Batch |") as bar:
              for inputs in train_loader:
                  # inputs[0].to(self.device)
                  train_loss, _ = self.train_step(inputs[0].to(self.device))
                  total_train_loss += train_loss

                  bar.update(1)

            # Validation phase
            self.model.eval()
            total_val_loss = 0.0
            total_val_acc = 0.0
            with tqdm(total=len(val_loader), desc="Validate Batch |") as bar:
              for inputs in val_loader:
                  # inputs[0].to(self.device)
                  val_loss, outputs = self.evaluate(inputs[0].to(self.device))
                  total_val_loss += val_loss

                  acc = 0
                  total_val_acc += acc

                  bar.update(1)

            # Average loss over all batches
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            avg_val_acc = total_val_acc / len(val_loader)

            duration = time()-s_time

            # Print epoch-wise loss
            print(f'Epoch [{epoch + 1}/{num_epochs}] in [{duration:.3f} s], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc:{avg_val_acc:.4f}')

            self.add_scalar_to_tensorboard('Train Loss', avg_train_loss, epoch+1)
            self.add_scalar_to_tensorboard('Val Loss', avg_val_loss, epoch+1)
            self.add_scalar_to_tensorboard('Val Acc', avg_val_acc, epoch+1)

            if save_weight_acc:
              if avg_val_acc > save_weight_acc:
                self.save_model_weights(self.save_path)
                reports = {
                          'epoch': epoch,
                          'train_loss': avg_train_loss,
                          'val_loss': avg_val_loss,
                          'val_acc': avg_val_acc,
                          'Time_s': duration,
                          'Seed': self.seed,
                          }
                self.save_reports(reports, os.path.join(self.save_path, f'training_reports_epoch_{epoch}.pth'))

        self.save_model_weights()

    def evaluate(self, inputs):
        with torch.no_grad():
            loss, outputs = self(inputs)
        return loss.item(), outputs

    def add_scalar_to_tensorboard(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def close_tensorboard_writer(self):
        self.writer.close()

    def save_model_weights(self, path='/content/MyDrive/MyDrive/Code_AR/DataVisualization/Reports/Weights/'):
        path = os.path.join(path, datetime.utcnow().strftime("%Y_%h_%d_%H_%M_%S_.pt"))
        torch.save(self.model.state_dict(), path)
        print('Model saved...')
        print('Save path:')
        print(path)

    def save_reports(self, reports, path='/content/MyDrive/MyDrive/Code_AR/DataVisualization/Reports/training_reports.pth'):
        torch.save(reports, path)

    def load_model_weights(self, path):
      self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
