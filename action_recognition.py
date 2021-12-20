import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision.models as models

from gensim.models import KeyedVectors

from pytorch_lightning.core.lightning import LightningModule

from video_dataset import VideoDataset

class ActionRecognition(LightningModule):
    def __init__(self, args):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 300)
        self.wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        self.train_embedding = None
        self.val_embedding = None
        self.criterion = nn.MSELoss()
        self.args = args
        
    
    def train_dataloader(self):
        dataset = VideoDataset('kinetics', 'kinetics700_2020', ['train.json', 'validate.json', 'test.json'], self.args, False)
        weight = []
        for i in range(len(dataset.actions)):
            words = torch.stack([torch.Tensor(self.wv[word]) for word in dataset.actions[i]], 0)
            weight.append(words.mean(0).view(-1))
        weight = torch.stack(weight, 0)
        self.train_embedding = nn.Embedding.from_pretrained(weight)
        for p in self.train_embedding.parameters():
            p.requires_grad = False
        return DataLoader(dataset, batch_size = self.args.batch, num_workers=0)

    def val_dataloader(self):
        dataset = VideoDataset('activity-net', 'activity-net', ['activity-net.json'], self.args, True)
        weight = []
        for i in range(len(dataset.actions)):
            words = torch.stack([torch.Tensor(self.wv[word]) for word in dataset.actions[i]], 0)
            weight.append(words.mean(0).view(-1))
        weight = torch.stack(weight, 0)
        self.val_embedding = nn.Embedding.from_pretrained(weight)
        for p in self.val_embedding.parameters():
            p.requires_grad = False
        return DataLoader(dataset, batch_size = self.args.batch, num_workers=0)
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.args.lr)
    
    def forward(self, batch, batch_idx):
        x = batch
        
        batch_size = x.size(0)
        length = x.size(1)
        x = x.transpose(-1, -3)
        x = x.view(-1, *x.shape[2:])
        
        
        y_hat = self.backbone(x)
        y_hat = y_hat.view(batch_size, length, -1)
        y_hat = self.fc(y_hat.relu())
        y_hat = y_hat.view(batch_size, length, 300)
        y_hat = y_hat.mean(1)
        
        
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x, batch_idx)
        self.train_embedding = self.train_embedding.to(y_hat.device)
        y = self.train_embedding(y)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y_hat = self(x, batch_idx)
        self.val_embedding = self.val_embedding.to(y_hat.device)
        y_idx = y
        y = self.val_embedding(y)
        
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        scores = torch.einsum('nc,mc->nm', y_hat, self.val_embedding.weight)
        scores = scores.argsort(1, descending = True)
        scores = scores[torch.arange(y.size(0)), y_idx]
        
        top1acc = (scores < 1).float().mean()
        top5acc = (scores < 5).float().mean()
        self.log("top1acc", top1acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("top5acc", top5acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        