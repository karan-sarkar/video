import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import torchvision.models as models

from gensim.models import KeyedVectors

from pytorch_lightning.core.lightning import LightningModule