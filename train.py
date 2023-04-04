import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import wandb
from hydra.utils import get_original_cwd

from torch.nn.parallel import DataParallel

from torch.profiler import profile, record_function, ProfilerActivity

from models.SparseAttention import SparseAttentionModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project='sparse-attention')

model = SparseAttentionModel().to(device)
model = DataParallel(model)

checkpoint_dir = './checkpoints'

for epoch in range(10):

    running_loss = 0.0
    for i, data in enumerate(trainloader):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        wandb.log({"epoch": epoch, "loss": loss.item()})

        running_loss += loss.item()

    checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch}.pt'
    torch.save(model.state_dict(), checkpoint_path)

    epoch_loss = running_loss / len(trainloader)
    print(f"Epoch {epoch+1} loss: {epoch_loss:.3f}")

torch.save(model.state_dict(), './final_model.pt')

wandb.finish()



