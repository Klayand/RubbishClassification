import timm
import torch
import torch.distributed as dist
import argparse

from torch import nn

from data import get_rubbish_classification_loader
from Solver import Solver
import torchvision.transforms as transforms
from torchvision.models import resnet34

from backbones import model_dict

# load teacher checkpoint and train student baseline
parser = argparse.ArgumentParser(description="hyper-parameters")

parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--ddp_mode', type=bool, default=False, help='Distributed DataParallel Training?')
parser.add_argument('--sync_bn', type=bool, default=False)
parser.add_argument('--fp16', type=bool, default=False)
parser.add_argument('--model', type=str)
parser.add_argument('--name', type=str, help='Experiment name')
parser.add_argument('--dataset', type=str, default='Rubbish')
parser.add_argument('--num_classes', type=int, default=16)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--save_path', type=bool, default=True)

parser = parser.parse_args()
print("generating config:")
print(f"Config: {parser}")
print('-'*100)


# -------- initialize model ----------------
if parser.pretrained:
    model = resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, parser.num_classes)
else:
    model = model_dict[parser.model](num_classes=parser.num_classes)

# ------- DDP -----------------
if parser.ddp_mode:
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    if parser.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

# ----------------------------------------------------------------------------------------------------------------------
# Get dataloader
if parser.dataset == 'Rubbish':
    train_loader = get_rubbish_classification_loader(mode='train', batch_size=parser.batch_size, ddp=parser.ddp_mode, shuffle=True)
    val_loader = get_rubbish_classification_loader(mode='val', batch_size=parser.batch_size, shuffle=False)

# train teacher baseline
if parser.ddp_mode:
    w = Solver(
        model=model,
        config=parser,
        local_rank=local_rank
    )
else:
    w = Solver(
        model=model,
        config=parser,
    )

w.train(train_loader=train_loader, validation_loader=val_loader, total_epoch=500, save_path=parser.save_path)

if parser.ddp_mode:
    dist.destroy_process_group()

print("-" * 100)
print("Training completed!")
