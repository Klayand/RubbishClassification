import torch
from torch import nn
from torch import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from optimizer import SGD, Adam
from scheduler import ALRS
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


def ce_loss(x, y):
    cross_entropy = F.cross_entropy(x, y)
    return cross_entropy


class Solver:
    def __init__(
            self,
            model: nn.Module,
            loss_function: Callable or None = None,
            optimizer: torch.optim.Optimizer or None = None,
            scheduler: Callable or None = None,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            config=None,
            local_rank=None
    ):
        self.config = config

        self.visual_path = self.config.name
        self.ddp_mode = self.config.ddp_mode
        self.local_rank = local_rank

        self.model = model

        self.criterion = loss_function if loss_function is not None else ce_loss

        if 'mobilenet' in self.config.model:
            self.optimizer = optimizer if optimizer is not None else SGD(self.model, lr=0.01)
        else:
            self.optimizer = optimizer if optimizer is not None else SGD(self.model)

        self.scheduler = scheduler if scheduler is not None else ALRS(self.optimizer)

        self.writer = None
        self.device = device

        # initialization
        self.init()

    def init(self):
        # change device
        self.model.to(self.device)

        if self.ddp_mode:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # tensorboard
        self.writer = SummaryWriter(log_dir=f"runs/{self.visual_path}")

        # module parameters initialization

    def train(
            self, train_loader: DataLoader, validation_loader: DataLoader, total_epoch=350, fp16=False, save_path=False
    ):
        from torch.cuda.amp import autocast, GradScaler
        BEST_ACC = -99
        scaler = GradScaler()

        for epoch in range(1, total_epoch + 1):
            train_loss, train_acc, validation_loss, validation_acc = 0, 0, 0, 0

            if self.ddp_mode:
                train_loader.sampler.set_epoch(epoch)

            # train
            self.model.train()
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)
                if fp16:
                    with autocast():
                        model_out, _ = self.model(x)
                        _, pre = torch.max(model_out, dim=1)
                        loss = self.criterion(model_out, y)

                else:
                    model_out, _ = self.model(x)
                    _, pre = torch.max(model_out, dim=1)
                    loss = self.criterion(model_out, y)

                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()

                self.optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    scaler.step(self.optimizer)
                    scaler.update()

                else:
                    loss.backward()
                    self.optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix_str(f"loss={train_loss / step}, acc={train_acc / step}")

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            # validation
            vbar = tqdm(validation_loader, colour="yellow")
            self.model.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(vbar, 1):
                    x, y = x.to(self.device), y.to(self.device)
                    model_out, _ = self.model(x)
                    _, pre = torch.max(model_out, dim=1)
                    loss = self.criterion(model_out, y)

                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)
                    validation_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    validation_loss += loss.item()

                    if step % 10 == 0:
                        vbar.set_postfix_str(f"loss={validation_loss / step}, acc={validation_acc / step}")

                validation_loss /= len(validation_loader)
                validation_acc /= len(validation_loader)

                self.writer.add_scalar("test/loss", validation_loss, epoch)
                self.writer.add_scalar("test/acc", validation_acc, epoch)

                if validation_acc > BEST_ACC:
                    BEST_ACC = validation_acc

            self.scheduler.step(train_loss, epoch)

            print(f"epoch {epoch}, train_loss = {train_loss}, train_acc = {train_acc}")
            print(f"epoch {epoch}, validation_loss = {validation_loss}, validation_acc = {validation_acc}")
            print("*" * 100)
        print(f"Best Acc: {BEST_ACC}")

        if save_path:
            torch.save(self.model.state_dict(), f'{self.visual_path}.pth')

        return self.model