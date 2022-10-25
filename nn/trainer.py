"""
"""

from copy import deepcopy
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

from .dataset import SeizureDataset
from .model import SeizureMLP
from utils import ReprMixin, get_kwargs
from metrics import ClassificationMetrics
from config import CFG


__all__ = [
    "SeizureTrainer",
]


class SeizureTrainer(ReprMixin):
    """ """

    __name__ = "SeizureTrainer"

    __DEFEAULT_TRAIN_CONFIG__ = CFG(
        epochs=200,
        lr=0.02,
        weight_decay=0.0001,
        early_stopping_patience=0.15,  # proportion or absolute number
        early_stopping_min_delta=0.0001,
        optimizer="Adam",
        optimizer_kw={},
        scheduler="linear",
        scheduler_kw={"num_warmup_steps": 100},
        label_smoothing=0.1,
        monitor="val_auc",
    )

    def __init__(
        self,
        model_cls: SeizureMLP = SeizureMLP,
        dataset_cls: SeizureDataset = SeizureDataset,
        train_config: Optional[CFG] = None,
        model_config: Optional[CFG] = None,
        preprocess_config: Optional[CFG] = None,
        feature_config: Optional[CFG] = None,
        feature_set: str = "TDSB",
        device: Optional[torch.device] = None,
        over_sampler: Optional[str] = None,
    ) -> None:
        """ """
        self.train_config = deepcopy(self.__DEFEAULT_TRAIN_CONFIG__)
        self.train_config.update(train_config or {})
        if 0 < self.train_config.early_stopping_patience < 1:
            self.train_config.early_stopping_patience = int(
                self.train_config.early_stopping_patience * self.train_config.epochs
            )

        self._cm = ClassificationMetrics(multi_label=False, macro=False)

        # setup dataloaders
        feature_config = feature_config or {}
        feature_config.update({"over_sampler": over_sampler})
        self.dataset_cls = dataset_cls
        train_dataset = self.dataset_cls(
            preprocess_config=preprocess_config,
            feature_config=feature_config,
            feature_set=feature_set,
            training=True,
        )
        val_dataset = self.dataset_cls(
            preprocess_config=preprocess_config,
            feature_config=feature_config,
            feature_set=feature_set,
            training=False,
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=4,
        )

        # setup model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model_config = model_config or {}
        assert set(model_config.keys()) <= set(
            get_kwargs(model_cls)
        ), f"`model_config` contains invalid keys: {set(model_config.keys()) - set(get_kwargs(model_cls))}"
        self.model = model_cls(
            n_features=train_dataset.n_features,
            n_classes=2,
            **model_config,
        ).to(self.device)

        # setup loss function
        all_y = torch.cat([train_dataset.tensors[1], val_dataset.tensors[1]])
        class_weights = (
            torch.from_numpy(
                compute_class_weight(
                    "balanced", classes=all_y.unique().numpy(), y=all_y.numpy()
                )
            )
            .float()
            .to(self.device)
        )
        del all_y
        print(f"Class weights: {class_weights}")
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device),
            reduction="mean",
            label_smoothing=self.train_config.label_smoothing,
        )

        # setup optimizer and scheduler
        self.optimizer = getattr(torch.optim, self.train_config.optimizer)(
            self.model.parameters(), **self.train_config.optimizer_kw
        )
        if "constant" not in self.train_config.scheduler:
            self.train_config.scheduler_kw.num_training_steps = (
                self.train_config.epochs * len(self.train_dataloader)
            )
        self.scheduler = SeizureTrainer.get_lr_scheduler(
            self.optimizer, self.train_config.scheduler, self.train_config.scheduler_kw
        )

        # cache
        self.best_state_dict = None
        self.best_epoch = 0
        self.best_metric = 0
        self.csv_logger = pd.DataFrame()

    def train(self, epochs: Optional[int] = None) -> None:
        """ """
        self.model.train()
        epochs = epochs or self.train_config.epochs
        pseudo_best_epoch = self.best_epoch  # 0
        with tqdm(range(epochs), desc="Training", mininterval=1, unit="epoch") as pbar:
            for epoch in pbar:
                epoch_loss = 0
                for batch in self.train_dataloader:
                    self.optimizer.zero_grad()
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)
                    epoch_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                epoch_loss /= len(self.train_dataloader)
                metrics = self.evaluate()
                self.model.train()
                metrics["train_loss"] = epoch_loss
                if metrics[self.train_config.monitor] > self.best_metric:
                    self.best_metric = metrics[self.train_config.monitor]
                    self.best_epoch = epoch
                    self.best_state_dict = deepcopy(self.model.state_dict())
                pbar.set_postfix_str(f"epoch_loss: {epoch_loss:.4f}")
                pbar.set_postfix_str(
                    f"best_metric ({self.train_config.monitor}): {self.best_metric:.4f}"
                )
                self.csv_logger = self.csv_logger.append(metrics, ignore_index=True)
                # early stopping callback
                if (
                    self.best_metric - metrics[self.train_config.monitor]
                    <= self.train_config.early_stopping_min_delta
                ):
                    pseudo_best_epoch = epoch
                if (
                    epoch - pseudo_best_epoch
                    >= self.train_config.early_stopping_patience
                ):
                    print(f"Early stopping triggered at epoch {epoch}")
                    pbar.close()
                    break

    @torch.no_grad()
    def evaluate(self) -> dict:
        """ """
        self.model.eval()
        y_true = torch.cat([y for _, y in self.val_dataloader], dim=0).to(self.device)
        y_pred = torch.cat(
            [self.model(x.to(self.device)) for x, _ in self.val_dataloader], dim=0
        )
        metrics = {}
        metrics["val_loss"] = self.criterion(y_pred, y_true).item()
        y_prob = torch.softmax(y_pred, dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        self._cm(y_true, y_prob.argmax(axis=1), num_classes=2)
        metrics.update({f"val_{k}": v for k, v in self._cm._metrics.items()})
        metrics["val_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        return metrics

    @staticmethod
    def get_lr_scheduler(optimizer: Optimizer, name: str, kw: dict) -> LambdaLR:
        """ """
        if name == "constant":
            return get_constant_schedule(optimizer, **kw)
        elif name == "constant_with_warmup":
            return get_constant_schedule_with_warmup(optimizer, **kw)
        elif name == "cosine":
            return get_cosine_schedule_with_warmup(optimizer, **kw)
        elif name == "cosine_with_hard_restarts":
            return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, **kw)
        elif name == "linear":
            return get_linear_schedule_with_warmup(optimizer, **kw)
        elif name == "polynomial_decay":
            return get_polynomial_decay_schedule_with_warmup(optimizer, **kw)
        else:
            raise ValueError(f"Unknown scheduler: {name}")
