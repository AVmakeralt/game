"""Minimal but complete training pipeline for SmallNet and MegaNet."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from nets import MegaNet, NetConfig, SmallNet


@dataclass
class TrainBatch:
    planes: Tensor
    policy_target: Tensor
    value_target: Tensor


class NPZChessDataset(Dataset[TrainBatch]):
    """Dataset backed by a .npz created from self-play or supervised labels.

    Required arrays:
      - planes: float32 [N, C, 8, 8]
      - policy: int64 [N]  (index in [0, 4095])
      - value: float32 [N] (in [-1, 1])
    """

    def __init__(self, npz_path: Path) -> None:
        super().__init__()
        raw = dict(**__import__("numpy").load(npz_path))
        self.planes = torch.from_numpy(raw["planes"]).float()
        self.policy = torch.from_numpy(raw["policy"]).long()
        self.value = torch.from_numpy(raw["value"]).float().view(-1, 1)
        if self.planes.ndim != 4:
            raise ValueError("planes must be [N, C, 8, 8]")
        if self.policy.ndim != 1:
            raise ValueError("policy must be [N]")
        if self.value.ndim != 2:
            raise ValueError("value must be [N, 1]")

    def __len__(self) -> int:
        return int(self.planes.shape[0])

    def __getitem__(self, index: int) -> TrainBatch:
        return TrainBatch(
            planes=self.planes[index],
            policy_target=self.policy[index],
            value_target=self.value[index],
        )


def collate(batch: list[TrainBatch]) -> TrainBatch:
    return TrainBatch(
        planes=torch.stack([b.planes for b in batch], dim=0),
        policy_target=torch.stack([b.policy_target for b in batch], dim=0),
        value_target=torch.stack([b.value_target for b in batch], dim=0),
    )


def build_model(kind: str, input_planes: int) -> torch.nn.Module:
    cfg = NetConfig(input_planes=input_planes)
    if kind == "small":
        return SmallNet(cfg)
    if kind == "mega":
        return MegaNet(cfg)
    raise ValueError(f"unknown model kind: {kind}")


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader[TrainBatch],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_weight: float,
) -> float:
    model.train()
    running = 0.0
    for batch in loader:
        planes = batch.planes.to(device)
        policy_target = batch.policy_target.to(device)
        value_target = batch.value_target.to(device)

        optimizer.zero_grad(set_to_none=True)
        policy_logits, value_pred = model(planes)
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        value_loss = F.mse_loss(value_pred, value_target)
        loss = policy_loss + value_weight * value_loss
        loss.backward()
        optimizer.step()
        running += float(loss.detach().cpu())
    return running / max(1, len(loader))


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader[TrainBatch], device: torch.device, value_weight: float) -> float:
    model.eval()
    running = 0.0
    for batch in loader:
        planes = batch.planes.to(device)
        policy_target = batch.policy_target.to(device)
        value_target = batch.value_target.to(device)
        policy_logits, value_pred = model(planes)
        policy_loss = F.cross_entropy(policy_logits, policy_target)
        value_loss = F.mse_loss(value_pred, value_target)
        running += float((policy_loss + value_weight * value_loss).cpu())
    return running / max(1, len(loader))


def iter_loader(dataset: Dataset[TrainBatch], batch_size: int, shuffle: bool) -> DataLoader[TrainBatch]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MegaNet or SmallNet.")
    parser.add_argument("--model", choices=["small", "mega"], required=True)
    parser.add_argument("--train", type=Path, required=True, help="Path to training .npz")
    parser.add_argument("--valid", type=Path, required=True, help="Path to validation .npz")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("weights.pt"))
    args = parser.parse_args()

    train_ds = NPZChessDataset(args.train)
    valid_ds = NPZChessDataset(args.valid)
    input_planes = int(train_ds.planes.shape[1])
    model = build_model(args.model, input_planes=input_planes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = iter_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    valid_loader = iter_loader(valid_ds, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.value_weight)
        valid_loss = validate(model, valid_loader, device, args.value_weight)
        print(f"epoch={epoch} train_loss={train_loss:.5f} valid_loss={valid_loss:.5f}")
        if valid_loss < best:
            best = valid_loss
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": args.model,
                    "input_planes": input_planes,
                    "state_dict": model.state_dict(),
                    "best_valid_loss": best,
                },
                args.out,
            )
            print(f"saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
