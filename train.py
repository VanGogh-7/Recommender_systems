import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import STLTripletDataset
from model import STLTwoTowerModel


def triplet_ranking_loss(
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
        margin: float = 1.0,
) -> torch.Tensor:
    """
    Compute triplet ranking loss.

    The goal is:
        positive_score > negative_score + margin

    Loss:
        max(0, margin + negative_score - positive_score)
    """
    loss = torch.relu(margin + negative_score - positive_score)
    return loss.mean()


def compute_accuracy(
        positive_score: torch.Tensor,
        negative_score: torch.Tensor,
) -> float:
    """
    Compute the fraction of samples where the positive score
    is larger than the negative score.
    """
    correct = positive_score > negative_score
    return correct.float().mean().item()


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        margin: float,
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for scene, positive_product, negative_product in dataloader:
        scene = scene.to(device)
        positive_product = positive_product.to(device)
        negative_product = negative_product.to(device)

        output = model(scene, positive_product, negative_product)

        positive_score = output["positive_score"]
        negative_score = output["negative_score"]

        loss = triplet_ranking_loss(
            positive_score=positive_score,
            negative_score=negative_score,
            margin=margin,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(positive_score, negative_score)

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

    return total_loss / num_batches, total_accuracy / num_batches


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        margin: float,
) -> tuple[float, float]:
    """
    Evaluate the model.
    """
    model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for scene, positive_product, negative_product in dataloader:
        scene = scene.to(device)
        positive_product = positive_product.to(device)
        negative_product = negative_product.to(device)

        output = model(scene, positive_product, negative_product)

        positive_score = output["positive_score"]
        negative_score = output["negative_score"]

        loss = triplet_ranking_loss(
            positive_score=positive_score,
            negative_score=negative_score,
            margin=margin,
        )

        accuracy = compute_accuracy(positive_score, negative_score)

        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

    return total_loss / num_batches, total_accuracy / num_batches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a two-tower CNN model with triplet ranking loss."
    )

    parser.add_argument("--triplet_file", type=str, default="data/fashion-triplets.jsonl")
    parser.add_argument("--image_dir", type=str, default="data/images")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="checkpoints/stl_two_tower.pt")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = STLTripletDataset(
        triplet_file=args.triplet_file,
        image_dir=args.image_dir,
        image_size=args.image_size,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = STLTwoTowerModel(
        embedding_dim=args.embedding_dim,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            margin=args.margin,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            margin=args.margin,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "embedding_dim": args.embedding_dim,
                    "image_size": args.image_size,
                },
                save_path,
            )

            print(f"Saved best model to: {save_path}")


if __name__ == "__main__":
    main()