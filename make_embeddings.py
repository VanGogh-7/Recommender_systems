import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import STLTwoTowerModel


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """
    Load a JSON Lines file.
    """
    data = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def collect_unique_ids(triplets: List[Dict[str, str]]) -> tuple[List[str], List[str]]:
    """
    Collect unique scene IDs and product IDs from triplets.
    """
    scene_ids = set()
    product_ids = set()

    for triplet in triplets:
        scene_ids.add(triplet["scene"])
        product_ids.add(triplet["positive_product"])
        product_ids.add(triplet["negative_product"])

    return sorted(scene_ids), sorted(product_ids)


def build_transform(image_size: int):
    """
    Build image preprocessing transform.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ]
    )


def load_image(image_dir: Path, image_id: str, transform) -> torch.Tensor:
    """
    Load one image and convert it into a tensor.
    """
    image_path = image_dir / f"{image_id}.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = transform(image)

    return image


@torch.no_grad()
def compute_embeddings(
        model: STLTwoTowerModel,
        image_ids: List[str],
        image_dir: Path,
        transform,
        encoder_type: str,
        device: torch.device,
        batch_size: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute embeddings for a list of images.

    encoder_type:
        "scene"   -> use scene encoder
        "product" -> use product encoder
    """
    model.eval()

    all_ids = []
    all_embeddings = []

    for start in tqdm(range(0, len(image_ids), batch_size), desc=f"Embedding {encoder_type}s"):
        batch_ids = image_ids[start:start + batch_size]

        images = [
            load_image(image_dir, image_id, transform)
            for image_id in batch_ids
        ]

        images = torch.stack(images, dim=0).to(device)

        if encoder_type == "scene":
            embeddings = model.scene_encoder(images)
        elif encoder_type == "product":
            embeddings = model.product_encoder(images)
        else:
            raise ValueError("encoder_type must be 'scene' or 'product'.")

        all_ids.extend(batch_ids)
        all_embeddings.append(embeddings.cpu())

    embedding_tensor = torch.cat(all_embeddings, dim=0)

    return {
        "ids": all_ids,
        "embeddings": embedding_tensor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute scene and product embeddings for STL recommender."
    )

    parser.add_argument(
        "--triplet_file",
        type=str,
        default="data/fashion-triplets-valid.jsonl",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/images",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stl_two_tower.pt",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="embeddings",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    triplet_file = Path(args.triplet_file)
    image_dir = Path(args.image_dir)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    embedding_dim = checkpoint["embedding_dim"]
    image_size = checkpoint["image_size"]

    model = STLTwoTowerModel(
        embedding_dim=embedding_dim,
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    triplets = load_jsonl(triplet_file)
    scene_ids, product_ids = collect_unique_ids(triplets)

    print("Number of unique scenes:", len(scene_ids))
    print("Number of unique products:", len(product_ids))

    transform = build_transform(image_size)

    scene_data = compute_embeddings(
        model=model,
        image_ids=scene_ids,
        image_dir=image_dir,
        transform=transform,
        encoder_type="scene",
        device=device,
        batch_size=args.batch_size,
    )

    product_data = compute_embeddings(
        model=model,
        image_ids=product_ids,
        image_dir=image_dir,
        transform=transform,
        encoder_type="product",
        device=device,
        batch_size=args.batch_size,
    )

    scene_output = output_dir / "scene_embeddings.pt"
    product_output = output_dir / "product_embeddings.pt"

    torch.save(scene_data, scene_output)
    torch.save(product_data, product_output)

    print(f"Saved scene embeddings to: {scene_output}")
    print(f"Saved product embeddings to: {product_output}")


if __name__ == "__main__":
    main()