import argparse
import json
from pathlib import Path
from typing import Dict, List


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


def save_jsonl(data: List[Dict[str, str]], path: Path) -> None:
    """
    Save records to a JSON Lines file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def image_exists(image_dir: Path, image_id: str) -> bool:
    """
    Check whether an image exists locally.
    """
    return (image_dir / f"{image_id}.jpg").exists()


def filter_valid_triplets(
        triplets: List[Dict[str, str]],
        image_dir: Path,
) -> List[Dict[str, str]]:
    """
    Keep only triplets whose scene, positive product, and negative product
    all exist locally.
    """
    valid_triplets = []

    for triplet in triplets:
        scene_ok = image_exists(image_dir, triplet["scene"])
        positive_ok = image_exists(image_dir, triplet["positive_product"])
        negative_ok = image_exists(image_dir, triplet["negative_product"])

        if scene_ok and positive_ok and negative_ok:
            valid_triplets.append(triplet)

    return valid_triplets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter triplets by locally downloaded images."
    )

    parser.add_argument(
        "--triplet_file",
        type=str,
        default="data/fashion-triplets.jsonl",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/images",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/fashion-triplets-valid.jsonl",
    )

    args = parser.parse_args()

    triplet_file = Path(args.triplet_file)
    image_dir = Path(args.image_dir)
    output_file = Path(args.output_file)

    triplets = load_jsonl(triplet_file)
    valid_triplets = filter_valid_triplets(triplets, image_dir)

    save_jsonl(valid_triplets, output_file)

    print("Original triplets:", len(triplets))
    print("Valid triplets:", len(valid_triplets))
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()

