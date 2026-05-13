import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_json_or_jsonl(path: Path) -> Any:
    """
    Load a JSON or JSON Lines file.

    Normal JSON example:
        {"a": 1, "b": 2}

    JSON Lines example:
        {"a": 1}
        {"b": 2}
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        data = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {path}"
                    ) from error

        return data


def build_triplets(
        pairs: List[Dict[str, Any]],
        catalog: Dict[str, str],
        seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Build triplet training samples.

    Each triplet has:
        scene: scene image ID
        positive_product: matched product ID
        negative_product: randomly sampled wrong product ID
    """
    random.seed(seed)

    all_product_ids = list(catalog.keys())
    triplets = []

    for pair in pairs:
        scene_id = pair["scene"]
        positive_product_id = pair["product"]

        negative_product_id = random.choice(all_product_ids)

        while negative_product_id == positive_product_id:
            negative_product_id = random.choice(all_product_ids)

        triplet = {
            "scene": scene_id,
            "positive_product": positive_product_id,
            "negative_product": negative_product_id,
        }

        triplets.append(triplet)

    return triplets


def save_jsonl(data: List[Dict[str, str]], output_path: Path) -> None:
    """
    Save data as a JSON Lines file.

    JSON Lines is useful for machine learning datasets because
    each line is one training sample.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def inspect_triplets(triplets: List[Dict[str, str]], num_examples: int = 5) -> None:
    """
    Print a few triplets for checking.
    """
    print("Number of triplets:", len(triplets))
    print("\nExamples:")

    for triplet in triplets[:num_examples]:
        print(triplet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build triplet training data for the STL recommender."
    )

    parser.add_argument(
        "--catalog_file",
        type=str,
        default="data/fashion-cat.json",
        help="Path to the product catalog JSON file.",
    )

    parser.add_argument(
        "--pair_file",
        type=str,
        default="data/fashion.json",
        help="Path to the scene-product pair JSON/JSONL file.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="data/fashion-triplets.jsonl",
        help="Path to the output triplet JSONL file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for negative sampling.",
    )

    args = parser.parse_args()

    catalog_path = Path(args.catalog_file)
    pair_path = Path(args.pair_file)
    output_path = Path(args.output_file)

    catalog = load_json_or_jsonl(catalog_path)
    pairs = load_json_or_jsonl(pair_path)

    if not isinstance(catalog, dict):
        raise TypeError("catalog_file should contain a dictionary.")

    if not isinstance(pairs, list):
        raise TypeError("pair_file should contain a list or JSON Lines records.")

    triplets = build_triplets(
        pairs=pairs,
        catalog=catalog,
        seed=args.seed,
    )

    save_jsonl(triplets, output_path)
    inspect_triplets(triplets)

    print(f"\nSaved triplets to: {output_path}")


if __name__ == "__main__":
    main()