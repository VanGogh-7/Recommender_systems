import json
import random
from pathlib import Path
from typing import Dict, List


TRIPLET_FILE = Path("data/fashion-triplets.jsonl")
CATALOG_FILE = Path("data/fashion-cat.json")
OUTPUT_FILE = Path("triplet_samples.html")


def load_json(path: Path) -> Dict[str, str]:
    """
    Load a normal JSON file.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """
    Load a JSON Lines file.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def pinterest_key_to_url(key: str) -> str:
    """
    Convert a Pinterest image key to a Pinterest image URL.
    """
    return f"https://i.pinimg.com/564x/{key[0:2]}/{key[2:4]}/{key[4:6]}/{key}.jpg"


def sample_triplets(
        triplets: List[Dict[str, str]],
        num_samples: int = 20,
        seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Randomly sample triplets for visual inspection.
    """
    random.seed(seed)
    copied = triplets.copy()
    random.shuffle(copied)
    return copied[:num_samples]


def write_triplets_to_html(
        triplets: List[Dict[str, str]],
        catalog: Dict[str, str],
        output_path: Path,
) -> None:
    """
    Write triplet samples to an HTML file.
    """
    rows = []

    for triplet in triplets:
        scene_id = triplet["scene"]
        positive_id = triplet["positive_product"]
        negative_id = triplet["negative_product"]

        positive_category = catalog.get(positive_id, "Unknown")
        negative_category = catalog.get(negative_id, "Unknown")

        scene_url = pinterest_key_to_url(scene_id)
        positive_url = pinterest_key_to_url(positive_id)
        negative_url = pinterest_key_to_url(negative_id)

        row = f"""
        <tr>
            <td>
                <div><b>Scene</b></div>
                <div>{scene_id}</div>
                <img src="{scene_url}" width="240" style="max-height: 280px; object-fit: contain;">
            </td>

            <td>
                <div><b>Positive Product</b></div>
                <div>{positive_id}</div>
                <div>{positive_category}</div>
                <img src="{positive_url}" width="180" style="max-height: 240px; object-fit: contain;">
            </td>

            <td>
                <div><b>Negative Product</b></div>
                <div>{negative_id}</div>
                <div>{negative_category}</div>
                <img src="{negative_url}" width="180" style="max-height: 240px; object-fit: contain;">
            </td>
        </tr>
        """

        rows.append(row)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Triplet Samples</title>
    </head>
    <body>
        <h1>Triplet Samples</h1>

        <p>
            Each row shows one training triplet:
            scene image, positive product, and negative product.
        </p>

        <table border="1" cellpadding="8">
            <tr>
                <th>Scene</th>
                <th>Positive Product</th>
                <th>Negative Product</th>
            </tr>
            {''.join(rows)}
        </table>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    catalog = load_json(CATALOG_FILE)
    triplets = load_jsonl(TRIPLET_FILE)

    sampled = sample_triplets(
        triplets=triplets,
        num_samples=20,
        seed=42,
    )

    write_triplets_to_html(
        triplets=sampled,
        catalog=catalog,
        output_path=OUTPUT_FILE,
    )

    print(f"Saved triplet samples to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()