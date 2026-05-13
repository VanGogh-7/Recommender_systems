import json
import random
from pathlib import Path
from typing import Any, Dict, List


DATA_DIR = Path("data")
CATALOG_FILE = DATA_DIR / "fashion-cat.json"
PAIR_FILE = DATA_DIR / "fashion.json"
OUTPUT_FILE = Path("scene_product_pairs.html")


def load_json_or_jsonl(path: Path) -> Any:
    """
    Load a JSON or JSON Lines file.
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


def pinterest_key_to_url(key: str) -> str:
    """
    Convert a Pinterest image key to a Pinterest image URL.
    """
    return f"https://i.pinimg.com/564x/{key[0:2]}/{key[2:4]}/{key[4:6]}/{key}.jpg"


def sample_pairs(pairs: List[Dict[str, Any]], num_pairs: int = 20, seed: int = 42):
    """
    Randomly sample scene-product pairs.
    """
    random.seed(seed)
    sampled = pairs.copy()
    random.shuffle(sampled)
    return sampled[:num_pairs]


def write_pairs_to_html(
        pairs: List[Dict[str, Any]],
        catalog: Dict[str, str],
        output_path: Path,
) -> None:
    """
    Write scene-product pairs to an HTML file.
    """
    rows = []

    for pair in pairs:
        scene_id = pair["scene"]
        product_id = pair["product"]
        category = catalog.get(product_id, "Unknown")

        scene_url = pinterest_key_to_url(scene_id)
        product_url = pinterest_key_to_url(product_id)

        row = f"""
        <tr>
            <td>{scene_id}</td>
            <td><img src="{scene_url}" width="240" style="max-height: 280px; object-fit: contain;"></td>
            <td>{product_id}</td>
            <td>{category}</td>
            <td><img src="{product_url}" width="180" style="max-height: 240px; object-fit: contain;"></td>
        </tr>
        """

        rows.append(row)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>STL Scene-Product Pairs</title>
    </head>
    <body>
        <h1>STL Scene-Product Pairs</h1>

        <p>This page shows positive scene-product pairs from fashion.json.</p>

        <table border="1" cellpadding="8">
            <tr>
                <th>Scene ID</th>
                <th>Scene Image</th>
                <th>Product ID</th>
                <th>Product Category</th>
                <th>Positive Product Image</th>
            </tr>
            {''.join(rows)}
        </table>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    catalog = load_json_or_jsonl(CATALOG_FILE)
    pairs = load_json_or_jsonl(PAIR_FILE)

    sampled_pairs = sample_pairs(
        pairs=pairs,
        num_pairs=20,
        seed=42,
    )

    write_pairs_to_html(
        pairs=sampled_pairs,
        catalog=catalog,
        output_path=OUTPUT_FILE,
    )

    print(f"Saved scene-product pairs to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
    