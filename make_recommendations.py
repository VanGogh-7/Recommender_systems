import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


def load_json(path: Path) -> Dict[str, str]:
    """
    Load a normal JSON file.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_embeddings(path: Path) -> Dict:
    """
    Load saved embedding data.
    """
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    return torch.load(path, map_location="cpu")


def pinterest_key_to_url(key: str) -> str:
    """
    Convert a Pinterest image key to a Pinterest image URL.
    """
    return f"https://i.pinimg.com/564x/{key[0:2]}/{key[2:4]}/{key[4:6]}/{key}.jpg"


def recommend_for_scene(
        scene_id: str,
        scene_data: Dict,
        product_data: Dict,
        top_k: int = 10,
) -> List[Dict]:
    """
    Recommend top-k products for one scene.
    """
    scene_ids = scene_data["ids"]
    scene_embeddings = scene_data["embeddings"]

    product_ids = product_data["ids"]
    product_embeddings = product_data["embeddings"]

    if scene_id not in scene_ids:
        raise ValueError(f"Scene ID not found in scene embeddings: {scene_id}")

    scene_index = scene_ids.index(scene_id)
    scene_embedding = scene_embeddings[scene_index]

    scores = product_embeddings @ scene_embedding

    top_scores, top_indices = torch.topk(scores, k=top_k)

    recommendations = []

    for score, index in zip(top_scores, top_indices):
        product_id = product_ids[index.item()]

        recommendations.append(
            {
                "product_id": product_id,
                "score": float(score.item()),
            }
        )

    return recommendations


def write_recommendations_html(
        scene_id: str,
        recommendations: List[Dict],
        catalog: Dict[str, str],
        output_path: Path,
) -> None:
    """
    Write recommendations to an HTML file.
    """
    scene_url = pinterest_key_to_url(scene_id)

    product_cards = []

    for rank, item in enumerate(recommendations, start=1):
        product_id = item["product_id"]
        score = item["score"]
        category = catalog.get(product_id, "Unknown")
        product_url = pinterest_key_to_url(product_id)

        card = f"""
        <td style="padding: 12px; text-align: center; vertical-align: top;">
            <div><b>Rank {rank}</b></div>
            <div>Score: {score:.4f}</div>
            <div style="font-size: 12px;">{category}</div>
            <img src="{product_url}" width="180" style="max-height: 240px; object-fit: contain;">
            <div style="font-size: 10px; word-break: break-all;">{product_id}</div>
        </td>
        """

        product_cards.append(card)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>STL Recommendations</title>
    </head>
    <body>
        <h1>STL Scene-to-Product Recommendations</h1>

        <h2>Query Scene</h2>
        <div>
            <img src="{scene_url}" width="320" style="max-height: 360px; object-fit: contain;">
            <div>{scene_id}</div>
        </div>

        <h2>Recommended Products</h2>
        <table border="1">
            <tr>
                {''.join(product_cards)}
            </tr>
        </table>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate product recommendations for one STL scene."
    )

    parser.add_argument(
        "--scene_embeddings",
        type=str,
        default="embeddings/scene_embeddings.pt",
    )

    parser.add_argument(
        "--product_embeddings",
        type=str,
        default="embeddings/product_embeddings.pt",
    )

    parser.add_argument(
        "--catalog_file",
        type=str,
        default="data/fashion-cat.json",
    )

    parser.add_argument(
        "--scene_id",
        type=str,
        default=None,
        help="Scene ID. If not provided, the first scene will be used.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default="recommendations.html",
    )

    args = parser.parse_args()

    scene_data = load_embeddings(Path(args.scene_embeddings))
    product_data = load_embeddings(Path(args.product_embeddings))
    catalog = load_json(Path(args.catalog_file))

    if args.scene_id is None:
        scene_id = scene_data["ids"][0]
    else:
        scene_id = args.scene_id

    recommendations = recommend_for_scene(
        scene_id=scene_id,
        scene_data=scene_data,
        product_data=product_data,
        top_k=args.top_k,
    )

    write_recommendations_html(
        scene_id=scene_id,
        recommendations=recommendations,
        catalog=catalog,
        output_path=Path(args.output_file),
    )

    print(f"Scene ID: {scene_id}")
    print(f"Saved recommendations to: {args.output_file}")


if __name__ == "__main__":
    main()