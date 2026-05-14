import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Set


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


def collect_images_from_first_triplets(
        triplets: List[Dict[str, str]],
        max_triplets: int,
) -> Set[str]:
    """
    Collect image IDs from the first max_triplets triplets.

    This guarantees that if downloads succeed, we will have complete triplets.
    """
    selected_triplets = triplets[:max_triplets]
    image_ids = set()

    for triplet in selected_triplets:
        image_ids.add(triplet["scene"])
        image_ids.add(triplet["positive_product"])
        image_ids.add(triplet["negative_product"])

    return image_ids


def download_image(image_id: str, output_dir: Path, timeout: int = 20) -> bool:
    """
    Download one image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_id}.jpg"

    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    url = pinterest_key_to_url(image_id)

    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            },
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            image_bytes = response.read()

        if len(image_bytes) == 0:
            return False

        output_path.write_bytes(image_bytes)
        return True

    except Exception as error:
        print(f"Failed to download {image_id}: {error}")

        if output_path.exists():
            output_path.unlink()

        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download images needed for complete triplets."
    )

    parser.add_argument(
        "--triplet_file",
        type=str,
        default="data/fashion-triplets.jsonl",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/images",
    )

    parser.add_argument(
        "--max_triplets",
        type=int,
        default=500,
        help="Download images needed by the first N triplets.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    triplets = load_jsonl(Path(args.triplet_file))
    image_ids = collect_images_from_first_triplets(
        triplets=triplets,
        max_triplets=args.max_triplets,
    )

    output_dir = Path(args.output_dir)

    print(f"Selected triplets: {args.max_triplets}")
    print(f"Unique images needed: {len(image_ids)}")
    print(f"Output directory: {output_dir}")

    success_count = 0
    failure_count = 0

    for index, image_id in enumerate(sorted(image_ids), start=1):
        ok = download_image(image_id, output_dir)

        if ok:
            success_count += 1
        else:
            failure_count += 1

        if index % 50 == 0:
            print(
                f"Progress: {index}/{len(image_ids)} | "
                f"success: {success_count} | failed: {failure_count}"
            )

        time.sleep(args.sleep)

    print("\nFinished.")
    print(f"Successful downloads: {success_count}")
    print(f"Failed downloads: {failure_count}")


if __name__ == "__main__":
    main()