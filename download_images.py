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


def collect_image_ids(triplets: List[Dict[str, str]]) -> Set[str]:
    """
    Collect all unique image IDs from triplets.
    """
    image_ids = set()

    for triplet in triplets:
        image_ids.add(triplet["scene"])
        image_ids.add(triplet["positive_product"])
        image_ids.add(triplet["negative_product"])

    return image_ids


def download_image(image_id: str, output_dir: Path, timeout: int = 20) -> bool:
    """
    Download one image by its Pinterest image ID.

    Returns True if the image exists locally after this function.
    Returns False if the download failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_id}.jpg"

    if output_path.exists() and output_path.stat().st_size > 0:
        return True

    url = pinterest_key_to_url(image_id)

    try:
        urllib.request.urlretrieve(url, output_path)

        if output_path.exists() and output_path.stat().st_size > 0:
            return True

        return False

    except Exception as error:
        print(f"Failed to download {image_id}: {error}")

        if output_path.exists():
            output_path.unlink()

        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download STL images used by triplet training data."
    )

    parser.add_argument(
        "--triplet_file",
        type=str,
        default="data/fashion-triplets.jsonl",
        help="Path to the triplet JSONL file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/images",
        help="Directory used to save downloaded images.",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=500,
        help="Maximum number of images to download. Use -1 for all images.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep time between downloads, in seconds.",
    )

    args = parser.parse_args()

    triplet_file = Path(args.triplet_file)
    output_dir = Path(args.output_dir)

    triplets = load_jsonl(triplet_file)
    image_ids = sorted(collect_image_ids(triplets))

    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]

    print(f"Number of images to download: {len(image_ids)}")
    print(f"Output directory: {output_dir}")

    success_count = 0
    failure_count = 0

    for index, image_id in enumerate(image_ids, start=1):
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
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()