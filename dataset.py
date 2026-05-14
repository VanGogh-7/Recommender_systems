import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class STLTripletDataset(Dataset):
    """
    Dataset for STL scene-product triplet training.

    Each sample contains:
        scene image
        positive product image
        negative product image
    """

    def __init__(
            self,
            triplet_file: str = "data/fashion-triplets.jsonl",
            image_dir: str = "data/images",
            image_size: int = 224,
    ) -> None:
        self.triplet_file = Path(triplet_file)
        self.image_dir = Path(image_dir)

        if not self.triplet_file.exists():
            raise FileNotFoundError(f"Triplet file not found: {self.triplet_file}")

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.triplets = self._load_triplets(self.triplet_file)
        self.triplets = self._filter_existing_images(self.triplets)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

        print(f"Loaded valid triplets: {len(self.triplets)}")

    def _load_triplets(self, path: Path) -> List[Dict[str, str]]:
        """
        Load triplets from a JSON Lines file.
        """
        triplets = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                triplets.append(json.loads(line))

        return triplets

    def _image_path(self, image_id: str) -> Path:
        """
        Convert image ID to local image path.
        """
        return self.image_dir / f"{image_id}.jpg"

    def _filter_existing_images(
            self,
            triplets: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Keep only triplets whose three images all exist locally.
        """
        valid_triplets = []

        for triplet in triplets:
            scene_path = self._image_path(triplet["scene"])
            positive_path = self._image_path(triplet["positive_product"])
            negative_path = self._image_path(triplet["negative_product"])

            if scene_path.exists() and positive_path.exists() and negative_path.exists():
                valid_triplets.append(triplet)

        return valid_triplets

    def _load_image(self, image_id: str) -> torch.Tensor:
        """
        Load one image and transform it into a tensor.
        """
        path = self._image_path(image_id)

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image

    def __len__(self) -> int:
        """
        Return the number of valid triplets.
        """
        return len(self.triplets)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return one triplet sample.
        """
        triplet = self.triplets[index]

        scene = self._load_image(triplet["scene"])
        positive_product = self._load_image(triplet["positive_product"])
        negative_product = self._load_image(triplet["negative_product"])

        return scene, positive_product, negative_product

    