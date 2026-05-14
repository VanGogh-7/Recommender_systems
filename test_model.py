import torch
from torch.utils.data import DataLoader

from dataset import STLTripletDataset
from model import STLTwoTowerModel


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = STLTripletDataset(
        triplet_file="data/fashion-triplets.jsonl",
        image_dir="data/images",
        image_size=224,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )

    scene, positive_product, negative_product = next(iter(dataloader))

    scene = scene.to(device)
    positive_product = positive_product.to(device)
    negative_product = negative_product.to(device)

    model = STLTwoTowerModel(
        embedding_dim=128,
        pretrained=True,
    ).to(device)

    model.eval()

    with torch.no_grad():
        output = model(scene, positive_product, negative_product)

    print("Positive score shape:", output["positive_score"].shape)
    print("Negative score shape:", output["negative_score"].shape)
    print("Scene embedding shape:", output["scene_embedding"].shape)
    print("Positive embedding shape:", output["positive_embedding"].shape)
    print("Negative embedding shape:", output["negative_embedding"].shape)

    print("Positive scores:", output["positive_score"])
    print("Negative scores:", output["negative_score"])


if __name__ == "__main__":
    main()