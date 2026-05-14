from torch.utils.data import DataLoader

from dataset import STLTripletDataset


def main() -> None:
    dataset = STLTripletDataset(
        triplet_file="data/fashion-triplets.jsonl",
        image_dir="data/images",
        image_size=224,
    )

    print("Dataset size:", len(dataset))

    scene, positive_product, negative_product = dataset[0]

    print("Scene shape:", scene.shape)
    print("Positive product shape:", positive_product.shape)
    print("Negative product shape:", negative_product.shape)

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
    )

    batch_scene, batch_positive, batch_negative = next(iter(dataloader))

    print("Batch scene shape:", batch_scene.shape)
    print("Batch positive shape:", batch_positive.shape)
    print("Batch negative shape:", batch_negative.shape)


if __name__ == "__main__":
    main()