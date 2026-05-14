import torch
from torch import nn
from torchvision import models


class ImageEncoder(nn.Module):
    """
    CNN image encoder.

    It converts an image into a fixed-size embedding vector.
    """

    def __init__(
            self,
            embedding_dim: int = 128,
            pretrained: bool = True,
            freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        num_features = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.backbone = resnet

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to normalized embedding vectors.
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings


class STLTwoTowerModel(nn.Module):
    """
    Two-tower model for STL scene-product recommendation.
    """

    def __init__(
            self,
            embedding_dim: int = 128,
            pretrained: bool = True,
            freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.scene_encoder = ImageEncoder(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

        self.product_encoder = ImageEncoder(
            embedding_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    def forward(
            self,
            scene: torch.Tensor,
            positive_product: torch.Tensor,
            negative_product: torch.Tensor,
    ):
        """
        Compute positive and negative compatibility scores.
        """
        scene_embedding = self.scene_encoder(scene)
        positive_embedding = self.product_encoder(positive_product)
        negative_embedding = self.product_encoder(negative_product)

        positive_score = torch.sum(scene_embedding * positive_embedding, dim=1)
        negative_score = torch.sum(scene_embedding * negative_embedding, dim=1)

        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "scene_embedding": scene_embedding,
            "positive_embedding": positive_embedding,
            "negative_embedding": negative_embedding,
        }
