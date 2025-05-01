import torch
import torch.nn as nn


class ZoomFusionClassifier(nn.Module):
    def __init__(self, feature_dim=1536, n_classes=2, fusion='avg'):
        super().__init__()
        self.fusion = fusion
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, feats_10x, feats_20x):
        if self.fusion == 'avg':
            fused = (feats_10x + feats_20x) / 2
        elif self.fusion == 'sum':
            fused = feats_10x + feats_20x
        else:
            raise ValueError(
                "Invalid fusion method. Choose 'avg', 'sum'")

        pooled = fused.mean(dim=0, keepdim=True)  # Global average pooling
        logits = self.classifier(pooled)  # Shape [1, n_classes]
        probs = torch.softmax(logits, dim=1)
        return probs, logits
