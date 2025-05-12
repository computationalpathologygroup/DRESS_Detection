import torch
import torch.nn as nn

from models.model_clam import CLAM_SB


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
    
model = CLAM_SB(
    gate=True,
    size_arg="small",
    dropout=0.25,
    k_sample=50,
    n_classes=2,
    subtyping=False,
    embed_dim=1536
)

checkpoint_path = "checkpoints/uni/s_0_checkpoint_Gigapath.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(checkpoint, strict=False)  # or checkpoint['model'] if it's nested

# Step 3: View architecture
print(model)