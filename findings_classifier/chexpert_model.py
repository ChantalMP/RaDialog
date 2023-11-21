import torch
from torch import nn

from biovil_t.pretrained import get_biovil_t_image_encoder


class ChexpertClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.biovil_encoder = get_biovil_t_image_encoder()

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.biovil_encoder(x).projected_patch_embeddings
        x = torch.nn.functional.avg_pool2d(x, 4)
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        # x = self.biovil_encoder(x).img_embedding
        return self.fc2(x)
