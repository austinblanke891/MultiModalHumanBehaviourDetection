import torch
import torch.nn as nn
import torch.nn.functional as F

class GazeFieldNet(nn.Module):
    def __init__(self):
        super(GazeFieldNet, self).__init__()

        # Full-frame image encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Head crop encoder
        self.head_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.head_bn1 = nn.BatchNorm2d(32)
        self.head_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.head_bn2 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16 + 64 * 16 * 16 + 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64 * 64)

    def forward(self, image, head_pos, head_crop):
        # Encode full frame
        x = F.relu(self.bn1(self.conv1(image)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        # Encode head crop
        h = F.relu(self.head_bn1(self.head_conv1(head_crop)))
        h = F.relu(self.head_bn2(self.head_conv2(h)))
        h = h.view(h.size(0), -1)

        # Combine features + head position
        combined = torch.cat([x, h, head_pos], dim=1)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        out = torch.sigmoid(self.fc3(combined))
        return out.view(-1, 1, 64, 64)
