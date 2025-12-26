from torch import nn
import torch


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, device: bool = None):
        super().__init__()

        self.convolution_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, device=device)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, device=device)
        self.relu1 = nn.ReLU()

        self.convolution_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, device=device)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, device=device)

        if in_channels == out_channels:
            self.x_skip_proj = nn.Identity()
        else:
            self.x_skip_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device)

        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x_skip = x.clone() # skip connection

        x = self.convolution_1(x)
        x = self.bn1(x) # post norm
        x = self.relu1(x)

        x = self.convolution_2(x)
        x = self.bn2(x) # post norm

        x_skip = self.x_skip_proj(x_skip) # project x_skip last dimension to x

        return  self.relu2(x + x_skip)
    

if __name__ == "__main__":
    resnet = ResNetBlock(in_channels=3, out_channels=64, device=torch.device("cuda"))
    x = torch.rand(size=(2, 3, 512, 512), device=torch.device("cuda"))
    print(resnet(x))

