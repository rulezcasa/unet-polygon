'''Added aggressive commenting to explain my logic.'''

#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Two 3x3 convolutions with activated by ReLU - Pretty standard Unet configuration, didn't experiment with this.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# Encoder Block : Compresses the image to a rich feature representation by max pooling. (Reduces spatial resolution by half)
# Feature extraction using DoubleConv. (Increases the number of feature channels).
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# Decoder Block : Decompresses and increases spatial representation by upsampling (by a factor of 2).
# Skip connections to concatenate the symmetrical features of the encoder onto the decoder.
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Skip connection
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

# Main UNet Block 
class ConditionalUNet(nn.Module):
    def __init__(self, n_classes=3, color_vocab_size=10, embed_dim=16):
        super().__init__()

        # Represents the color into a learnable dense vector (embedding)
        self.embedding = nn.Embedding(color_vocab_size, embed_dim)

        #Initial conv block :
            # Input channels : 3(RGB) + tiled color embeddings
            # Output : 64 feature maps
        self.inc = DoubleConv(3 + embed_dim, 64)
        
        # Decoder segments : Increasing feature maps while reducing resolution.
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        # Encoder segments : Increasing resolution while halving feature representation concatenated by skip connections.
        self.up1 = Up(512 + 256, 256)
        self.up2 = Up(256 + 128, 128)
        self.up3 = Up(128 + 64, 64)
        
        
        # Final output layer
           # in_channels = 64 (flowing from the previous layer)
           # out_channels = n_classes (3 for RGB images)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x, color_id):
        batch_size, _, h, w = x.shape
        embed = self.embedding(color_id).unsqueeze(2).unsqueeze(3).expand(-1, -1, h, w)
        x = torch.cat([x, embed], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return torch.sigmoid(self.outc(x))
