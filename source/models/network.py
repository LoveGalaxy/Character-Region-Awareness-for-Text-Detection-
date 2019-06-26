import torch
import torch.nn as nn

class Base_with_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, up=False):
        super(Base_with_bn_block, self).__init__()
        self.up = up
        if up:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size/2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()
    
    def forward(self, x):

        if self.up:
            x = self.up(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Base_down_block(nn.Module):
    def __init__(self, in_channels, out_channels, times):
        super(Base_down_block, self).__init__()
        
        self.blocks = [Base_with_bn_block(in_channels, out_channels, 3)]
        for i in range(times-1):
            self.blocks += [Base_with_bn_block(out_channels, out_channels, 3)]
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x):
        out = self.blocks(x)
        return out

class Base_up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Base_up_block, self).__init__()
        self.block1 = Base_with_bn_block(in_channels, out_channels*2, 1, up=True)
        self.block2 = Base_with_bn_block(out_channels*2, out_channels, 3)
    def forward(self, x1, x2):
        out = torch.cat([x1, x2], 1)
        out = self.block1(out)
        out = self.block2(out)
        return out

class UP_VGG(nn.Module):
    def __init__(self):
        super(UP_VGG, self).__init__()
        self.layers = nn.Sequential(*[Base_down_block(3, 64, 2),
                        Base_down_block(64, 128, 2),
                        Base_down_block(128, 256, 3),
                        Base_down_block(256, 512, 3),
                        Base_down_block(512, 512, 3),
                        Base_down_block(512, 512, 3),])

        self.up_layers = nn.Sequential(*[Base_up_block(512+512, 256),
                        Base_up_block(512+256, 128),
                        Base_up_block(256+128, 64),
                        Base_up_block(128+64, 32)])

        self.detector = nn.Sequential(*[Base_with_bn_block(32, 32, 3),
                        Base_with_bn_block(32, 32, 3),
                        Base_with_bn_block(32, 16, 3),
                        Base_with_bn_block(16, 16, 1)])
        
        self.region = nn.Conv2d(16, 1, 1)
        self.affinity = nn.Conv2d(16, 1, 1)

        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        features = []
        for i in range(5):
            x = self.layers[i](x)
            x = self.pooling(x)
            features.append(x)
        x = self.layers[-1](x)
        for index in range(4):
            x = self.up_layers[index](features[-index-1], x)
        x = self.detector(x)
        reg = self.region(x)
        aff = self.affinity(x)
        return reg, aff

if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256).to("cuda")
    net = UP_VGG().to("cuda")
    reg, aff = net(x)
    print(net)
    print(reg.shape, aff.shape)