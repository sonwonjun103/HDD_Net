import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 encoder=True):
        super().__init__()
        self.encoder = encoder
        if not mid_channels:
            mid_channels = out_channels

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.block(x)
        return x
    
class Down(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.block = ConvBlock(in_channel,
                               out_channel)
        
    def forward(self, x):
        x = self.maxpool(x)
        x = self.block(x)

        return x

# ======================================================================================= #
# Decoder
class Up(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.block = ConvBlock(in_channel,
                               out_channel,
                               mid_channels = in_channel // 2,
                               encoder=False)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2 ,x1], dim=1) 

        return self.block(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
       
    def forward(self, x):
        
        return self.block(x)
    
class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
        )
       
    def forward(self, x):
        return self.block(x)

# ======================================================================================= #

class Residual(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 f=None):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.f = F
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.block(x)

        # if self.f:
        #     x = self.maxpool(x)
        return x
    
class CA(nn.Module):
    def __init__(self,
                 in_channel):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // 2, in_channel),
            #nn.Sigmoid()
        )

    def forward(self, x):
        maxpool = self.maxpool(x)
        attention = self.mlp(maxpool.view(x.shape[0], -1))
        
        return x * attention.unsqueeze(2).unsqueeze(2).unsqueeze(2)
    
# ======================================================================================= #
    
class convblock(nn.Module):
    def __init__(self,
                 in_channel,
                 kernel_size,
                 padding=0,
                 dilate=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channel, 
                      in_channel, 
                      kernel_size=kernel_size,
                      dilation=dilate,
                      padding=padding),
            nn.BatchNorm3d(in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        ) 

    def forward(self, x):
        return self.block(x)
    

class DilatedConv(nn.Module):
    def __init__(self, 
                 in_channel):
        super().__init__()

        self.conv1x1x1 = convblock(in_channel, 
                                   kernel_size=1)
        self.conv3x3x3 = convblock(in_channel,
                                   kernel_size=3,
                                   padding=3,
                                   dilate=3)
        self.conv5x5x5 = convblock(in_channel,
                                   kernel_size=3,
                                   padding=5,
                                   dilate=5)
        self.conv7x7x7 = convblock(in_channel,
                                   kernel_size=3,
                                   padding=7,
                                   dilate=7)
        
    def forward(self, x):
        conv1 = self.conv1x1x1(x)
        conv3 = self.conv3x3x3(x)
        conv5 = self.conv5x5x5(x)
        conv7 = self.conv7x7x7(x)

        return conv1 + conv3 + conv5 + conv7

class Module(nn.Module):
    def __init__(self,
                 in_channel):
        super().__init__()
        self.seg_ca = CA(in_channel)
        self.edge_ca = CA(in_channel)

        self.seg_dilate = DilatedConv(in_channel)
        self.edge_dilate = DilatedConv(in_channel)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seg, edge):
        seg_ca = self.seg_ca(seg)
        seg_edge = self.edge_ca(edge)

        dilate_seg = self.upsample(self.seg_dilate(seg_ca))
        dilate_edge = self.upsample(self.edge_dilate(seg_edge))


        return edge + self.sigmoid(dilate_seg), seg * self.sigmoid(dilate_edge)

# ======================================================================================= #    

class Model(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 width_multiplier = 1):
        super().__init__()

        _channels = (32, 64, 128, 256, 512)

        self.n_channels = _channels
        self.out_channel = out_channel

        self.channels = [int(c*width_multiplier) for c in _channels]

        self.residual1 = Residual(in_channel, self.n_channels[0], f=True)
        self.residual2 = Residual(self.n_channels[0], self.n_channels[1])
        self.residual3 = Residual(self.n_channels[1], self.n_channels[2])
        self.residual4 = Residual(self.n_channels[2], self.n_channels[3])
        
        factor = 2

        self.inc = ConvBlock(in_channel, self.channels[0])
        self.down1 = Down(self.channels[0], self.channels[1])
        self.down2 = Down(self.channels[1], self.channels[2])
        self.down3 = Down(self.channels[2], self.channels[3])
        self.down4= Down(self.channels[3], self.channels[4] // factor)

        self.up1 = Up(self.channels[4], self.channels[3] // factor)
        self.up2 = Up(self.channels[3], self.channels[2] // factor)
        self.up3 = Up(self.channels[2], self.channels[1] // factor)
        self.up4 = Up(self.channels[1], self.channels[0])
        self.outc = OutConv(self.channels[0], self.channels[0] // factor)
        self.final = FinalConv(self.channels[0] // factor, out_channel)

        self.edge_up1 = Up(self.channels[4], self.channels[3] // factor)
        self.edge_up2 = Up(self.channels[3], self.channels[2] // factor)
        self.edge_up3 = Up(self.channels[2], self.channels[1] // factor)
        self.edge_up4 = Up(self.channels[1], self.channels[0])
        self.edge_outc = OutConv(self.channels[0], self.channels[0] // factor)
        self.edge_final = FinalConv(self.channels[0] // factor, out_channel)

        self.module1 = Module(128)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.CA1 = CA(self.n_channels[0])
        self.CA2 = CA(self.n_channels[1])
        self.CA3 = CA(self.n_channels[2])
        self.CA4 = CA(self.n_channels[3])

        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        
        a1 = self.CA1(x1 + x)
        x2 = self.down1(a1)
        
        a2 = self.CA2(x2 + self.residual2(x1))
        x3 = self.down2(a2)
        
        a3 = self.CA3(x3 + self.residual3(x2))
        x4 = self.down3(a3)
        
        a4 = self.CA4(x4 + self.residual4(x3))
        x5 = self.down4(a4)
        
        #print(f"x1 : {x1.shape} x2 : {x2.shape} x3 : {x3.shape} x4 : {x4.shape} x5 : {x5.shape}")
        # decoder
        # seg
        x1_d = self.up1(x5, x4)
        edge_x1_d = self.edge_up1(x5, x4)

        module1_seg, module1_edge = self.module1(x1_d, edge_x1_d)

        # edge
        x2_d = self.up2(module1_seg, x3)
        edge_x2_d = self.edge_up2(module1_edge, x3)

        # module 1
        x3_d = self.up3(x2_d, x2)
        edge_x3_d = self.edge_up3(edge_x2_d, x2)

        x4_d = self.up4(x3_d, x1)
        edge_x4_d = self.edge_up4(edge_x3_d, x1)

        # module 2
        #module2_seg, module2_edge = self.module2(x4_d, edge_x4_d)
        seg_output = self.outc(x4_d)
        edge_output = self.edge_outc(edge_x4_d)

        seg_pred = self.final(seg_output)
        edge_pred = self.edge_final(edge_output)
        
        return seg_pred, edge_pred#,\
                # x1, x2, x3, x4, x5,\
                # x1_d, x2_d, x3_d, x4_d, \
                # edge_x1_d, edge_x2_d, edge_x3_d, edge_x4_d,\
                # a1, a2, a3, a4, module2_edge, module2_seg,\
                # seg_output, edge_output
    
if __name__=='__main__':
    sample = torch.randn(4, 1, 96, 128, 128)
    model = Model(1, 1)

    print(f"Model Parameter : {sum(p.numel() for p in model.parameters())}")

    pred = model(sample)