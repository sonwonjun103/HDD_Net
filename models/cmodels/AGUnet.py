import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self,
                 in_channel):
        super(AttentionGate, self).__init__()
        out_channel = in_channel
        self.ReLU = nn.ReLU(inplace=True)
        self.Sigmoid = nn.Sigmoid()
        
        self.conv1x1x1_1 = Conv1x1x1(in_channel, out_channel)
        self.conv1x1x1_2 = Conv1x1x1(in_channel, out_channel)
        self.conv1x1x1_3 = Conv1x1x1(in_channel, out_channel)
        
    def forward(self, x):
        o = self.conv1x1x1_3(x)
        
        x = self.conv1x1x1_1(x)
        x = self.ReLU(x)
        x = self.conv1x1x1_2(x)
        x = self.Sigmoid(x)
        #print(f"o : {o.shape} x : {x.shape}")
        output = torch.add(o, x)
        
        return output

class ConvInit(nn.Module):
    def __init__(self, in_channels):
        super(ConvInit, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        bn1 = torch.nn.BatchNorm3d(self.num_features)
        relu1 = nn.ReLU()

        self.norm = nn.Sequential(bn1, relu1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.norm(y1)

        return y1, y2

class ConvRed(nn.Module):
    def __init__(self, in_channels):
        super(ConvRed, self).__init__()
        self.num_features = 16
        self.in_channels = in_channels

        bn1 = nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=1)
        self.conv_red = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_red(x)
    
class DilatedConv2(nn.Module):
    def __init__(self, in_channels):
        super(DilatedConv2, self).__init__()
        self.num_features = 32
        self.in_channels = in_channels
        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=2, dilation=2)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)

class DilatedConv4(nn.Module):
    def __init__(self, in_channels):
        super(DilatedConv4, self).__init__()
        self.num_features = 52
        self.in_channels = in_channels

        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=3, padding=4, dilation=4)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)
    
class Conv1x1x1(nn.Module):
    def __init__(self, in_channels, classes):
        super(Conv1x1x1, self).__init__()
        self.num_features = classes
        self.in_channels = in_channels

        bn1 = torch.nn.BatchNorm3d(self.in_channels)
        relu1 = nn.ReLU()
        conv1 = nn.Conv3d(self.in_channels, self.num_features, kernel_size=1)

        self.conv_dil = nn.Sequential(bn1, relu1, conv1)

    def forward(self, x):
        return self.conv_dil(x)
    
class AGUnet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut_type='A',
                 dropout_layer = True):
        super(AGUnet, self).__init__()
        self.shortcut_type = shortcut_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.init_channels = 16
        self.red_channels= 16
        self.dil2_channels = 32
        self.dil4_channels = 52
        self.conv_out_channels = 64

        self.conv_init = ConvInit(self.in_channels)

        self.red_blocks1 = self.create_red(self.init_channels)
        self.red_blocks2 = self.create_red(self.red_channels)
        self.red_blocks3 = self.create_red(self.red_channels)
        self.red_attention1 = AttentionGate(self.red_channels)
        self.red_attention2 = AttentionGate(self.red_channels)
        self.red_attention3 = AttentionGate(self.red_channels)

        self.dil2block1 = self.create_dil2(self.red_channels)
        self.dil2block2 = self.create_dil2(self.dil2_channels)
        self.dil2block3 = self.create_dil2(self.dil2_channels)
        self.dil2attention1 = AttentionGate(self.dil2_channels)
        self.dil2attention2 = AttentionGate(self.dil2_channels)
        self.dil2attention3 = AttentionGate(self.dil2_channels)

        self.dil4block1 = self.create_dil4(self.dil2_channels)
        self.dil4block2 = self.create_dil4(self.dil4_channels)
        self.dil4block3 = self.create_dil4(self.dil4_channels)
        self.dil4attention1 = AttentionGate(self.dil4_channels)
        self.dil4attention2 = AttentionGate(self.dil4_channels)
        self.dil4attention3 = AttentionGate(self.dil4_channels)

        if dropout_layer:
            conv_out = nn.Conv3d(self.dil4_channels, self.conv_out_channels, kernel_size=1)
            drop3d = nn.Dropout3d()
            conv1x1x1 = Conv1x1x1(self.conv_out_channels, self.out_channels)
            self.conv_out = nn.Sequential(conv_out, drop3d, conv1x1x1)
        else:
            self.conv_out = Conv1x1x1(self.dil4_channels, self.out_channels)

    def shortcut_pad(self, x, desired_channels):
        if self.shortcut_type == 'A':
            batch_size, channels, dim0, dim1, dim2 = x.shape
            extra_channels = desired_channels - channels
            zero_channels = int(extra_channels / 2)
            zeros_half = x.new_zeros(batch_size, zero_channels, dim0, dim1, dim2)
            y = torch.cat((zeros_half, x, zeros_half), dim=1)
        elif self.shortcut_type == 'B':
            if desired_channels == self.dil2_channels:
                y = self.res_pad_1(x)
            elif desired_channels == self.dil4_channels:
                y = self.res_pad_2(x)
        return y     

    def create_red(self, in_channel):
        conv_red_1 = ConvRed(in_channel)
        conv_red_2 = ConvRed(self.red_channels)

        return nn.Sequential(conv_red_1, conv_red_2)
    
    def create_dil2(self, in_channels):
        conv_dil2_1 = DilatedConv2(in_channels)
        conv_dil2_2 = DilatedConv2(self.dil2_channels)
        return nn.Sequential(conv_dil2_1, conv_dil2_2)

    def create_dil4(self, in_channels):
        conv_dil4_1 = DilatedConv4(in_channels)
        conv_dil4_2 = DilatedConv4(self.dil4_channels)
        return nn.Sequential(conv_dil4_1, conv_dil4_2)
    
    def red_forward(self, x):
        x, x_res = self.conv_init(x)
        x_red_1 = self.red_blocks1(x)
        x_red_attention1 = self.red_attention1(x_red_1)
        x_red_2 = self.red_blocks2(x_red_attention1 + x_res)
        x_red_attention2 = self.red_attention2(x_red_2)
        x_red_3 = self.red_blocks3(x_red_attention2 + x_red_1)
        x_red_attnetion3 = self.red_attention3(x_red_3)
        return x_red_attention2, x_red_attnetion3

    def dilation2(self, x_red_3, x_red_2):
        x_dil2_1 = self.dil2block1(x_red_3 + x_red_2)
        #print(x_dil2_1.shape ,x_red_3.shape )

        x_red_padded = self.shortcut_pad(x_red_3, self.dil2_channels)
        x_dil_attention1 = self.dil2attention1(x_red_padded)
        
        x_dil2_2 = self.dil2block2(x_dil2_1 + x_dil_attention1)
        x_dil_attention2 = self.dil2attention2(x_dil2_2)
        
        x_dil2_3 = self.dil2block3(x_dil_attention2 + x_dil2_1)
        x_dil_attention3 = self.dil2attention3(x_dil2_3)
        return x_dil_attention2, x_dil_attention3

    def dilation4(self, x_dil2_3, x_dil2_2):
        x_dil4_1 = self.dil4block1(x_dil2_3 + x_dil2_2)
        x_dil2_padded = self.shortcut_pad(x_dil2_3, self.dil4_channels)
        x_dil4_attention1 = self.dil4attention1(x_dil2_padded)
        
        x_dil4_2 = self.dil4block2(x_dil4_1 + x_dil4_attention1)
        x_dil4_attention2 = self.dil4attention2(x_dil4_2)
        
        x_dil4_3 = self.dil4block3(x_dil4_attention2 + x_dil4_1)
        x_dil4_attention3 = self.dil4attention3(x_dil4_3)
        
        return x_dil4_attention2 + x_dil4_attention3

    def forward(self, x):
        x_red_3, x_red_2 = self.red_forward(x)
        x_dil2_3, x_dil2_2 = self.dilation2(x_red_3, x_red_2)
        x_dil4 = self.dilation4(x_dil2_3, x_dil2_2)
        y = self.conv_out(x_dil4)
        return y
    
if __name__=='__main__':
    sample = torch.randn(4, 1, 96, 128, 128)
    model = AGUnet(in_channels=1,
                   out_channels=1)
    #model = torch.nn.DataParallel(model).cuda()
    pred = model(sample)
    print(pred.shape)