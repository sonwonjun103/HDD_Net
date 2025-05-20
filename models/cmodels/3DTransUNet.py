import torch
import copy
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn.modules.utils import _triple

class StdConv3d(nn.Conv3d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        if self.padding[0]==0:
            xout=F.conv3d(x, w, bias=self.bias, stride=self.stride, padding=[0],
                        dilation=self.dilation, groups=self.groups)
        elif self.padding[0]==1:
            xout=F.conv3d(x, w, bias=self.bias, stride=self.stride, padding=[1],
                        dilation=self.dilation, groups=self.groups)
        elif self.padding[0]==3:
            xout=F.conv3d(x, w, bias=self.bias, stride=self.stride, padding=[3],
                        dilation=self.dilation, groups=self.groups)
        else:
            xout=F.conv3d(x, w, bias=self.bias, stride=self.stride, padding=[0],
                        dilation=self.dilation, groups=self.groups)
        return xout


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=3, stride=stride,
                     padding=[1], bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv3d(cin, cout, kernel_size=1, stride=stride,
                     padding=[0], bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)   

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class ResNetV23d(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv3d(3, width, kernel_size=7, stride=2, bias=False, padding=[3])),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width * 4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 4, cout=width * 4, cmid=width)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 4, cout=width * 8, cmid=width * 2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 8, cout=width * 8, cmid=width * 2)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width * 8, cout=width * 16, cmid=width * 4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width * 16, cout=width * 16, cmid=width * 4)) for i in
                 range(2, block_units[2] + 1)],
            ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _, _ = x.size()
        
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i + 1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3], 0:x.size()[4]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)

class EncoderBlock3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        # convolve 1
        self.conv1 = Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # convolve 2
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # downsample
        self.down = F.interpolate

    def forward(self, x, skip=None):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EncoderCup3d(nn.Module):  # cascade downsampler
    def __init__(self, config):
        super().__init__()
        self.config = config

        if len(config.decoder_channels) == 4:
            out_channels = [config.decoder_channels[-2], config.decoder_channels[-3], config.decoder_channels[-4]]
            in_channels = [3, config.decoder_channels[-2], config.decoder_channels[-3]]
        elif len(config.decoder_channels) == 3:
            out_channels = [config.decoder_channels[-2], config.decoder_channels[-3]]
            in_channels = [3, config.decoder_channels[-2]]

        blocks = [
            EncoderBlock3d(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, config.skip_channels)
        ]

        self.blocks = nn.ModuleList(blocks)
        self.down = F.interpolate

    def forward(self, x):
        features_tmp = []
        ctr = 4
        for i, encoder_block in enumerate(self.blocks):
            x = encoder_block(x)
            features_tmp.append(self.down(input=x, scale_factor=1 / ctr, mode='trilinear', align_corners=True))
            ctr *= 2
        features = []
        for i in range(len(features_tmp)):
            features.append(features_tmp[-(i + 1)])

        return x, features

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer_num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer_attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer_attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer_mlp_dim)
        self.fc2 = nn.Linear(config.transformer_mlp_dim, config.hidden_size)
        self.act_fn = nn.functional.gelu
        self.dropout = nn.Dropout(config.transformer_dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class DecoderBlock3d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        # convolve 1
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # convolve 2
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # upsample
        self.up = F.interpolate

    def forward(self, x, skip=None):
        x = self.up(input=x, scale_factor=2, mode='trilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x.float())
        x = self.conv2(x)
        return x
    
class Embeddings3d(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings3d, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _triple(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (
            img_size[0] // config.patches.size // grid_size[0], img_size[1] // config.patches.size // grid_size[1],
            img_size[2] // config.patches.size // grid_size[2])
            patch_size_real = (patch_size[0] * config.patches.size, patch_size[1] * config.patches.size,
                               patch_size[2] * config.patches.size)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) * (
                        img_size[2] // patch_size_real[2])
            self.hybrid = True
        else:
            patch_size = _triple(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV23d(block_units=config.resnet.num_layers,
                                           width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        else:
            self.convolutions = EncoderCup3d(config)
            if len(config.decoder_channels) == 4:
                in_channels = config.decoder_channels[-4]
            elif len(config.decoder_channels) == 3:
                in_channels = config.decoder_channels[-3]

        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        # self.position_embeddings = nn.Parameter(torch.reshape(torch.arange(0,n_patches * config.hidden_size), (1, n_patches, config.hidden_size)).float())

        self.dropout = nn.Dropout(config.transformer_dropout_rate)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            x, features = self.convolutions(x)

        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # add position to the patches
        z = self.position_embeddings
        embeddings = x + z
        embeddings = self.dropout(embeddings)
        return embeddings, features
    
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # make the layers within the transformer
        for _ in range(config.transformer_num_layers):
            # block contains multi-head attention and mlp
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
    
class Transformer3d(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer3d, self).__init__()
        self.embeddings = Embeddings3d(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features
    
class DecoderCup3d(nn.Module):  # cascade upsampler
    def __init__(self, config):
        super().__init__()
        self.config = config
        if len(config.decoder_channels) == 4:
            head_channels = 512
        elif len(config.decoder_channels) == 3:
            head_channels = 256

        # define a model that takes the hidden input size and applies a convolution with output size = head_channels
        # (D, H/16, W/16)
        self.conv_more = Conv3dReLU(
            config.hidden_size,  # these are the inchannels
            head_channels,  # these are the outchannels
            kernel_size=3,
            padding=1,
            stride=1,
            use_batchnorm=True,
        )

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            if len(config.skip_channels) == 4:
                for i in range(4 - self.config.n_skip):  # re-select the skip channels according to n_skip
                    skip_channels[3 - i] = 0
            elif len(config.skip_channels) == 3:
                for i in range(3 - self.config.n_skip):  # re-select the skip channels according to n_skip
                    skip_channels[2 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]

        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])

        out_channels = decoder_channels

        blocks = [
            DecoderBlock3d(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
            zip(in_channels, out_channels, skip_channels)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):

        B, n_patch, hidden = hidden_states.size()
        # reshape from (B, n_patch, hidden) to (B, h, w, l, hidden)

        h, w, l = int(np.cbrt(n_patch)), int(np.cbrt(n_patch)), int(np.cbrt(n_patch))
        # h,w,l are the number of rows/columns/... in the grid of patches. 
        # i.e. for a 224 input with 16x16 patches there are 14x14
        x = hidden_states.permute(0, 2, 1)
        # rearranges the original tensor according to the desired ordering and returns a new multidimensional rotated
        # tensor
        x = x.contiguous().view(B, hidden, h, w, l)  # [24, 768, 14, 14, 14]
        # .contiguous Returns a contiguous in memory tensor containing the same data as self tensor
        # reshapes
        x = self.conv_more(x)

        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None

            x = decoder_block(x, skip=skip)

        return x
        
class SegmentationHead3d(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear3d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv3d, upsampling)

class VisionTransformer3d(nn.Module):
    def __init__(self, config, img_size=224, zero_head=False, vis=False):
        super(VisionTransformer3d, self).__init__()  # defining this as a model in nn
        self.classifier = config
        self.transformer = Transformer3d(config, img_size, vis)
        self.decoder = DecoderCup3d(config)
        self.segmentation_head = SegmentationHead3d(
            in_channels=config['decoder_channels'][-1],  # final upsampling conv output size
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)  # turning the image into 3 channels

        # first run transformer
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        # then decoder
        x = self.decoder(x, features)
        # then segmentation_head
        logits = self.segmentation_head(x)

        return logits
    
if __name__=='__main__':
    model = VisionTransformer3d('seg',
                                256)
    sample = torch.randn(2, 1, 256, 256, 256)
    pred = model(sample)