import torch
import torch.nn as nn

from einops import rearrange

# Basic Convolutional Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        features = self.conv(x)
        downsampled = self.pool(features)
        return downsampled, features
    
# Transformer Encoder Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x

# Vision Transformer Encoder
class ViTEncoder(nn.Module):
    def __init__(self, input_dim, patch_size, embed_dim, num_heads, depth, mlp_dim):
        super(ViTEncoder, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1, 1 + embed_dim, embed_dim))
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        x = rearrange(x, 'b c d h w -> b (d h w) c')
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positional_embedding[:, :x.size(1)]

        for block in self.transformer:
            x = block(x)

        return x[:, 1:]  # Remove class token
    
# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)  # Skip connection
        x = self.conv(x)
        return x
    
# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, mlp_dim, cnn_feature_dim):
        super(TransformerDecoder, self).__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, cnn_feature_dim, embed_dim))
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, cnn_features):
        B, C, D, H, W = cnn_features.size()
        print(f"cnn features : {cnn_features.shape}")
        cnn_features = rearrange(cnn_features, 'b c d h w -> b (d h w) c')
        B = x.size(0)
        
        queries = self.query_tokens.expand(B, -1, -1)
        for block in self.transformer:
            attn_out, _ = self.cross_attention(queries, cnn_features, cnn_features)
            queries = queries + attn_out
        return queries
    
class MaskedCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(MaskedCrossAttention, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, queries, features):
        queries_norm = self.norm1(queries)
        features_norm = self.norm1(features)
        
        attn_out, _ = self.cross_attn(queries_norm, features_norm, features_norm)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm2(queries))
        
        return queries
    
class TransformerDecoder(nn.Module):
    def __init__(self,
                 embed_dim, 
                 num_heads,
                 depth,
                 mlp_dim):
        super(TransformerDecoder, self).__init__()
        self.layers = MaskedCrossAttention(embed_dim, num_heads, mlp_dim)
    
        
    def forward(self, queries, features):
        queries = self.layers(queries, features)
        return queries
        
class TransUNet3D(nn.Module):
    def __init__(self,
                 in_channels,
                 base_features,
                 num_classes,
                 embed_dim = 768,
                 patch_size = (4, 4, 4),
                 num_heads = 12,
                 depth = 4,
                 mlp_dim = 1024,
                 num_queries = 100):
        super(TransUNet3D, self).__init__()
        
        # CNN Encoder
        self.encoder1 = EncoderBlock(in_channels, base_features)
        self.encoder2 = EncoderBlock(base_features, base_features * 2)
        self.encoder3 = EncoderBlock(base_features * 2, base_features * 4)
        
        self.vit_encoder = ViTEncoder(base_features * 4, patch_size, embed_dim, num_heads, depth, mlp_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        self.transformer_decoder = nn.ModuleList([
            TransformerDecoder(embed_dim, num_heads, depth, mlp_dim) for _ in range(3)
        ])
        
        self.decoder3 = DecoderBlock(base_features * 8, base_features * 4)      
        self.decoder2 = DecoderBlock(base_features * 4, base_features * 2)
        self.decoder1 = DecoderBlock(base_features * 2, base_features)
        
        self.final_conv = nn.Conv3d(embed_dim, num_classes, kernel_size=1)
        
    def forward(self, x):
        x1, f1 = self.encoder1(x)
        x2, f2 = self.encoder2(x1)
        x3, f3 = self.encoder3(x2)
        
        vit_features = self.vit_encoder(x3)
        
        B = x.size(0)
        queries = self.query_tokens.expand(B, -1, -1)
        
        print(f"x1 : {x1.shape}, x2 : {x2.shape} x3 : {x3.shape} vit_features : {vit_features.shape}")
        print(f"f1 : {f1.shape} f2 : {f2.shape} f3 : {f3.shape} queries : {queries.shape}")
        d3 = self.decoder3(vit_features, f3)
        queries = self.transformer_decoder[0](queries, d3)
        
        d2 = self.decoder3(d3, f2)
        queries = self.transformer_decoder[1](queries, d2)
        
        d1 = self.decoder3(d2, f1)
        queries = self.transformer_decoder[2](queries, d1)
        
        outputs = self.final_conv(queries)
        
        return outputs
    
if __name__=='__main__':
    sample = torch.randn(4, 1, 96, 128, 128)
    model = TransUNet3D(1, 32, 1)
    pred = model(sample)