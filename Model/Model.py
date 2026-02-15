import torch
import torch.nn as nn
import math

n_class = 8 # assign no of classes in the dataset


class Overlap_Patch_Embedding(nn.Module):
    def __init__(self, in_channel, hidden_proj_dim, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, hidden_proj_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(hidden_proj_dim, eps=1e-6)

    def forward(self, f_map):
        x = self.proj(f_map)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        x = self.norm(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, hidden_proj_dim, seq_reduction_ratio, num_attention_heads, dropout_rate=0.0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_proj_dim = hidden_proj_dim
        self.scale = (hidden_proj_dim // num_attention_heads) ** -0.5
        self.sr_ratio = seq_reduction_ratio
        
        self.attention_head_size = int(self.hidden_proj_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.q = nn.Linear(self.hidden_proj_dim, self.all_head_size)
        self.kv = nn.Linear(self.hidden_proj_dim, self.all_head_size*2)
        self.proj = nn.Linear(self.hidden_proj_dim, self.all_head_size)
        
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        
        if seq_reduction_ratio > 1:
            self.sr = nn.Conv2d(self.hidden_proj_dim, self.hidden_proj_dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(self.hidden_proj_dim)
    
    def transpose_for_scores(self, hidden_states):
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(*new_shape)
        return hidden_states.permute(0, 2, 1, 3)
    
    def forward(self, x):
        B, N, C = x.shape
        q = self.transpose_for_scores(self.q(x))
        
        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = x.shape
            # Reshape to (batch_size, num_channels, height, width)
            x = x.permute(0, 2, 1).reshape(batch_size, num_channels, int(math.sqrt(N)), int(math.sqrt(N)))
            # Apply sequence reduction
            x = self.sr(x)
            # Reshape back to (batch_size, seq_len, num_channels)
            x = x.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            x = self.norm(x)
            

        kv = self.kv(x)
        kv = kv.reshape(B, kv.shape[1], 2, self.num_attention_heads, C//self.num_attention_heads).permute(2, 0, 3, 1, 4)

        k = kv[0]
        v = kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DW_Conv(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.activ = nn.GELU()

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        x = self.dwconv(x)
        x = self.activ(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class MLP_block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, out_channel)
        self.fc2 = nn.Linear(out_channel, in_channel)
        self.drop = nn.Dropout(p=dropout_rate)

        self.dwconv = DW_Conv(out_channel)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)

        x = self.dwconv(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    def __init__(self, hidden_proj_dim, seq_reduction_ratio, num_attention_heads, mlp_expan_ratio, dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_proj_dim, eps=1e-6)

        self.attn = SelfAttention(hidden_proj_dim, seq_reduction_ratio, num_attention_heads, dropout_rate=dropout_rate)

        self.norm2 = nn.LayerNorm(hidden_proj_dim, eps=1e-6)

        self.mlp = MLP_block(in_channel=hidden_proj_dim, out_channel=int(hidden_proj_dim*mlp_expan_ratio), dropout_rate=dropout_rate)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class Decoder(nn.Module):
    def __init__(self, n_class, dropout_rate=0.0):
        super().__init__()
        self.block_4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

        self.block_1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        
        self.final_out = nn.Sequential(
            nn.Conv2d(256, n_class, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=4, mode='bilinear'),
        )

    def forward(self, block_1_out, block_2_out, block_3_out, block_4_out):
        block_4 = self.block_4(block_4_out)
        block_3 = self.block_3(block_3_out)
        block_2 = self.block_2(block_2_out)
        block_1 = self.block_1(block_1_out)

        blocks_add = block_4 + block_3 + block_2 + block_1

        output = self.final_out(blocks_add)

        return output

class Model(nn.Module):
    def __init__(self, input_channel, n_class, dropout_rate=0.0):
        super().__init__()
         # patch_embeds = [hidden_proj_dim, kernel_size, stride, padding]
        self.patch_embed_dim = [[64, 7, 4, 3], [128, 3, 2, 1], [320, 3, 2, 1], [512, 3, 2, 1]]

        # stages = [reduction_ratio, attention_heads, mlp_expansion_ration, num_encoder_blocks]
        self.stages_dim = [[8, 1, 4, 3], [4, 2, 4, 6], [2, 5, 4, 40], [1, 8, 4, 3]]

        self.patch_embed1 = Overlap_Patch_Embedding(input_channel, self.patch_embed_dim[0][0], self.patch_embed_dim[0][1], self.patch_embed_dim[0][2], self.patch_embed_dim[0][3])
        self.patch_embed2 = Overlap_Patch_Embedding(self.patch_embed_dim[0][0], self.patch_embed_dim[1][0], self.patch_embed_dim[1][1], self.patch_embed_dim[1][2], self.patch_embed_dim[1][3])
        self.patch_embed3 = Overlap_Patch_Embedding(self.patch_embed_dim[1][0], self.patch_embed_dim[2][0], self.patch_embed_dim[2][1], self.patch_embed_dim[2][2], self.patch_embed_dim[2][3])
        self.patch_embed4 = Overlap_Patch_Embedding(self.patch_embed_dim[2][0], self.patch_embed_dim[3][0], self.patch_embed_dim[3][1], self.patch_embed_dim[3][2], self.patch_embed_dim[3][3])

        self.norm1 = nn.LayerNorm(self.patch_embed_dim[0][0], eps=1e-6)
        self.norm2 = nn.LayerNorm(self.patch_embed_dim[1][0], eps=1e-6)
        self.norm3 = nn.LayerNorm(self.patch_embed_dim[2][0], eps=1e-6)
        self.norm4 = nn.LayerNorm(self.patch_embed_dim[3][0], eps=1e-6)

        # self.block1 = nn.ModuleList(
        #     [
        #         Block(
        #             hidden_size, 
        #             reduction_ratio, 
        #             num_attention_head, 
        #             mlp_expansion_ratio,
        #             dropout_rate=dropout_rate
        #         )
        #         for _ in range(depth)
        #     ]
        # )

        self.block1 = nn.ModuleList(
            [
                Block(
                    self.patch_embed_dim[0][0], 
                    self.stages_dim[0][0], 
                    self.stages_dim[0][1], 
                    self.stages_dim[0][2],
                    dropout_rate=dropout_rate
                )
                for _ in range(self.stages_dim[0][3])
            ]
        )

        self.block2 = nn.ModuleList(
            [
                Block(
                    self.patch_embed_dim[1][0], 
                    self.stages_dim[1][0], 
                    self.stages_dim[1][1], 
                    self.stages_dim[1][2],
                    dropout_rate=dropout_rate
                )
                for _ in range(self.stages_dim[1][3])
            ]
        )

        self.block3 = nn.ModuleList(
            [
                Block(
                    self.patch_embed_dim[2][0], 
                    self.stages_dim[2][0], 
                    self.stages_dim[2][1], 
                    self.stages_dim[2][2],
                    dropout_rate=dropout_rate
                )
                for _ in range(self.stages_dim[2][3])
            ]
        )

        self.block4 = nn.ModuleList(
            [
                Block(
                    self.patch_embed_dim[3][0], 
                    self.stages_dim[3][0], 
                    self.stages_dim[3][1], 
                    self.stages_dim[3][2],
                    dropout_rate=dropout_rate
                )
                for _ in range(self.stages_dim[3][3])
            ]
        )

        self.decoder = Decoder(n_class)

    def forward(self, x):
        x = self.patch_embed1(x)
        for i, block in enumerate(self.block1):
            x = block(x)

        x = self.norm1(x)
        block_1_out = x.transpose(1, 2)
        block_1_out = torch.reshape(block_1_out, (block_1_out.shape[0], block_1_out.shape[1], int(math.sqrt(block_1_out.shape[2])), int(math.sqrt(block_1_out.shape[2]))))
        # print(block_1_out.shape)

        x = self.patch_embed2(block_1_out)
        for i, block in enumerate(self.block2):
            x = block(x)

        x = self.norm2(x)
        block_2_out = x.transpose(1, 2)
        block_2_out = torch.reshape(block_2_out, (block_2_out.shape[0], block_2_out.shape[1], int(math.sqrt(block_2_out.shape[2])), int(math.sqrt(block_2_out.shape[2]))))
        # print(block_2_out.shape)

        x = self.patch_embed3(block_2_out)
        for i, block in enumerate(self.block3):
            x = block(x)

        x = self.norm3(x)
        block_3_out = x.transpose(1, 2)
        block_3_out = torch.reshape(block_3_out, (block_3_out.shape[0], block_3_out.shape[1], int(math.sqrt(block_3_out.shape[2])), int(math.sqrt(block_3_out.shape[2]))))
        # print(block_3_out.shape)

        x = self.patch_embed4(block_3_out)
        for i, block in enumerate(self.block4):
            x = block(x)

        x = self.norm4(x)
        block_4_out = x.transpose(1, 2)
        block_4_out = torch.reshape(block_4_out, (block_4_out.shape[0], block_4_out.shape[1], int(math.sqrt(block_4_out.shape[2])), int(math.sqrt(block_4_out.shape[2]))))
        # print(block_4_out.shape)


        out = self.decoder(block_1_out, block_2_out, block_3_out, block_4_out)


        return out

model = Model(3, n_class, dropout_rate=0.2)