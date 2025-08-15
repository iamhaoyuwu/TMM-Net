import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
# from timm.models.vision_transformer import _cfg

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print(k.shape)
        # print(v.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=32, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.perprocess1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=1, padding=0, dilation=1, ceil_mode=False)
        )
        self.perprocess2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=0, dilation=1, ceil_mode=False)
        )
        self.perprocess3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1, dilation=1, ceil_mode=False)
        )

        # patch_embed
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])
        # self.patch_embed4 = PatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dims[2],
        #                                embed_dim=embed_dims[3])
        # patch_embed_up
        self.patch_embed_up_3 = PatchEmbed(img_size=14, patch_size=1, in_chans=320,
                                       embed_dim=embed_dims[2])  # 要改
        self.patch_embed_up_2 = PatchEmbed(img_size=14, patch_size=1, in_chans=320,
                                           embed_dim=embed_dims[1])  # 要改
        self.patch_embed_up_1 = PatchEmbed(img_size=28, patch_size=1, in_chans=128,
                                           embed_dim=embed_dims[0])  # 要改

        # pos_embed
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)

        # pos_embed_up
        self.pos_embed_up3 = nn.Parameter(torch.zeros(1, self.patch_embed_up_3.num_patches, embed_dims[2]))
        self.pos_drop_up3 = nn.Dropout(p=drop_rate)
        self.pos_embed_up2 = nn.Parameter(torch.zeros(1, self.patch_embed_up_2.num_patches, embed_dims[1]))
        self.pos_drop_up2 = nn.Dropout(p=drop_rate)
        self.pos_embed_up1 = nn.Parameter(torch.zeros(1, self.patch_embed_up_1.num_patches, embed_dims[0]))
        self.pos_drop_up1 = nn.Dropout(p=drop_rate)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        # transformer encoder upsample
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1_up = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2_up = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3_up = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])

        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed_up3, std=.02)
        trunc_normal_(self.pos_embed_up2, std=.02)
        trunc_normal_(self.pos_embed_up1, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.conv1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(4, 4), bias=False),
                                       nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True),
                                       nn.ReLU(inplace=True),
                                       )
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                       nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True),
                                       nn.ReLU(inplace=True),
                                       )
        self.conv3 = nn.Sequential(nn.Conv2d(128, 320, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                       nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True,track_running_stats=True),
                                       nn.ReLU(inplace=True),
                                       )

        self.fusion1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.fusion2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),                                     )
        self.fusion3 = nn.Sequential(nn.Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True))

        self.upsample_layer3 = nn.Sequential(nn.Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                                     nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True))

        self.upsample_layer2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
                                                            track_running_stats=True),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                       bias=False),
                                             nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
                                                            track_running_stats=True),
                                             nn.ReLU(inplace=True))

        self.upsample_layer1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                            track_running_stats=True),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                       bias=False),
                                             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                                                            track_running_stats=True),
                                             nn.ReLU(inplace=True))

        self.res_up1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True))
        self.res_up2 = nn.Sequential(nn.Conv2d(320, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True))
        self.res_up3 = nn.Sequential(nn.Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     nn.ReLU(inplace=True))

        self.add_ch1 = nn.Sequential(nn.Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                             nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True,
                                                            track_running_stats=True),
                                             nn.ReLU(inplace=True),
                                             )
        self.add_ch2 = nn.Sequential(nn.Conv2d(256, 320, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     )
        self.add_ch3 = nn.Sequential(nn.Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True,
                                                    track_running_stats=True),
                                     nn.ReLU(inplace=True),
                                     )


    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]
            # print(dpr[cur + i])

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x1 = self.perprocess1(x)
        x2 = self.perprocess2(x)
        x1 = F.interpolate(input=x1, size=(300, 300), mode="nearest")
        x2 = F.interpolate(input=x2, size=(300, 300), mode="nearest")
        x3 = x1+x2
        x3 = torch.sigmoid(x3)
        x = self.perprocess3(x)+x3
        x = F.interpolate(input=x, size=(224, 224), mode="nearest")

        B = x.shape[0]

        # stage 1
        x_down = self.conv1(x)
        x, (H, W) = self.patch_embed1(x)
        x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x1 = torch.cat((x1,x_down),dim=1)
        x1 = self.fusion1(x1)
        x1 = F.relu(x1)

        # stage 2
        x2_down = self.conv2(x1)
        x1_mid, (H, W) = self.patch_embed2(x1)
        x1_mid = x1_mid + self.pos_embed2
        x1_mid = self.pos_drop2(x1_mid)
        for blk in self.block2:
            x1_mid = blk(x1_mid, H, W)
        x2 = x1_mid.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2 = torch.cat((x2, x2_down), dim=1)
        x2 = self.fusion2(x2)
        x2 = F.relu(x2)

        # stage 3
        x3_down = self.conv3(x2)
        x2_mid, (H, W) = self.patch_embed3(x2)
        x2_mid = x2_mid + self.pos_embed3
        x2_mid = self.pos_drop3(x2_mid)
        for blk in self.block3:
            x2_mid = blk(x2_mid, H, W)
        x3 = x2_mid.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x3 = torch.cat((x3, x3_down), dim=1)
        x3 = self.fusion3(x3)
        x3 = F.relu(x3)

        # stage 3 upsample
        x3_mid_up, (H, W) = self.patch_embed_up_3(x3)
        x3_temp = self.res_up3(x3)
        x3_mid_up = x3_mid_up + self.pos_embed_up3
        x3_mid_up = self.pos_drop_up3(x3_mid_up)
        for blk in self.block3_up:
            x3_mid_up = blk(x3_mid_up, H, W)
        x3_up = x3_mid_up.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x3_up = self.upsample_layer3(x3_up)+x3_temp

        # stage 2 upsample
        x2_mid_up, (H, W) = self.patch_embed_up_2(x3_up)
        x3_up_temp = F.interpolate(input=x3_up, size=(28, 28), mode="nearest")
        x2 = self.res_up2(x3_up_temp)
        x2_mid_up = x2_mid_up + self.pos_embed_up2
        x2_mid_up = self.pos_drop_up2(x2_mid_up)
        for blk in self.block2_up:
            x2_mid_up = blk(x2_mid_up, H, W)
        x2_up = x2_mid_up.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x2_up = F.interpolate(input=x2_up, size=(28, 28), mode="nearest")
        x2_up = self.upsample_layer2(x2_up)+x2

        # stage 1 upsample
        x1_mid_up, (H, W) = self.patch_embed_up_1(x2_up)
        x2_up_temp = F.interpolate(input=x2_up, size=(56, 56), mode="nearest")
        x1 = self.res_up1(x2_up_temp)
        x1_mid_up = x1_mid_up + self.pos_embed_up1
        x1_mid_up = self.pos_drop_up1(x1_mid_up)
        for blk in self.block1_up:
            x1_mid_up = blk(x1_mid_up, H, W)
        x1_up = x1_mid_up.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x1_up = F.interpolate(input=x1_up, size=(56, 56), mode="nearest")
        x1_up = self.upsample_layer1(x1_up)+x1

        temp1 = torch.cat((x1,x1_up),dim=1)
        temp2 = torch.cat((x2,x2_up),dim=1)
        temp3 = torch.cat((x3,x3_up),dim=1)
        res1 = self.add_ch1(temp1)
        res2 = self.add_ch2(temp2)
        res3 = self.add_ch3(temp3)

        return res1,res2,res3


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 27, 3], sr_ratios=[8, 4, 2, 1],
        # drop_rate=0.0, drop_path_rate=0.05)
        **kwargs)


    return model

if __name__ == '__main__':
    img = torch.ones([1, 3, 224, 224])
    model = pvt_medium()
    out = model(img)
    # print(model)
    # print(out)
    # print(out.shape)

