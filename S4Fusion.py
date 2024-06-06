from torch import nn
from modules.utils import *
from modules.layers import *
import math
from timm.models.layers import trunc_normal_
from modules.fusion import FusionBlock
import matplotlib.pyplot as plt
from modules.fold import Fold

class VSSM(nn.Module): # Core
    def __init__(self, patch_size=4, in_chans=3, out_channel=1, depths=[2, 4, 2], depths_decoder=[2, 4, 2],
                 dims=[48, 96, 192], dims_decoder=[192, 96, 48], d_state=16, d_state_fusion=24, drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, patch_norm=True, 
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.out_channel = out_channel
        self.num_layers = len(depths)

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed_x = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        self.patch_embed_y = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers_x = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_x.append(layer)

        self.layers_y = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_y.append(layer)
        
        self.fusion_layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = FusionLayer(
                d_model=dims[i_layer],
                drop_rate=drop_rate,
                d_state=d_state_fusion,
                num_fusion_layer=3
            )
            self.fusion_layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state, # 20240109
                drop_rate=drop_rate, 
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, out_channel, 1)

        self.fold = Fold(kernel_size=4, stride=3)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, y):
        skip_list = []
        x = self.patch_embed_x(x) 
        x = self.pos_drop(x)
        y = self.patch_embed_y(y)
        y = self.pos_drop(y)

        for layer_x, layer_y in zip(self.layers_x, self.layers_y):
            skip_list.append((x, y))
            x, y = layer_x(x), layer_y(y)
        return x, y, skip_list
    
    def forward_features_up(self, skip_list, x):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])
        return x
    
    def forward_final(self, z, H, W):
        z = self.final_up(z)
        z = z.permute(0,3,1,2) # B C H W
        B, C, h, w = z.shape
        z = z.reshape(B, C, -1)
        z = self.fold(z, (H, W))
        z = self.final_conv(z)
        return z
    
    def forward_fusion(self, skip_list):
        for i, ((x, y), fusion) in enumerate(zip(skip_list, self.fusion_layers)):
            skip_list[i] = fusion(x, y)
        return skip_list

    def forward(self, x, y):
        B, C, H, W = x.shape
        x, y, skip_list = self.forward_features(x, y)
        skip_list = self.forward_fusion(skip_list)
        x = self.forward_features_up(skip_list, x + y)
        x = self.forward_final(x, H, W)
        return x

class MambaNet(nn.Module):
    def __init__(self, 
                 input_channels=1, 
                 out_channel=1,
                 depths=[1, 2, 1], 
                 depths_decoder=[1, 2, 1],
                 drop_path_rate=0.,
                 drop_rate=0.,
                 d_state=12, # original: 12
                 d_state_fusion=18
                ):
        super().__init__()

        self.num_classes = out_channel

        self.vmunet = VSSM(in_chans=input_channels,
                           out_channel=out_channel,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           drop_rate=drop_rate,
                           d_state=d_state,
                           d_state_fusion=d_state_fusion
                        )
    
    def forward(self, x, y):
        logits = self.vmunet(x, y)
        return logits