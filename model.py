import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.crossvit import CrossAttention
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from torch.nn import MultiheadAttention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
def film(x, shift, scale):
    # x: (N, C, L); shift/scale: (N, C)
    return x * (1 + scale[:, :, None]) + shift[:, :, None]

class CondMLP(nn.Module):
    """把条件向量 c -> (shift, scale)"""
    def __init__(self, c_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_dim, 2 * out_channels)
        )
    def forward(self, c):
        sh, sc = self.net(c).chunk(2, dim=1)
        return sh, sc

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, c_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.emb = CondMLP(c_dim, out_ch)     # 生成 (shift, scale)
        self.act = nn.SiLU()
        self.skip = (in_ch != out_ch)
        if self.skip:
            self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x, c):
        # 第一层
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # FiLM 调制
        shift, scale = self.emb(c)
        h = film(h, shift, scale)
        # 第二层
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        # 残差
        if self.skip:
            x = self.proj(x)
        return x + h

class Down1D(nn.Module):
    def __init__(self, ch, c_dim):
        super().__init__()
        self.res1 = ResBlock1D(ch, ch, c_dim)
        self.res2 = ResBlock1D(ch, ch, c_dim)
        self.down = nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1)  # /2
    def forward(self, x, c):
        x = self.res1(x, c); x = self.res2(x, c)
        skip = x
        x = self.down(x)
        return x, skip

class Up1D(nn.Module):
    def __init__(self, ch, c_dim):
        super().__init__()
        self.res1 = ResBlock1D(ch*2, ch, c_dim)
        self.res2 = ResBlock1D(ch, ch, c_dim)
        self.up = nn.ConvTranspose1d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip, c):
        # 先上采样，再对齐长度
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            x = F.interpolate(x, size=skip.size(-1), mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, c)
        x = self.res2(x, c)
        return x
class UNet1D(nn.Module):
    """
    输入: (N, C_in, L)；输出: (N, C_out, L)
    其中 C_in = hidden_size, L = NumGene
    """
    def __init__(self, C_in, C_base, C_out, c_dim, num_levels=3):
        super().__init__()
        self.stem = nn.Conv1d(C_in, C_base, kernel_size=3, padding=1)

        downs, ups = [], []
        ch = C_base
        self.down_channels = [ch]
        for _ in range(num_levels):
            downs.append(Down1D(ch, c_dim))
            self.down_channels.append(ch)
        self.downs = nn.ModuleList(downs)

        # bottleneck
        self.mid1 = ResBlock1D(ch, ch, c_dim)
        self.mid2 = ResBlock1D(ch, ch, c_dim)

        # ups
        for _ in range(num_levels):
            ups.append(Up1D(ch, c_dim))
        self.ups = nn.ModuleList(ups)

        self.head = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv1d(ch, C_out, kernel_size=3, padding=1)
        )

    def forward(self, x, c):
        # x: (N, C_in, L)
        x = self.stem(x)
        skips = []
        for down in self.downs:
            x, s = down(x, c)
            skips.append(s)

        x = self.mid1(x, c)
        x = self.mid2(x, c)

        for up in self.ups[::-1]:
            s = skips.pop()
            x = up(x, s, c)

        out = self.head(x)
        return out  # (N, C_out, L)
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class inputEmbedder(nn.Module):
    def __init__(self, input_size, hidden_dim):
        """
        input_size: number of genes in input
        hidden_dim: num hidden dimension
        """
        super().__init__()
        self.ebd_layer_1 = nn.Parameter(torch.empty((input_size, hidden_dim)),
                                          requires_grad=True)  # trainable look-up table to encode gene name
        torch.nn.init.kaiming_uniform_(self.ebd_layer_1, a=math.sqrt(5))
        self.ebd_layer_2 = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        torch.nn.init.xavier_uniform_(self.ebd_layer_2[0].weight)
        torch.nn.init.xavier_uniform_(self.ebd_layer_2[2].weight)

        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        x: (N, NumGene) tensor of inputs

        """
        x = self.ebd_layer_2(x.squeeze(1).unsqueeze(2))  # (N, NumGene, hidden_dim)
        ebd = self.ebd_layer_1
        x = torch.add(ebd, x)
        return x  # (N, NumGene, hidden_dim)


class crossAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()



#################################################################################
#                                 Core Model                                    #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim,
                       act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, 2, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = torch.permute(self.linear(x), (0, 2, 1))  # (N, NumGene, hidden_size) -> (N, 2, NumGene)
        return x


y_all = []


class HisDiff(nn.Module):
    def __init__(self,
                 input_size=310,
                 hidden_size=1152,
                 depth=28,
                 num_heads=16,
                 mlp_ratio=4.0,
                 label_size=512,
                 learn_sigma=True,
                 ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.label_size = label_size
        # input MLP
        self.inputEmbedder = inputEmbedder(self.input_size, self.hidden_size)
        # time step embedding
        self.time_embed = TimestepEmbedder(self.hidden_size)
        # label embedding (input label is already in embedding form, here just reorganize the size using linear layer)
        self.crossAttention_ln = MultiheadAttention(1024,num_heads=8,batch_first=True)
        self.crossAttention_gn = MultiheadAttention(1024,num_heads=8,batch_first=True)
        self.img_ebd = nn.Sequential(
            nn.Linear(label_size, label_size, bias=True),
            nn.ReLU(),
            nn.Linear(label_size, hidden_size, bias=True),
        )
        # no positional embedding
        #self.unet = UNet1D(C_in=hidden_size, C_base=hidden_size // 2, C_out=2, c_dim=hidden_size, num_levels=3)
        #self.initialize_weights_unet()
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(self.hidden_size)
        self.initialize_weights()
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.img_ebd[2].weight, std=0.02)
        nn.init.normal_(self.img_ebd[2].weight, std=0.02)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    def forward(self, x, t, **kwargs):
        """
        Forward pass of DiT.
        x: (N, NumGene) tensor of inputs
        t: (N,) tensor of diffusion timesteps
        y: (N, 512) tensor of conditions
        """
        local_ebd = kwargs['local_ebd']
        global_ebd = kwargs['global_ebd']
        neighbor_ebd = kwargs['neighbor_ebd']
        local_ebd = local_ebd.unsqueeze(1)
        local_ebd = local_ebd
        y1,_ = self.crossAttention_ln(local_ebd, neighbor_ebd,neighbor_ebd)
        y1 = y1.squeeze(1) #(b,1024)
        y2,_ = self.crossAttention_gn(local_ebd, global_ebd,global_ebd)
        y2 = y2.squeeze(1)#(b,1024)
        y = y2+y1
        y = self.img_ebd(y)#(b,384)
        x = self.inputEmbedder(x)  # (N, NumGene, hidden_dim)
        t = self.time_embed(t)  # (N, hidden_dim) [time_ebd]
        c = t + y#(b,384)
        for block in self.blocks:
            x = block(x, c)  # (N, NumGene, hidden_dim)
        x = self.final_layer(x, c)  # (N, 2, NumGene)
        return x


class Unet(nn.Module):
    def __init__(self,
                 input_size=200,
                 hidden_size=1152,
                 depth=28,
                 num_heads=16,
                 mlp_ratio=4.0,
                 label_size=512,
                 learn_sigma=True,
                 ):
        super().__init__()

        self.learn_sigma = learn_sigma
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.label_size = label_size
        # input MLP
        self.inputEmbedder = inputEmbedder(self.input_size, self.hidden_size)
        # time step embedding
        self.time_embed = TimestepEmbedder(self.hidden_size)
        # label embedding (input label is already in embedding form, here just reorganize the size using linear layer)

        self.crossAttention_ln = MultiheadAttention(1024,num_heads=8,batch_first=True)
        self.crossAttention_gn = MultiheadAttention(1024,num_heads=8,batch_first=True)

        self.img_ebd = nn.Sequential(
            nn.Linear(label_size, label_size, bias=True),
            nn.ReLU(),
            nn.Linear(label_size, hidden_size, bias=True),
        )
        # no positional embedding
        self.unet = UNet1D(C_in=hidden_size, C_base=hidden_size // 2, C_out=2, c_dim=hidden_size, num_levels=3)
        self.initialize_weights_unet()
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        self.final_layer = FinalLayer(self.hidden_size)
        #self.initialize_weights()
    # def initialize_weights(self):
    #     # Initialize transformer layers:
    #     def _basic_init(module):
    #         if isinstance(module, nn.Linear):
    #             torch.nn.init.xavier_uniform_(module.weight)
    #             if module.bias is not None:
    #                 nn.init.constant_(module.bias, 0)
    #
    #     self.apply(_basic_init)
    def initialize_weights_unet(self):
        def _init(module):
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.ConvTranspose1d) or isinstance(module,
                                                                                                     nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if getattr(module, 'bias', None) is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.img_ebd[0].weight, std=0.02)
        nn.init.normal_(self.img_ebd[2].weight, std=0.02)
        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        #
        # # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, **kwargs):
        """
        Forward pass of DiT.
        x: (N, NumGene) tensor of inputs
        t: (N,) tensor of diffusion timesteps
        y: (N, 512) tensor of conditions
        """
        # global_ebd = kwargs['global_ebd']
        local_ebd = kwargs['local_ebd']
        global_ebd = kwargs['global_ebd']
        neighbor_ebd = kwargs['neighbor_ebd']
        local_ebd = local_ebd.unsqueeze(1)
        #local_ebd = local_ebd
        y1,_ = self.crossAttention_ln(local_ebd, neighbor_ebd,neighbor_ebd)
        y1 = y1.squeeze(1) #(b,1024)
        y2,_ = self.crossAttention_gn(local_ebd, global_ebd,global_ebd)
        y2 = y2.squeeze(1)#(b,1024)
        y = y2+y1
        y = self.img_ebd(y)#(b,384)
        x = self.inputEmbedder(x)  # (N, NumGene, hidden_dim)
        t = self.time_embed(t)  # (N, hidden_dim) [time_ebd]
        c = t + y#(b,384)
        x = x.permute(0, 2, 1)
        # for block in self.blocks:
        #     x = block(x, c)  # (N, NumGene, hidden_dim)
        # x = self.final_layer(x, c)  # (N, 2, NumGene)
        out = self.unet(x, c)
        return out
