from layers import *
import numbers
from einops import rearrange
from DeformableNet import *

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, BasicConv=BasicConv):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = BasicConv(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, bias=bias, relu=False, groups=hidden_features * 2)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, scale, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.scale = scale
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Q 的生成：三种空洞卷积后相加
        self.q_conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=bias)  # 空洞率 1
        self.q_conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias)  # 空洞率 2
        self.q_conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3, bias=bias)  # 空洞率 3

        # K 和 V 的生成：空洞率 2
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias)

        # 输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

    def forward(self, x):
        b, c, h, w = x.shape

        # 生成 Q：三种空洞卷积特征相加
        q1 = self.q_conv1(x)
        q2 = self.q_conv2(x)
        q3 = self.q_conv3(x)
        q = q1 + q2 + q3

        # 生成 K 和 V：均为空洞率 2
        k = self.k_conv(x)
        v = self.v_conv(x)

        # 调整形状以适配多头注意力
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化 Q 和 K
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力矩阵并激活
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.relu(attn)  # 使用 ReLU 激活函数

        # 计算注意力输出
        # out = (attn1 @ v)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 输出投影
        out = self.project_out(out)
        return out


class Attention_DCNv4(nn.Module):
    def __init__(self, scale, dim, num_heads, bias):
        super(Attention_DCNv4, self).__init__()
        self.num_heads = num_heads

        self.scale = scale
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # Q
        self.q_conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, bias=bias)
        self.q_conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias)
        self.q_conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3, dilation=3, bias=bias)

        # K 和 V 的生成：空洞率 2
        self.k_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=bias)

        # 偏移量生成：额外添加 offset_conv
        self.offset_conv = nn.Conv2d(dim, 2 * num_heads, kernel_size=3, padding=1)

        # 输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

    def forward(self, x):
        b, c, h, w = x.shape
        
        x = self.qkv(x)  # 输入映射

        # 生成 Q：三种空洞卷积特征相加
        q1 = self.q_conv1(x)
        q2 = self.q_conv2(x)
        q3 = self.q_conv3(x)
        q = q1 + q2 + q3

        # 生成 K 和 V：均为空洞率 2
        k = self.k_conv(x)
        v = self.v_conv(x)

        # 偏移量生成
        offsets = self.offset_conv(x)
        offset_x, offset_y = torch.chunk(offsets, 2, dim=1)  # 分割为 x 和 y 偏移
        offset_x = offset_x.unsqueeze(2)  # (b, num_heads, 1, h, w)
        offset_y = offset_y.unsqueeze(2)

        # 生成采样网格
        base_grid = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device))
        base_grid = torch.stack(base_grid, dim=-1).float()  # (h, w, 2)
        base_grid = base_grid.unsqueeze(0).unsqueeze(1).unsqueeze(2)  # (1, 1, 1, h, w, 2)
        base_grid = base_grid.repeat(b, self.num_heads, 1, 1, 1, 1)

        # 偏移量归一化到特征图范围
        offset_normalizer = torch.tensor([w, h], device=x.device).view(1, 1, 1, 1, 2)  # (1, 1, 1, 1, 2)
        sampling_grid = base_grid + torch.stack((offset_x, offset_y), dim=-1) / offset_normalizer

        # grid_sample 要求输入范围为 [-1, 1]
        sampling_grid = 2.0 * sampling_grid / torch.tensor([w - 1, h - 1], device=x.device).view(1, 1, 1, 1, 2) - 1.0
        sampling_grid = sampling_grid.view(-1, h, w, 2)  # (b*num_heads, h, w, 2)

        # 采样 K 和 V
        k = F.grid_sample(k.view(b * self.num_heads, -1, h, w), sampling_grid, mode='bilinear', align_corners=True)
        v = F.grid_sample(v.view(b * self.num_heads, -1, h, w), sampling_grid, mode='bilinear', align_corners=True)

        # 恢复形状
        k = k.view(b, self.num_heads, -1, h * w)
        v = v.view(b, self.num_heads, -1, h * w)

        # 调整 Q 的形状
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化 Q 和 K
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力矩阵并使用 softmax 激活
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)  # 使用 softmax 激活

        # 计算注意力输出
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 输出投影
        out = self.project_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, scale,dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, BasicConv=BasicConv):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(scale,dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, BasicConv=BasicConv)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class TransformerBlock_DCNv4(nn.Module):
    def __init__(self, scale,dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, BasicConv=BasicConv):
        super(TransformerBlock_DCNv4, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_DCNv4(scale,dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, BasicConv=BasicConv)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class FECB_SCTB(nn.Module):
    def __init__(self , out_channel, num_res=8, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=1, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias', scale=1):
        super(FECB_SCTB, self).__init__()

        layers = []
        for _ in range(num_res):
            layers.append(ResBlock(out_channel))
            layers.append(TransformerBlock(scale=scale, dim=out_channel, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type, BasicConv=BasicConv))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class FECB_SCTB_DCNv4(nn.Module):
    def __init__(self , out_channel, num_res=8, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=1, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias', scale=1):
        super(FECB_SCTB_DCNv4, self).__init__()

        layers = []
        for _ in range(num_res):
            layers.append(ResBlock(out_channel))
            layers.append(TransformerBlock_DCNv4(scale=scale, dim=out_channel, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type, BasicConv=BasicConv))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class FECB_SCTB_wo_SFFM(nn.Module):
    def __init__(self , out_channel, num_res=8, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=1, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias', scale=1):
        super(FECB_SCTB_wo_SFFM, self).__init__()

        layers = []
        for _ in range(num_res):
            # layers.append(ResBlock(out_channel))
            layers.append(TransformerBlock_DCNv4(scale=scale, dim=out_channel, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type, BasicConv=BasicConv))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DGFM(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(DGFM, self).__init__()
        self.conv_max = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            DeformableConv2d(out_channel, out_channel, kernel_size=7, padding=3, stride=1)
        )
        self.conv_mid = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            DeformableConv2d(out_channel, out_channel, kernel_size=5, padding=2, stride=1)
        )
        self.conv_small = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            DeformableConv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1)
        )

        self.conv1 =BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)


    def forward(self, x_max, x_mid, x_small):

        y_max=x_max + x_mid + x_small

        x_max = self.conv_max(x_max)
        x_mid = self.conv_mid(x_mid)
        x_small = self.conv_small(x_small)

        x =F.tanh(x_mid) * x_max
        x = self.conv1(x)

        x =F.tanh(x_small) * x
        x = self.conv2(x)

        return x+y_max
    
    
class DGFM_SConv(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(DGFM_SConv, self).__init__()
        self.conv_max = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=7, stride=1)
        )
        self.conv_mid = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=5, stride=1)
        )
        self.conv_small = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1)
        )

        self.conv1 =BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)


    def forward(self, x_max, x_mid, x_small):

        y_max=x_max + x_mid + x_small

        x_max = self.conv_max(x_max)
        x_mid = self.conv_mid(x_mid)
        x_small = self.conv_small(x_small)

        x =F.tanh(x_mid) * x_max
        x = self.conv1(x)

        x =F.tanh(x_small) * x
        x = self.conv2(x)

        return x+y_max
    
class DGFM_DWConv(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(DGFM_DWConv, self).__init__()
        self.conv_max = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=7, stride=1, groups=out_channel)
        )
        self.conv_mid = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=5, stride=1, groups=out_channel)
        )
        self.conv_small = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, groups=out_channel)
        )

        self.conv1 =BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True)


    def forward(self, x_max, x_mid, x_small):

        y_max=x_max + x_mid + x_small

        x_max = self.conv_max(x_max)
        x_mid = self.conv_mid(x_mid)
        x_small = self.conv_small(x_small)

        x =F.tanh(x_mid) * x_max
        x = self.conv1(x)

        x =F.tanh(x_small) * x
        x = self.conv2(x)

        return x+y_max


class DAM_wo_gate(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(DAM_wo_gate, self).__init__()
        self.conv_max = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=7, stride=1)
        )
        self.conv_mid = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=5, stride=1)
        )
        self.conv_small = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1)
        )

        self.conv =BasicConv(out_channel * 3, out_channel, kernel_size=1, stride=1, relu=True)

    def forward(self, x_max, x_mid, x_small):
        x_max = self.conv_max(x_max)
        x_mid = self.conv_mid(x_mid)
        x_small = self.conv_small(x_small)

        x = torch.cat((x_max, x_mid, x_mid), dim=1)

        return self.conv(x)

class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-inchannel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class DeRestormer(nn.Module):
    def __init__(self, num_res=8, inference=False):
        super(DeRestormer, self).__init__()
        self.inference = inference
        if not inference:
            BasicConv = BasicConv_do
            ResBlock = ResBlock_do_FECB_bench
        else:
            BasicConv = BasicConv_do_eval
            ResBlock = ResBlock_do_FECB_bench_eval
        base_channel = 32

        heads = [1, 2, 4]
        ffn_expansion_factor = 2.66
        bias = False
        LayerNorm_type = 'WithBias'
        scale = [1,0.5,0.25]

        self.Encoder = nn.ModuleList([
            FECB_SCTB_DCNv4(base_channel, num_res, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=heads[0],
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,scale= scale[0]),
            FECB_SCTB_DCNv4(base_channel * 2, num_res, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=heads[1],
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,scale= scale[1]),
            FECB_SCTB_DCNv4(base_channel * 4, num_res, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=heads[2],
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,scale= scale[2]),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            FECB_SCTB_DCNv4(base_channel * 4, num_res, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=heads[2],
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,scale= 0.25),
            FECB_SCTB_DCNv4(base_channel * 2, num_res, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=heads[1],
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,scale= 0.5),
            FECB_SCTB_DCNv4(base_channel, num_res, ResBlock=ResBlock, BasicConv=BasicConv, num_heads=heads[0],
                   ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,scale= 1)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.DGFMs = nn.ModuleList([
            DGFM(32, 32, BasicConv=BasicConv),
            DGFM(64, 64, BasicConv=BasicConv)
        ])

        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

        self.down_1 = Downsample(32)

        self.up_1 = Upsample(64)
        self.up_2 = Upsample(128)
        self.up_3 = Upsample(64)
    
    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)

        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)

        z = self.Encoder[2](z)

        z21 = self.up_1(res2)
        z42 = self.up_2(z)
        z41 = self.up_3(z42)

        z12 = self.down_1(res1)

        res1 = self.DGFMs[0](res1,z21,z41)
        res2 = self.DGFMs[1](z12,res2,z42)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        if not self.inference:
            outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)

        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        if not self.inference:
            outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)

        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            outputs.append(z+x)
            return outputs[::-1]
        

if __name__ == "__main__":
    # torch.cuda.set_device(0)
    model = DeRestormer_wo_SFFM()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #  [B, C, H, W]
    batch_size = 1
    channels = 3
    height = 256
    width = 256

    input_tensor = torch.randn(batch_size, channels, height, width)
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    print(output)
