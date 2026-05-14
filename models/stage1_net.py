import pdb
import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from einops import rearrange
import math
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def compute_decom_lime(input_img):
    ill_map = torch.max(input_img, dim=1, keepdim=True)[0]
    ref_map = input_img / (ill_map + 1e-6)
    
    return ref_map, ill_map
     
     
class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_block, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        out = self.model(x) + self.conv(x)

        return out


class upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(upsampling, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out
    

class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=(1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=(3, 3), stride=(1, 1),
                                    padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(Cross_Attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):

        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer
    
    
class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=5):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


class Retinex_decom(nn.Module):
    def __init__(self, channels):
        super(Retinex_decom, self).__init__()

        self.conv0 = nn.Conv2d(2, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks0 = nn.Sequential(Res_block(channels, channels),
                                     Res_block(channels, channels),
                                     nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1))

        self.conv1 = nn.Conv2d(1, channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks1 = nn.Sequential(Res_block(channels, channels),
                                     Res_block(channels, channels),
                                     nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1))
        
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            CBAMBlock(channel=channels),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1))
        
        self.blocks2 = nn.Sequential(Res_block(channels * 2, channels),
                                     Res_block(channels, channels),
                                     nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=(1, 1), padding=1))
        
        self.head_params  = nn.Conv2d(channels, 2, 1)
        self.head_weights = nn.Conv2d(channels, 2, 1)
        
    @staticmethod  
    def rolloff(y, k):  # 单调压高光
        return 1. - (1. - y) / (1. + k * (1. - y) + 1e-8)
    
    @staticmethod
    def map_sigmoid(raw, lo, hi):
        return lo + (hi - lo) * torch.sigmoid(raw)
    
    def forward(self, input):
        
        init_reflectance, init_illumination = compute_decom_lime(input)
        
        structure_guidence = self.blocks0(self.conv0(torch.cat([
            init_illumination,
            init_reflectance.var(1, keepdim=True) + 1e-6
            ], dim=1)))
        
        initial_gamma = self.blocks1(self.conv1(init_illumination))
        
        gamma_map = self.channel_attention(self.blocks2(torch.cat((initial_gamma, structure_guidence), dim=1)))
        
        params  = self.head_params(gamma_map)
        weights = F.softmax(self.head_weights(structure_guidence), dim=1)
        gamma_s = self.map_sigmoid(params[:,0:1], 0.0, 1.0)
        gamma_m = self.map_sigmoid(params[:,1:2], 1.0, 5.0)
        
        output_s = torch.pow(init_illumination, gamma_s)
        output_m = torch.pow(init_illumination, gamma_m)

        output_illumination = (weights[:,0:1]*output_s + weights[:,1:2]*output_m)
        
        output_illumination = torch.cat([output_illumination, output_illumination, output_illumination], dim=1)
        
        output_img = output_illumination * init_reflectance

        return output_img


class Net(nn.Module):
    def __init__(self, channels=64):
        super(Net, self).__init__()

        self.retinex = Retinex_decom(channels)

    def forward(self, x):
        
        low_img = x[:, :3, :, :]
        
        output_img = self.retinex(low_img)

        return output_img