
import torch.nn as nn
import torch.nn.functional as F


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats):
        super().__init__()

        self.conv1 = nn.Conv2d(n_feats, esa_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(esa_channels, esa_channels, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(esa_channels, esa_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv_f = nn.Conv2d(esa_channels, esa_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(esa_channels, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self, in_channels, mid_channels=None, out_channels=None, esa_channels=16, act='relu'):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.c2_r = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.c3_r = nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, padding_mode='reflect')

        self.c5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.esa = ESA(esa_channels, out_channels)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'prelu':
            self.act = nn.PReLU(num_parameters=mid_channels, init=0.05)
        else:
            raise ValueError(f"Unsupported activation: {act}")

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out
    

class RLFN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for 
    Efficient Super-Resolution`
    """

    def __init__(self, in_channels=3, out_channels=3, feature_channels=52, upscale=2, act='relu'):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1, padding_mode='reflect')

        self.block_1 = RLFB(feature_channels, act=act)
        self.block_2 = RLFB(feature_channels, act=act)
        self.block_3 = RLFB(feature_channels, act=act)
        self.block_4 = RLFB(feature_channels, act=act)
        self.block_5 = RLFB(feature_channels, act=act)
        self.block_6 = RLFB(feature_channels, act=act)

        self.conv_2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, padding_mode='reflect')

        self.upsampler = nn.Sequential(nn.Conv2d(feature_channels, out_channels * (upscale ** 2), kernel_size=3, padding=1, padding_mode='reflect'),
                                       nn.PixelShuffle(upscale))

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)

        out_low_resolution = self.conv_2(out_b6) + out_feature
        output = self.upsampler(out_low_resolution)

        return output