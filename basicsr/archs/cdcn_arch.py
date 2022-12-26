# GENERATE TIME: Fri Apr  8 06:20:38 2022
# CMD:
# basicsr/train.py -opt=options/train/DC/train_DCv1_x4_5G10B_bs16_setting1.yml

# GENERATE TIME: Tue Apr  5 11:17:46 2022
# CMD:
# basicsr/train.py -opt=options/train/DC/train_DCv1_x4_5G10B_bs16_setting1.yml

# GENERATE TIME: Tue Jan 11 14:40:40 2022
# CMD:
# basicsr/train.py -opt=options/train/DC/train_DCv1_x3_5G10B_bs16_setting1.yml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from basicsr.utils.registry import ARCH_REGISTRY

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class CALayer(nn.Module):
    def __init__(self, nf, reduction=16):
        super(CALayer, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf // reduction, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(nf // reduction, nf // 2, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        y = self.avg(x)
        y = self.body(y)
        return y

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers) 

class MultiscaleFusionModule(nn.Module):
    def __init__(self, num_feat):
        super(MultiscaleFusionModule, self).__init__()
        self.cons3 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2))
        self.cons5 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 5, 1, 2),
                                   nn.LeakyReLU(0.2))
        self.cons7 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 7, 1, 3),
                                   nn.LeakyReLU(0.2))
        self.conr3 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                                   nn.LeakyReLU(0.2))
        self.conr5 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 5, 1, 2),
                                   nn.LeakyReLU(0.2))
        self.conr7 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 7, 1, 3),
                                   nn.LeakyReLU(0.2))
        self.cond3 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                                   nn.LeakyReLU(0.2))
        self.cond5 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 5, 1, 2),
                                   nn.LeakyReLU(0.2))
        self.cond7 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 7, 1, 3),
                                   nn.LeakyReLU(0.2))
        self.dense1 = nn.Sequential(nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1),
                                    nn.LeakyReLU(0.2))
        self.dense2 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                                    nn.LeakyReLU(0.2))
        self.dense3 = nn.Sequential(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1),
                                    nn.LeakyReLU(0.2))
        self.dense4 = nn.Sequential(nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1),
                                    nn.LeakyReLU(0.2))
        self.dense5 = nn.Sequential(nn.Conv2d(num_feat * 4, num_feat, 3, 1, 1),
                                    nn.LeakyReLU(0.2))
        self.dense6 = nn.Sequential(nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                                    nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1),
                                   nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(num_feat * 3, num_feat, 5, 1, 2),
                                   nn.LeakyReLU(0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(num_feat * 3, num_feat, 7, 1, 3),
                                   nn.LeakyReLU(0.2))

    def forward(self, structure, detail):
        structure_3 = self.cons3(structure)
        structure_5 = self.cons5(structure)
        structure_7 = self.cons7(structure)
        detail_3 = self.cond3(detail)
        detail_5 = self.cond5(detail)
        detail_7 = self.cond7(detail)
        recon_3 = self.conr3(detail + structure)
        recon_5 = self.conr5(detail + structure)
        recon_7 = self.conr7(detail + structure)
        feat_3 = self.conv3(torch.cat((structure_3, detail_3, recon_3), dim=1))
        feat_5 = self.conv5(torch.cat((structure_5, detail_5, recon_5), dim=1))
        feat_7 = self.conv7(torch.cat((structure_7, detail_7, recon_7), dim=1))
        dense_1 = self.dense1(
            torch.cat((feat_3, feat_5, feat_7),
                      dim=1))
        dense_2 = self.dense2(dense_1)
        dense_3 = self.dense3(torch.cat((dense_1, dense_2), dim=1))
        dense_4 = self.dense4(torch.cat((dense_1, dense_2, dense_3), dim=1))
        dense_5 = self.dense5(torch.cat((dense_1, dense_2, dense_3, dense_4), dim=1))
        dense_6 = self.dense6(dense_5)
        return detail + structure + dense_6

class MCB(nn.Module):
    def __init__(self, num_feat):
        super(MCB, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )
        self.channel_attention_s = CALayer(nf=num_feat*2)
        self.channel_attention_d = CALayer(nf=num_feat*2)

    def forward(self, input):
        s = input[0]
        d = input[1]
        s_feat = self.body1(s)
        d_feat = self.body2(d)
        cross_feat = torch.cat((s_feat, d_feat), dim=1)
        alpha_s = self.channel_attention_s(cross_feat)
        alpha_d = self.channel_attention_d(cross_feat)
        channel_s = alpha_d * s_feat
        channel_d = alpha_s * d_feat
        s = s + channel_s
        d = d + channel_d

        return [s, d]

class RG(nn.Module):
    def __init__(self, num_feat, num_block):
        super(RG, self).__init__()
        self.residual_group = make_layer(
            MCB,
            num_block,
            num_feat=num_feat)
        self.conv_str = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_detail = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, input):
        x = input[0]
        detail_feat = input[1]
        res, res_detail_feat = self.residual_group([x, detail_feat])
        return [x + self.conv_str(res), detail_feat + self.conv_detail(res_detail_feat)]

@ARCH_REGISTRY.register()
class CDCN(nn.Module):
    def __init__(self, num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_group=5,
                 num_block=10,
                 upscale=2):
        super(CDCN, self).__init__()

        self.num_group = num_group

        self.head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.fex_structure = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )

        self.fex_detail = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )

        self.body = make_layer(
            RG,
            num_group,
            num_feat=num_feat,
            num_block=num_block)

        self.conv_after_body_structure = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_after_body_detail = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.conv_after_body_sr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upsample_structure = Upsample(upscale, num_feat)

        self.upsample_detail = Upsample(upscale, num_feat)

        self.upsample_sr = Upsample(upscale, num_feat)

        self.conv_last_structure = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.conv_last_detail = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.conv_last_sr = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.refine = MultiscaleFusionModule(num_feat=64)

    def forward(self, x):
        init_feat = self.head(x)

        init_structure_feat, init_detail_feat = self.fex_structure(init_feat), self.fex_detail(init_feat)

        res_structure_feat, res_detail_feat = self.body([init_structure_feat, init_detail_feat])

        res_structure_feat = self.conv_after_body_structure(res_structure_feat)

        res_detail_feat = self.conv_after_body_detail(res_detail_feat)

        res_structure_feat += init_feat

        res_detail_feat += init_detail_feat

        structure = self.conv_last_structure(self.upsample_structure(res_structure_feat))

        detail = self.conv_last_detail(self.upsample_detail(res_detail_feat))
        
        sr = self.conv_last_sr(
            self.upsample_sr(self.conv_after_body_sr(self.refine(res_structure_feat,res_detail_feat))))

        return structure, detail, sr

#from torchstat import stat
#import torchvision.models as models
#model = DCV1()
#stat(model, (3, 128, 128))

'''import time
from thop import profile
torch.cuda.synchronize()
model = CDCN(upscale=2).cuda()
model.load_state_dict(torch.load('cdcn_setting1_x2.pth')['params_ema'])
input = torch.randn(1, 3, 128, 128).cuda()
time_start = time.time()
flops, params = profile(model, inputs=(input,))
torch.cuda.synchronize()
time_end = time.time()
print('totally cost', time_end - time_start)
print(flops / 1e9)
print(params / 1e6)'''
# 6.18
# 185.187237152
# 9.939545