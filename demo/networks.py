import random
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ED_Aggregation(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh, inner_batch=2, count=0):
        super(ED_Aggregation, self).__init__()
        ## downsampling
        expand_ratio = 2
        network1 = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*4*expand_ratio, kernel_size=3, stride=1, padding=1, bias=False),  # 8x8 bucket
            nn.GroupNorm(16*4*expand_ratio, nf*4*expand_ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4*expand_ratio, nf*4, kernel_size=3, stride=1, padding=1, bias=False),  # 8x8 bucket
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*8, nf*8*expand_ratio, kernel_size=3, stride=1, padding=1, bias=False),  # 4x4 bucket
            nn.GroupNorm(16*8*expand_ratio, nf*8*expand_ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8*expand_ratio, nf*8, kernel_size=3, stride=1, padding=1, bias=False),  # 4x4 bucket
            nn.GroupNorm(16*8, nf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*8, zdim*2, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        combine_net = [
            nn.Conv2d(zdim, zdim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(zdim, zdim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        ]
        ## upsampling
        network = [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network1 = nn.Sequential(*network1)
        #self.network2 = nn.Sequential(*network2)
        self.combine_net = nn.Sequential(*combine_net)
        self.network = nn.Sequential(*network)
        self.inner_batch = inner_batch
        self.zdim = zdim
        self.count = count

    def forward(self, input, rand_inner_batch):
        latent_codes = []
        weights = []
        for i in range(rand_inner_batch):
            feat = self.network1(input[:,i*3:(i+1)*3, :, :])
            latent_code = feat[:,:self.zdim,:,:]
            weight = feat[:,self.zdim:,:,:]
            latent_codes.append(latent_code)
            weights.append(weight)
        weights = torch.cat(weights,2)
        weights = F.softmax(weights,2)
        lc_n = 0
        for i in range(rand_inner_batch):
            lc_n += weights[:,:,i,:].unsqueeze(2)*latent_codes[i]
        #lc_cat = sum(share_code) / self.inner_batch
        lc = self.combine_net(lc_n)
        lc = self.network[:18](lc)
        out = self.network[18:](lc)
        return out

class ED_attribute_refining(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(ED_attribute_refining, self).__init__()
        ## downsampling
        expand_ratio = 2
        network_consist = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4+nf*4, nf*4*expand_ratio, kernel_size=3, stride=1, padding=1, bias=False),  # 8x8 bucket
            nn.GroupNorm(16*4*expand_ratio, nf*4*expand_ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4*expand_ratio, nf*4, kernel_size=3, stride=1, padding=1, bias=False),  # 8x8 bucket
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*8+nf*8, nf*8*expand_ratio, kernel_size=3, stride=1, padding=1, bias=False),  # 4x4 bucket
            nn.GroupNorm(16*8*expand_ratio, nf*8*expand_ratio),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8*expand_ratio, nf*8, kernel_size=3, stride=1, padding=1, bias=False),  # 4x4 bucket
            nn.GroupNorm(16*8, nf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),

            nn.Conv2d(zdim*2, zdim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(zdim, zdim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)]
        
        network_inject = [
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, 32),
            nn.LeakyReLU(0.2, inplace=True),
    
            nn.Conv2d(32, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]

        ## spatial concat layer
        fusion2_1 = [
            nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True)
        ]
        fusion2_2 = [
            nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True)
        ]
        fusion3_1 = [
            nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True)
        ]
        fusion3_2 = [
            nn.Conv2d(nf*2, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True)
        ]

        ## upsampling
        network_decoder = [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64x64 -> 128x128
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network_decoder += [activation()]
        self.network_consist = nn.Sequential(*network_consist)
        self.network_inject = nn.Sequential(*network_inject)
        self.network_decoder = nn.Sequential(*network_decoder)
        self.fusion2_1 = nn.Sequential(*fusion2_1)
        self.fusion2_2 = nn.Sequential(*fusion2_2)
        self.fusion3_1 = nn.Sequential(*fusion3_1)
        self.fusion3_2 = nn.Sequential(*fusion3_2)
        self.softmax = nn.Softmax2d()
        self.zdim = zdim

    def forward(self, input, input_single):
        img_f1 = self.network_inject[:12](input_single) # 8x8 
        img_f2 = self.network_inject[12:14](img_f1) # 4x4
        img_f3 = self.network_inject[14:](img_f2) # 1x1

        lc1 = self.network_consist[:3](input) # 64x64
        lc2 = self.network_consist[3:6](lc1) # 32x32
        lc3 = self.network_consist[6:12](lc2) # 8x8
        lc3 = torch.cat([lc3, img_f1],1)

        lc4 = self.network_consist[12:20](lc3) # 4x4
        lc4 = torch.cat([lc4, img_f2],1)

        lc5 = self.network_consist[20:28](lc4) # 1x1
        lc5 = torch.cat([lc5, img_f3],1)
        lc5 = self.network_consist[28:](lc5)

        out5 = self.network_decoder[:5](lc5) # 8x8

        out4 = self.network_decoder[5:8](out5) # 16x16

        out3 = self.network_decoder[8:11](out4) # 32x32
        att_map3 = self.softmax(self.fusion3_1(torch.cat([out3, lc2],1)))
        out3 = self.fusion3_2(torch.cat([out3, lc2*att_map3], 1))

        out2 = self.network_decoder[11:15](out3) # 64x64
        att_map2 = self.softmax(self.fusion2_1(torch.cat([out2, lc1],1)))
        out2 = self.fusion2_2(torch.cat([out2, lc1*att_map2], 1))

        out1 = self.network_decoder[15:](out2)

        return out1

class Encoder(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False), # 128x128 -> 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)