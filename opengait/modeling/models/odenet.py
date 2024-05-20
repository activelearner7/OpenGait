import torch
import copy
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv2d, SetBlockWrapper, SetBlockWrapper3D, HorizontalPoolingPyramid, PackSequenceWrapper

def max_mean_pooling(x, dim):
    max_values, _ = torch.max(x, dim=dim)
    mean_values = torch.mean(x, dim=dim)
    return max_values + mean_values

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        n, c, s, h, w = x.size()
        x = self.conv(x.transpose(
            1, 2).reshape(-1, c, h, w))
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
        # x = self.conv(x)
        # return x

class Conv1x1Block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv1x1Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        n, c, s, h, w = x.size()
        x = self.conv(x.transpose(
            1, 2).reshape(-1, c, h, w))
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
    
class Conv3DBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv3DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,
                      stride=(1,1,1), padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        n, c, s, h, w = x.size()
        x = self.conv(x.reshape(-1, c, s, h, w))
        output_size = x.size()
        return x.contiguous()

class MaxPoolBlock(nn.Module):
    def __init__(self):
        super(MaxPoolBlock, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        n, c, s, h, w = x.size()
        x = self.Maxpool(x.transpose(
            1, 2).reshape(-1, c, h, w))
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
       
class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        n, c, s, h, w = x.size()
        x = self.up(x.transpose(
            1, 2).reshape(-1, c, h, w))
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
        


class OdeNet(BaseModel):

    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        self.Maxpool = MaxPoolBlock()
        self.block1 = ConvBlock(ch_in=in_c[0], ch_out=in_c[1])

        self.block2 = ConvBlock(ch_in=in_c[1], ch_out=in_c[2])

        self.block3 = ConvBlock(ch_in=in_c[2], ch_out=in_c[3])

        self.block4 = Conv1x1Block(ch_in=in_c[3],ch_out=in_c[4])

        self.block5 = Conv3DBlock(ch_in=in_c[4], ch_out=in_c[5])

        self.block6 = Conv3DBlock(ch_in=in_c[5],ch_out=in_c[6])

        self.block7 = UpConv(ch_in=in_c[6],ch_out=in_c[7])

        self.block8 = ConvBlock(ch_in=in_c[6], ch_out=in_c[7])
        
        self.block9 = UpConv(ch_in=in_c[7],ch_out=in_c[8])

        self.block10 = ConvBlock(ch_in=in_c[7],ch_out=in_c[8])

        self.block11 = UpConv(ch_in=in_c[8],ch_out=in_c[9])

        self.block12 = ConvBlock(ch_in=in_c[8],ch_out=in_c[9])

        self.block13 = Conv1x1Block(ch_in=in_c[9], ch_out=in_c[10])



        

        
        # self.set_block2 = nn.Sequential(nn.Conv2d(in_c[1], in_c[2], kernel_size=3,
        #                                         stride=1, padding=1),
        #                                 nn.BatchNorm2d(in_c[2]),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.MaxPool2d(kernel_size=2, stride=2))
        
        # self.set_block3 = nn.Sequential(nn.Conv2d(in_c[2], in_c[3], kernel_size=3,
        #                                         stride=1, padding=1),
        #                                 nn.BatchNorm2d(in_c[3]),
        #                                 nn.ReLU(inplace=True),
        #                                 nn.MaxPool2d(kernel_size=2, stride=2))
        
        # self.set_block4 = nn.Sequential(nn.Conv2d(in_c[3], in_c[4], kernel_size=1,
        #                                             stride=1, padding=0),
        #                                 nn.BatchNorm2d(in_c[4]),
        #                                 nn.ReLU(inplace=True))

        # self.set_block5 = nn.Sequential(nn.Conv3d(in_c[4], in_c[5], kernel_size=3,
        #                                           stride=(3,1,1), padding=1),
        #                                           nn.BatchNorm3d(in_c[5]),
        #                                           nn.ReLU(inplace=True))


        # self.set_block1 = SetBlockWrapper(self.set_block1)
        # self.set_block2 = SetBlockWrapper(self.set_block2)
        # self.set_block3 = SetBlockWrapper(self.set_block3)
        # self.set_block4 = SetBlockWrapper(self.set_block4)
        # self.set_block5 = SetBlockWrapper3D(self.set_block5)

        # self.set_pooling = PackSequenceWrapper(torch.max)

        # self.fc1 = SeparateFCs(**model_cfg['SeparateFC1'])
        # self.fc2 = SeparateFCs(**model_cfg['SeparateFC2'])
        # self.fc3 = SeparateFCs(**model_cfg['SeparateFC3'])
        # self.fc4 = SeparateFCs(**model_cfg['SeparateFC4'])
       

    def forward(self, inputs):
        ipts, labs, positionalLabels, videoLabels, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        print(sils.size())
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        del ipts

        first_frames = sils[:, :, 0, :, :]
        print(first_frames.shape)

        x1 = self.block1(sils)
        print("x1",x1.shape)
        x2 = self.Maxpool(x1)
        print("x2",x2.shape)

        x2 = self.block2(x2)
        x3 = self.Maxpool(x2)
        print("x3",x3.shape)
        
        x3 = self.block3(x3)
        x4 = self.Maxpool(x3)
        print("x4",x4.shape)

        x4 = self.block4(x4)
        print("x4",x4.shape)
        
        # 3 D Conv Starts
        x5 = self.block5(x4)
        print("x5",x5.shape)

        x6 = self.block6(x5)
        print("x6",x6.shape)

        # UP Conv Starts
        
        # UPConv1
        d4 = self.block7(x6)
        print("d4",d4.shape)

        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.block8(d4)
        print("d4",d4.shape)

        # UPConv2
        d3 = self.block9(d4)
        print("d3",d3.shape)
        d3 = torch.cat((x2,d3),dim = 1)
        d3 = self.block10(d3)
        print("d3",d3.shape)

        # UPConv3 
        d2 = self.block11(d3)
        print("d2",d2.shape)
        d2 = torch.cat((x1,d2),dim = 1)
        d2 = self.block12(d2)
        print("d2",d2.shape)

        # Final Image
        d1 = self.block13(d2)
        print("d1",d1.shape)

        # Remove the T dimension
        outs = self.set_pooling(x5, seqL, options={"dim": 2})[0]
        print("outs", outs.shape())
        # 128 x 64 x 8 x 8

        # flatten the last 2 dimension
        outs = torch.reshape(outs, (outs.size(0), outs.size(1), -1))
        print("outs", outs.shape())
        # 128 x 64 x 64


        # Fully Connected for Triplet Loss
        # Layer1
        feature1 = self.fc1(outs)

        # Layer22
        embs_triloss = self.fc2(feature1)


        # Fully Connected for Cross Entropy
        # Layer1
        feature2 = self.fc3(outs)

        # Layer2
        embs_ce = self.fc4(feature2)


        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs_triloss, 'labels': labs},
                'softmax': {'logits': embs_ce, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embs_triloss
            }
        }
        return retval
