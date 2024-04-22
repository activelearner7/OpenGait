import torch
import copy
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv2d, SetBlockWrapper, SetBlockWrapper3D, HorizontalPoolingPyramid, PackSequenceWrapper

def max_mean_pooling(x, dim):
    max_values, _ = torch.max(x, dim=dim)
    mean_values = torch.mean(x, dim=dim)
    return max_values + mean_values

class OdeNet(BaseModel):

    def build_network(self, model_cfg):
        in_c = model_cfg['in_channels']
        self.set_block1 = nn.Sequential(nn.Conv2d(in_c[0], in_c[1], kernel_size=3,
                                                stride=1, padding=1),
                                        nn.BatchNorm2d(in_c[1]),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.set_block2 = nn.Sequential(nn.Conv2d(in_c[1], in_c[2], kernel_size=3,
                                                stride=1, padding=1),
                                        nn.BatchNorm2d(in_c[2]),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.set_block3 = nn.Sequential(nn.Conv2d(in_c[2], in_c[3], kernel_size=3,
                                                stride=1, padding=1),
                                        nn.BatchNorm2d(in_c[3]),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.set_block4 = nn.Sequential(nn.Conv2d(in_c[3], in_c[4], kernel_size=1,
                                                    stride=1, padding=0),
                                        nn.BatchNorm2d(in_c[4]),
                                        nn.ReLU(inplace=True))

        self.set_block5 = nn.Sequential(nn.Conv3d(in_c[4], in_c[5], kernel_size=3,
                                                  stride=(3,1,1), padding=1),
                                                  nn.BatchNorm3d(in_c[5]),
                                                  nn.ReLU(inplace=True))


        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)
        self.set_block4 = SetBlockWrapper(self.set_block4)
        self.set_block5 = SetBlockWrapper3D(self.set_block5)

        self.set_pooling = PackSequenceWrapper(torch.max)

        self.fc1 = SeparateFCs(**model_cfg['SeparateFC1'])
        self.fc2 = SeparateFCs(**model_cfg['SeparateFC2'])
        self.fc3 = SeparateFCs(**model_cfg['SeparateFC3'])
        self.fc4 = SeparateFCs(**model_cfg['SeparateFC4'])
        # self.Head = SeparateFCs(**model_cfg['SeparateFCs'])

        # self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, labs, positionalLabels, videoLabels, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        print(sils.size())
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        print("sils size:", sils.size())
        outs = self.set_block1(sils)
        print("outs",outs.size())
        outs = self.set_block2(outs)
        print("outs",outs.size())
        outs = self.set_block3(outs)
        print("outs",outs.size())
        outs = self.set_block4(outs)
        print("outs",outs.size())

        # 3D convolution, for temporal learning
        outs = self.set_block5(outs)
        print("outs",outs.size())

        # Remove the T dimension
        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        print("outs",outs.size())

        # flatten the last 2 dimension
        outs = torch.reshape(outs, (outs.size(0), outs.size(1), -1))
        print("outs",outs.size())

        # Fully Connected for Triplet Loss
        # Layer1
        feature1 = self.fc1(outs)
        print("outs",feature1.size())
        # Layer2
        embs_triloss = self.fc2(feature1)
        print("embs_triloss",embs_triloss.size())

        # Fully Connected for Cross Entropy
        # Layer1
        feature2 = self.fc3(outs)
        print("outs",feature2.size())
        # Layer2
        embs_ce = self.fc4(feature2)
        print("embs_ce",embs_ce.size())

        # Horizontal Pooling Matching, HPM
        # feature1 = self.HPP(outs)  # [n, c, p]
        # feature2 = self.HPP(gl)  # [n, c, p]
        # feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        # embs = self.Head(feature)

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
