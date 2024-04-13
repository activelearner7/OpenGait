import torch
import copy
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv2d, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper


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
                                                    stride=1, padding=1),
                                        nn.BatchNorm2d(in_c[4]),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block5 = nn.Sequential(nn.Conv3d(in_c[4], in_c[5], kernel_size=3,
                                                  stride=1, padding=1),
                                                  nn.BatchNorm2d(in_c[5]),
                                                  nn.ReLU(inplace=True))


        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)
        self.set_block4 = SetBlockWrapper(self.set_block4)
        self.set_block5 = SetBlockWrapper(self.set_block5)

        # self.set_pooling = PackSequenceWrapper(torch.max)

        # self.Head = SeparateFCs(**model_cfg['SeparateFCs'])

        # self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

    def forward(self, inputs):
        ipts, labs, positionalLabels, videoLabels, seqL = inputs
        sils = ipts[0]  # [n, s, h, w]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts
        outs = self.set_block1(sils)
        gl = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, seqL, options={"dim": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs = self.Head(feature)

        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embs, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embs
            }
        }
        return retval
