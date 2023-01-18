import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(1024, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(1024, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.dc1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024+1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, track_running_stats=False),
            nn.ReLU(True),
            )
        self.dc2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512+512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),
        )
        self.dc3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256+256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )
        self.dc4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128+128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )
        self.dc5 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )
        self.attrin = nn.Sequential(
            nn.Linear(in_features=38, out_features=16 * 16),
            nn.ReLU(True)
        )    
        self.attrmiddle = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True)
        )
        self.attrdc1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True) 
        )
        self.attrdc2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True) 
        )
        self.attrdc3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True) 
        )

        if init_weights:
            self.init_weights()

    def forward(self, x, attr):
        x = self.encoder(x)
        x = self.middle(x)
        attr = self.attrin(attr).view(-1, 1, 16, 16)
        attr1 = self.attrmiddle(attr)
        x = self.dc1(torch.cat((x, attr1), dim=1))
        attr2 = self.attrdc1(attr1)
        x = self.dc2(torch.cat((x, attr2), dim=1))
        attr3 = self.attrdc2(attr2)
        x = self.dc3(torch.cat((x, attr3), dim=1))
        attr4 = self.attrdc3(attr3)
        x = self.dc4(torch.cat((x, attr4), dim=1))
        x = self.dc5(x)
        x = (torch.tanh(x) + 1) / 2

        return x
        

class AttrGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(AttrGenerator, self).__init__()

        self.ShareConv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.GenderConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.GenderFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=1),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.NoseConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.NoseFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=2),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.MouthConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.MouseFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=4),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.EyesConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.EyesFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=5),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.FaceConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.FaceFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=6),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.AroundHeadConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.AroundHeadFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=11),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.FacialHairConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.FacialHairFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=5),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.CheeksConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.CheeksFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=2),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.FatConv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2))

        self.FatFC = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(in_features=512, out_features=2),
            nn.ReLU(True),
            nn.Dropout(0.5))

        self.Fusion = nn.Sequential(
            nn.Linear(in_features=38, out_features=38))

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.ShareConv(x)
        Gender = self.GenderConv(x).view(-1, 512)
        Gender = self.GenderFC(Gender)
        Nose = self.NoseConv(x).view(-1, 512)
        Nose = self.NoseFC(Nose)
        Mouth = self.MouthConv(x).view(-1, 512)
        Mouth = self.MouseFC(Mouth)
        Eyes = self.EyesConv(x).view(-1, 512)
        Eyes = self.EyesFC(Eyes)
        Face = self.FaceConv(x).view(-1, 512)
        Face = self.FaceFC(Face)
        AroundHead = self.AroundHeadConv(x).view(-1, 512)
        AroundHead = self.AroundHeadFC(AroundHead)
        FacialHair = self.FacialHairConv(x).view(-1, 512)
        FacialHair = self.FacialHairFC(FacialHair)
        Cheeks = self.CheeksConv(x).view(-1, 512)
        Cheeks = self.CheeksFC(Cheeks)
        Fat = self.FatConv(x).view(-1, 512)
        Fat = self.FatFC(Fat)
        out = self.Fusion(torch.cat((Gender, Nose, Mouth, Eyes, Face, AroundHead, FacialHair, Cheeks, Fat), dim=1))
        out = torch.sigmoid(out)
        return out

class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ResnetBlock(nn.Module):
    def   __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out
 

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
