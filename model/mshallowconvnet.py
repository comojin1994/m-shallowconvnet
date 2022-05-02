import torch
import torch.nn as nn
from model.layers import Conv2dWithConstraint, LazyLinearWithConstraint


class MShallowConvNet(nn.Module):
    def __init__(self,
                num_channels: int,
                sampling_rate: int,
                dropout_rate: float=0.5):
        super(MShallowConvNet, self).__init__()
    
        kernel_size = int(sampling_rate * 0.12)
        
        pooling_size = 0.3
        hop_size = 0.7
        pooling_kernel_size = int(sampling_rate * pooling_size)
        pooling_stride_size = int(sampling_rate * pooling_size * (1 - hop_size))
        
        depth = 24
        self.temporal_conv = Conv2dWithConstraint(1, depth, kernel_size=[1, kernel_size], padding='same', max_norm=2.)
        self.spatial_conv = Conv2dWithConstraint(depth, depth, kernel_size=[num_channels, 1], padding='valid', max_norm=2.)
        self.bn = nn.BatchNorm2d(depth)
        self.avg_pool = nn.AvgPool2d(kernel_size=[1, pooling_kernel_size], stride=[1, pooling_stride_size])
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        
    
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-06))
        x = self.bn(x)
        
        x = self.dropout(x)
        x = self.flatten(x)
        return x
    

class classifier(nn.Module):
    def __init__(self, num_classes):
        super(classifier, self).__init__()
        
        self.flatten = nn.Flatten()
        self.dense = LazyLinearWithConstraint(num_classes, max_norm=0.5)

        
    def forward(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        return x


class Net(nn.Module):   
    def __init__(self,
                num_classes: int,
                num_channels: int,
                sampling_rate: int):
        super(Net, self).__init__()

        self.backbone = MShallowConvNet(num_channels=num_channels,
                                        sampling_rate=sampling_rate)

        self.classifier = classifier(num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def get_model(args):
    
    model = Net(num_classes=args.num_classes,
                num_channels=args.num_channels,
                sampling_rate=args.sampling_rate)

    return model

