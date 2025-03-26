import torch.nn as nn
import torch


# Resnet 18/34ʹ�ô˲в��
class BasicBlock(nn.Module):  # ���2�㣬F(X)��X��ά�����
    # expansion��F(X)���Xά����չ�ı���
    expansion = 1  # �в�ӳ��F(X)��ά����û�з����仯��1��ʾû�б仯��downsample=None

    # in_channel����������������(ͼ��ͨ���������������RGB����������ʹ��������������������3)��out_channel���������������(����˸���)��stride���������downsample���������в����ݺ;�����ݵ�shape�����ͬ������ֱ�ӽ�����Ӳ�����
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN����conv��relu��֮��

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out


# Resnet 50/101/152ʹ�ô˲в��
class Bottleneck(nn.Module):  # ���3�㣬F(X)��X��ά�Ȳ���
    """
    ע�⣺ԭ�����У������߲в�ṹ������֧�ϣ���һ��1x1�����Ĳ�����2���ڶ���3x3����㲽����1��
    ����pytorch�ٷ�ʵ�ֹ������ǵ�һ��1x1�����Ĳ�����1���ڶ���3x3����㲽����2��
    ��ô���ĺô����ܹ���top1���������0.5%��׼ȷ�ʡ�
    """
    # expansion��F(X)���Xά����չ�ı���
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # �˴�width=out_channel

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        # downsample���������в����ݺ;�����ݵ�shape�����ͬ������ֱ�ӽ�����Ӳ�����
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # ʹ�õĲв������
                 blocks_num,  # ÿ������㣬ʹ�òв��ĸ���
                 num_classes=1000,  # ѵ������ǩ�ķ������
                 include_top=True,  # �Ƿ��ڲв�ṹ�����pooling��fc��softmax
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # ��һ������������������ȣ�Ҳ�Ǻ��������������������

        self.groups = groups
        self.width_per_group = width_per_group

        # �������RGB����������ʹ��������������������3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # _make_layer(�в�����ͣ��в���е�һ�������ľ���˸������в��������в���о������)���������ɶ�������Ĳв��Ĳв�ṹ
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:  # Ĭ��ΪTrue������pooling��fc��softmax
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # ����Ӧƽ���ػ��²�����������������shapeΪ���٣�output size��Ϊ�ĸ߿��Ϊ1x1
            # ʹ����չƽΪ�������磨W,H,C��->(1,1,W*H*C)�����ΪW*H*C
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # ȫ���Ӳ㣬512 * block.expansionΪ������ȣ�num_classesΪ����������

        for m in self.modules():  # ��ʼ��
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # _make_layer()���������ɶ�������Ĳв�飬(�в�����ͣ��в���е�һ�������ľ���˸������в��������в���о������)
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # Ѱ�ң����������Ϊ1����������б仯������F(X)��X��shape��ͬ�Ĳв�飬��Ҫ��X�����²���������ʹ֮shape��ͬ
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # layers����˳�򴢴�������в��
        # ÿ���в�ṹ����һ���в���Ϊ��Ҫ��X�²����Ĳв�飬����Ĳв�鲻��Ҫ��X�²���
        layers = []
        # ��ӵ�һ���в�飬��һ���в���Ϊ��Ҫ��X�²����Ĳв��
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion
        # ����Ĳв�鲻��Ҫ��X�²���
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # �Էǹؼ��ֲ�����ʽ����layers�б�����Sequential(),ʹ���вв�鴮��Ϊһ���в�ṹ
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:  # һ��ΪTrue
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

# ����resnet�Ļ�����ܾ�д����
# ��������������������������������������������������������������������������������������������������������������������������������������������������������������������
# ���涨�岻ͬ���resnet


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

