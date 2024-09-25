import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

###scSE

class sSE(nn.Module):  # 空间(Space)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        q = self.conv(x)  # b c h w -> b 1 h w
        q = self.norm(q)  # b 1 h w
        return x * q  # 广播机制


class cSE(nn.Module):  # 通道(channel)注意力
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # b c 1 1
        self.relu = nn.ReLU()
        self.Conv_Squeeze = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
        self.Conv_Excitation = nn.Conv2d(in_ch // 2, in_ch, kernel_size=1, bias=False)

    def forward(self, x):
        z = self.avgpool(x)  # b c 1 1
        z = self.Conv_Squeeze(z)  # b c//2 1 1
        z = self.relu(z)
        z = self.Conv_Excitation(z)  # b c 1 1
        z = self.norm(z)
        return x * z.expand_as(x)  # 扩展


class scSE(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.cSE = cSE(in_ch)
        self.sSE = sSE(in_ch)

    def forward(self, x):
        c_out = self.cSE(x)
        s_out = self.sSE(x)
        return c_out + s_out

###scSE

###ODCOV
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)
###ODCOV




def LinearModule(hidden_dim):
    return nn.ModuleList(
        [nn.Linear(hidden_dim, hidden_dim),
         nn.ReLU(inplace=True)])


class FeatureResize(nn.Module):
    def __init__(self, size=(10, 25)):
        super(FeatureResize, self).__init__()
        self.size = size

    def forward(self, x):
        x = F.interpolate(x, self.size)
        return x.flatten(2)


class ROIGather(nn.Module):
    '''
    ROIGather module for gather global information
    Args: 
        in_channels: prior feature channels
        num_priors: prior numbers we predefined
        sample_points: the number of sampled points when we extract feature from line
        fc_hidden_dim: the fc output channel
        refine_layers: the total number of layers to build refine
    '''
    def __init__(self,
                 in_channels,
                 num_priors,
                 sample_points,
                 fc_hidden_dim,
                 refine_layers,
                 mid_channels=48):
        super(ROIGather, self).__init__()
        self.ODCOV = ODConv2d(64, 64, kernel_size=1)
        self.scSE = scSE(64)
        self.in_channels = in_channels
        self.num_priors = num_priors
        self.f_key = ConvModule(in_channels=self.in_channels,
                                out_channels=self.in_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                norm_cfg=dict(type='BN'))

        self.f_query = nn.Sequential(
            nn.Conv1d(in_channels=num_priors,
                      out_channels=num_priors,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=num_priors),
            nn.ReLU(),
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.W = nn.Conv1d(in_channels=num_priors,
                           out_channels=num_priors,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           groups=num_priors)

        self.resize = FeatureResize()
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.convs = nn.ModuleList()
        self.catconv = nn.ModuleList()
        for i in range(refine_layers):
            self.convs.append(
                ConvModule(in_channels,
                           mid_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

            self.catconv.append(
                ConvModule(mid_channels * (i + 1),
                           in_channels, (9, 1),
                           padding=(4, 0),
                           bias=False,
                           norm_cfg=dict(type='BN')))

        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)

        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    def roi_fea(self, x, layer_index):
        feats = []
        for i, feature in enumerate(x):
            feat_trans = self.convs[i](feature)
            feats.append(feat_trans)
        cat_feat = torch.cat(feats, dim=1)
        cat_feat = self.catconv[layer_index](cat_feat)
        return cat_feat

    def forward(self, roi_features, x, layer_index):
        '''
        Args:
            roi_features: prior feature, shape: (Batch * num_priors, prior_feat_channel, sample_point, 1)  type:list
            x: feature map  x: torch.Size([8, 64, 68, 45])
            layer_index: currently on which layer to refine # 0,1,2
        Return: 
            roi: prior features with gathered global information, shape: (Batch, num_priors, fc_hidden_dim)
        '''
        x = self.ODCOV(x)
        x = self.scSE(x)
        x = self.ODCOV(x)
        roi = self.roi_fea(roi_features, layer_index)  #roi = torch.Size([1536, 64, 36, 1])
        bs = x.size(0)
        roi = roi.contiguous().view(bs * self.num_priors, -1)  # roi = torch.Size([1536, 2304])

        roi = F.relu(self.fc_norm(self.fc(roi)))  #roi: torch.Size([1536, 64])

        roi = roi.view(bs, self.num_priors, -1)  #roi: torch.Size([8, 192, 64])

        #return roi  #move roi gather  start here

        query = roi
        value = self.resize(self.f_value(x))
        query = self.f_query(query)
        key = self.f_key(x)
        value = value.permute(0, 2, 1)
        key = self.resize(key)
        sim_map = torch.matmul(query, key)
        sim_map = (self.in_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  #sim_map: torch.Size([8, 192, 250])
        context = torch.matmul(sim_map, value) # context: torch.Size([8, 192, 64])
        context = self.W(context)  # context: torch.Size([8, 192, 64])
        roi = roi + F.dropout(context, p=0.1, training=self.training) # roi torch.Size([8, 192, 64])

        return roi
