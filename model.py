import math
import torch
import torch.nn as nn
from torch.nn import init, Softmax
import torch.nn.functional as F
from utils import weights_init_classifier, weights_init_kaiming
from resnet import resnet50
from loss import MMDLoss

class GeMP(nn.Module):
    def __init__(self, p=3.0, eps=1e-12):
        super(GeMP, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        p, eps = self.p, self.eps
        if x.ndim != 2:
            batch_size, fdim = x.shape[:2]
            x = x.view(batch_size, fdim, -1)
        return (torch.mean(x ** p, dim=-1) + eps) ** (1 / p)


class visible_module(nn.Module):
    def __init__(self, pretrained=True):
        super(visible_module, self).__init__()
        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        # x = self.visible.layer1(x)
        # x = self.visible.layer2(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, pretrained=True):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        # x = self.thermal.layer1(x)
        # x = self.thermal.layer2(x)
        return x


class mid_module(nn.Module):
    def __init__(self, pretrained=True):
        super(mid_module, self).__init__()

        model_mid = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.mid = model_mid

    def forward(self, x):
        x = self.mid.conv1(x)
        x = self.mid.bn1(x)
        x = self.mid.relu(x)
        x = self.mid.maxpool(x)
        # x = self.mid.layer1(x)
        # x = self.mid.layer2(x)
        return x


class base_module12(nn.Module):
    def __init__(self, pretrained=True):
        super(base_module12, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.base = base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        return x


class base_module34(nn.Module):
    def __init__(self, pretrained=True):
        super(base_module34, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.base = base

    def forward(self, x):
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class Enhancement(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Enhancement, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

    def forward(self, x, x1, x2):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x1 = self.phi(x1).view(batch_size, self.inter_channels, -1)
        f1 = torch.matmul(theta_x, phi_x1)
        N1 = f1.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C1 = f1 / N1

        phi_x2 = self.phi(x2).view(batch_size, self.inter_channels, -1)
        f2 = torch.matmul(theta_x, phi_x2)
        N2 = f2.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C2 = f2 / N2

        a = torch.clamp(self.a, 0, 1)
        # b = torch.clamp(self.b, 0, 1)
        f_div_C = self.b * (a * f_div_C1 + (1-a) * f_div_C2)
        # a = torch.clamp(self.a, 0, 1)
        # b = torch.clamp(self.b, 0, 1)
        # f_div_C = a * f_div_C1 + b * f_div_C2

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class Dilation(nn.Module):
    def __init__(self, channel, M: int = 2, G: int = 32, r: int = 16, stride: int = 1, L: int = 32):
        super().__init__()
        d = max(int(channel / r), L)
        self.M = M
        self.c = channel
        # 1.split
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i, groups=G,
                          bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            ))
        # 2.fuse
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(channel, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        # 3.select
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, channel, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        # 1.split
        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.c, feats.shape[2], feats.shape[3])
        # print('feats.shape', feats.shape)
        # 2.fuse
        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)
        # print('feats_U.shape', feats_U.shape)
        # print('feats_S.shape', feats_S.shape)
        # print('feats_Z.shape', feats_Z.shape)
        # 3.select
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        # print('attention_vectors.shape', attention_vectors.shape)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.c, 1, 1)
        # print('attention_vectors.shape', attention_vectors.shape)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(feats * attention_vectors, dim=1)
        # print('feats_V.shape', feats_V.shape)
        return feats_V


class Refinement(nn.Module):
    def __init__(self, cin, cout, flag, r=16, L=32):
        super(Refinement, self).__init__()
        d = max(int(cout / r), L)
        self.c = cout
        if flag == 0:
            self.conv = nn.Conv2d(cin, cout, kernel_size=1, stride=2, padding=0)
        else:
            self.conv = nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0)
        self.dila1 = Dilation(cout)
        self.dila2 = Dilation(cout)
        self.a = nn.Parameter(torch.rand(1))

    def forward(self, x1, x2):
        f1 = self.dila1(self.conv(x1))
        f2 = self.dila2(x2)
        a = torch.clamp(self.a, 0, 1)
        x = a * f1 + (1-a) * f2

        return x


class CosSim(nn.Module):
    def __init__(self):
        super(CosSim, self).__init__()
        self.dim = 2048
        self.part_num = 6
        self.spatial_attention = nn.Conv2d(self.dim, self.part_num, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.constant_(self.spatial_attention.bias, 0.0)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        masks = x
        masks = self.spatial_attention(masks)
        masks = self.activation(masks)
        # feats = []
        for i in range(self.part_num):
            mask = masks[:, i:i + 1, :, :]
            feat = mask * x

            feat = F.avg_pool2d(feat, feat.size()[2:])
            feat = feat.view(feat.size(0), -1)
            v, m, t = feat.chunk(3, 0)
            v = F.normalize(v, dim=1)
            m = F.normalize(m, dim=1)
            t = F.normalize(t, dim=1)
            if i == 0:
                # sim = (F.cosine_similarity(v, m, dim=1) + F.cosine_similarity(t, m, dim=1)) / 2
                dist = 2 / (F.cosine_similarity(v, m, dim=1) + F.cosine_similarity(t, m, dim=1)) - 1
            else:
                # sim = (F.cosine_similarity(v, m, dim=1) + F.cosine_similarity(t, m, dim=1)) / 2
                dist += 2 / (F.cosine_similarity(v, m, dim=1) + F.cosine_similarity(t, m, dim=1)) - 1

            # feats.append(feat)
        loss = torch.mean(dist)
        masks = masks.view(b, self.part_num, w * h)
        loss_ort = torch.bmm(masks, masks.permute(0, 2, 1))
        loss_ort = torch.triu(loss_ort, diagonal=1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
        return loss, loss_ort


class embed_net(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(embed_net, self).__init__()

        self.visible = visible_module(pretrained=pretrained)
        self.base12_v = base_module12(pretrained=pretrained)
        self.thermal = thermal_module(pretrained=pretrained)
        self.base12_t = base_module12(pretrained=pretrained)
        self.mid = mid_module(pretrained=pretrained)
        self.base12_m = base_module12(pretrained=pretrained)

        self.base34 = base_module34(pretrained=pretrained)

        self.part_num = 6
        self.spatial_attention = nn.Conv2d(pool_dim, self.part_num, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.constant_(self.spatial_attention.bias, 0.0)
        self.activation = nn.Sigmoid()

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.visible_classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.infrared_classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.visible_classifier_ = nn.Linear(pool_dim, class_num, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data
        self.infrared_classifier_ = nn.Linear(pool_dim, class_num, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')

        self.relu = nn.ReLU()
        self.pool = GeMP()
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.enhance = Enhancement(64)
        self.refine3 = Refinement(512, 1024, 0)
        self.refine4 = Refinement(1024, 2048, 1)

        self.cos = CosSim()

    def forward(self, x1, mid, x2, modal=0):
        if modal == 0:
            x1 = self.visible(x1)
            x2 = self.thermal(x2)
            mid = self.mid(mid)
            mid = self.enhance(mid, x1, x2)

            x1 = self.base12_v(x1)
            x2 = self.base12_t(x2)
            mid = self.base12_m(mid)

            x_2 = torch.cat((x1, mid, x2), 0)
        elif modal == 1:
            x = self.visible(x1)
            x_2 = self.base12_v(x)
        elif modal == 2:
            x = self.thermal(x2)
            x_2 = self.base12_t(x)

        x_3 = self.base34.base.layer3(x_2)
        x = self.refine3(x_2, x_3)
        x_4 = self.base34.base.layer4(x)
        x = self.refine4(x, x_4)

        x4 = self.relu(x)
        b, c, h, w = x4.shape
        x = x4.view(b, c, h * w)
        x_pool = self.pool(x)
        feat = self.bottleneck(x_pool)

        if self.training:
            x_v, x_a, x_t = feat[0:b // 3], feat[b // 3: (b // 3) * 2], feat[(b // 3) * 2: b]

            # cls_id = self.classifier_idc(feat)

            logit_v = self.visible_classifier(x_v)
            logit_t = self.infrared_classifier(x_t)
            # logit_mml = [logit_v, logit_t]
            logit_ori = torch.cat((logit_v, logit_t), 0).float()

            # according to the 'MPANet'
            with torch.no_grad():
                self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * 0.8 \
                                                        + self.infrared_classifier.weight.data * 0.2
                self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * 0.8 \
                                                       + self.visible_classifier.weight.data * 0.2

                logit_v_ = self.infrared_classifier_(x_v)
                logit_t_ = self.visible_classifier_(x_t)
                logit_now = torch.cat((logit_v_, logit_t_), 0).float()

            logit_ori = F.softmax(logit_ori, 1)
            logit_now = F.softmax(logit_now, 1)
            kl_loss = self.KLDivLoss(logit_now.log(), logit_ori)

            loss, loss_ort = self.cos(x4)

            x_2 = self.relu(x_2)
            b2, c2, h2, w2 = x_2.shape
            x_2 = x_2.view(b2, c2, h2 * w2)
            x2_pool = self.pool(x_2)
            return x_pool, self.classifier(feat), x2_pool, loss, loss_ort, logit_v, logit_t, kl_loss
        else:
            return F.normalize(x_pool, p=2.0, dim=1), F.normalize(feat, p=2.0, dim=1)
