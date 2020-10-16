#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ctpn import config

'''
回归损失: smooth L1 Loss
只针对正样本求取回归损失
L = 0.5*x**2  |x|<1
L = |x| - 0.5
sigma: 平滑系数
1、从预测框p和真值框g中筛选出正样本
2、|x| = |g - p|
3、求取loss，这里设置了一个平滑系数 1/sigma
  (1) |x|>1/sigma: loss = |x| - 0.5/sigma
  (2) |x|<1/sigma: loss = 0.5*sigma*|x|**2
'''
class RPN_REGR_Loss(nn.Module):
    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device
    
    def forward(self, input, target):
        try:
            cls = target[0, :, 0]
            regression = target[0, :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, 0]
            regr_true = regression[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff<1.0/self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            print('RPN_REGR_Loss Exception:', e)
            loss = torch.tensor(0.0)

        return loss.to(self.device)

'''
分类损失: softmax loss
1、OHEM模式
  (1) 筛选出正样本，求取softmaxloss
  (2) 求取负样本数量N_neg, 指定样本数量N, 求取负样本的topK loss, 其中K = min(N_neg, N - len(pos_num))
  (3) loss = loss1 + loss2
2、求取NLLLoss，截断在(0, 10)区间
'''
class RPN_CLS_Loss(nn.Module):
    def __init__(self,device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        # input: 学習器から出てきたcls分類の結果 shape -> [b, h * w * 10, 2]
        # target: 正解 shape -> (batch_size, (1, (anchor_size)))
        if config.OHEM:
            cls_gt = target[0][0] # 各アンカーについて、それが検出対象かどうか
            num_pos = 0
            loss_pos_sum = 0

            if len((cls_gt == 1).nonzero()) != 0: 
                cls_pos = (cls_gt == 1).nonzero()[:, 0] # 検出対象となるアンカー(正解)のインデックスを取得
                gt_pos = cls_gt[cls_pos].long() # [1, 1, ...., 1, 1, 1]
                cls_pred_pos = input[0][cls_pos] # 学習器から出てきたinputの検出対象となるアンカーの行列のみ抜き出す
                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1)) # ロスを計算
                loss_pos_sum = loss_pos.sum() # ロスの合計
                num_pos = len(loss_pos) # 検出対象となるアンカーが何個あったかを保存

            cls_neg = (cls_gt == 0).nonzero()[:, 0] # 検出対象でないアンカーのインデックスの取得
            gt_neg = cls_gt[cls_neg].long() # [0, 0, 0, 0, 0, 0, 0]
            cls_pred_neg = input[0][cls_neg] # 学習器から出てきたinputの検出対象でないアンカーの行列のみ抜き出す

            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1)) # ロスの計算
            loss_neg_topK, _ = torch.topk(loss_neg, min(len(loss_neg), config.RPN_TOTAL_NUM - num_pos)) # 閾値を越えないよう数だけ大きい順に抜きだす
            loss_cls = loss_pos_sum + loss_neg_topK.sum() # 1がちゃんと検出対象であるか＆0がちゃんと検出対象でないかの合計
            loss_cls = loss_cls / config.RPN_TOTAL_NUM # 正規化

            return loss_cls.to(self.device)
        else:
            y_true = target[0][0] # 各アンカーについて、それが検出対象かどうか
            cls_keep = (y_true != -1).nonzero()[:, 0] # 値が(0, 1)のインデックスを取得
            cls_true = y_true[cls_keep].long() # [[0,1]*]
            cls_pred = input[0][cls_keep] # 正解ラベルのほうの値が(0, 1)のインデックスに対応する値を取得
            loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)
            loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)

            return loss.to(self.device)


class basic_conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(basic_conv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


'''
image -> feature map -> rpn -> blstm -> fc -> classifier
                                           -> regression
'''
class CTPN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)
        # stride 1, 0paddingを画像の両サイドに追加する(padding=1)ことで画像サイズを変えないようにする
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nn.GRU(512,128, bidirectional=True, batch_first=True)
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        # h: feature map1枚の縦の長さ(元画像の高さの1/16)
        # w: feature map1枚の横の長さ(元画像の幅の1/16)
        # rpn
        x = self.rpn(x)    #[b, c, h, w](画像の枚数、チャンネル数、高さ、幅)

        
        x1 = x.permute(0, 2, 3, 1).contiguous()  # channels last   [b, h, w, c: 512]
        b = x1.size()  # b, h, w, c
        x1 = x1.view(b[0]*b[1], b[2], b[3]) # b * h, w, c

        x2, _ = self.brnn(x1)
        # x2.size() [b * h, w, 256]

        xsz = x.size() # [b, c, h, w]
        print(xsz)
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)  # torch.Size([4(batch size), 20(height), 20(width), 256])

        x3 = x3.permute(0, 3, 1, 2).contiguous()  # channels first [b, 256, h, w]
        x3 = self.lstm_fc(x3) # [b, 512, h, w]
        x = x3

        cls = self.rpn_class(x) # [b, 20, h, w]
        regression = self.rpn_regress(x) # [b, 20, h, w]

        cls = cls.permute(0, 2, 3, 1).contiguous() # [b, h, w, 20]
        regression = regression.permute(0, 2, 3, 1).contiguous() # [b, h, w, 20]

        cls = cls.view(cls.size(0), cls.size(1)*cls.size(2)*10, 2) # [b, h * w * 10, 2]
        regression = regression.view(regression.size(0), regression.size(1)*regression.size(2)*10, 2) # [b, h * w * 10, 2]

        return cls, regression

if __name__=='__main__':
    CTPN_Model()
