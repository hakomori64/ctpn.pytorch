#-*- coding:utf-8 -*-
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from ctpn.utils import cal_rpn

IMAGE_MEAN = [123.68, 116.779, 103.939]

'''
从xml文件中读取图像中的真值框
'''
def readxml(path):
    gtboxes = []
    xml = ET.parse(path)
    for elem in xml.iter():
        if 'object' in elem.tag:
            for attr in list(elem):
                if 'bndbox' in attr.tag:
                    xmin = int(round(float(attr.find('xmin').text)))
                    ymin = int(round(float(attr.find('ymin').text)))
                    xmax = int(round(float(attr.find('xmax').text)))
                    ymax = int(round(float(attr.find('ymax').text)))
                    gtboxes.append((xmin, ymin, xmax, ymax))

    return np.array(gtboxes)


'''
读取VOC格式数据，返回用于训练的图像、anchor目标框、标签
'''
class VOCDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def generate_gtboxes(self, xml_path,rescale_fac = 1.0):
        base_gtboxes = readxml(xml_path)
        gtboxes = []
        for base_gtbox in base_gtboxes:
            xmin, ymin, xmax, ymax = base_gtbox
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            # 横を16pxづつ区切り、それぞれをgtboxesに入れる
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16*i-0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)

    def __getitem__(self, idx):
        # datadirからイメージのidx番目を抜き出す
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        rescale_fac = max(h, w) / 1000
        # 画像の横幅、縦幅のどちらかが1000を越えていた場合、長いほうが1000になるようにリサイズ
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            img = cv2.resize(img,(w,h))

        xml_path = os.path.join(self.labelsdir, img_name.split('.')[0]+'.xml')
        gtbox = self.generate_gtboxes(xml_path, rescale_fac)

        # 1/2の確率で画像に以下の処理する
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :] # X方向にひっくり返す
            newx1 = w - gtbox[:, 2] - 1 # gtboxもそれに併せてひっくり返す
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr] = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        # cls.shape == (アンカー数), # regr.shape = (アンカー数, 2)
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        # regr:
        # [[cls, anchor[0], anchor[1]] # あるアンカーについて、そのアンカーは検出対象か、中心はどれくらいずれているか、高さはどれくらいことなるか
        # .....
        # ]
        cls = np.expand_dims(cls, axis=0) # １重配列を２重配列に、大きく括弧をつけて囲む

        m_img = img - IMAGE_MEAN # RGBについて平均値を引いておく
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float() # チャンネルから始めるようにする
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr
        # m_img: 正規化後の画像
        # cls: それぞれのアンカーについて、それは検出対象か？ shape -> (numOfAnchor, 3)
        # regr: それぞれのアンカーについて、[それは検出対象か、一番重なりが大きいものとどれくらい中心がずれているか、幅はどれくらいずれているか] shape -> (1, numOfAnchor)


################################################################################


class ICDARDataset(Dataset):
    def __init__(self, datadir, labelsdir):
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)

    def box_transfer(self, coor_lists, rescale_fac = 1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2*i]) for i in range(4)]
            coors_y = [int(coor_list[2*i+1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            gtboxes.append((xmin, ymin, xmax, ymax))
        return np.array(gtboxes)

    def box_transfer_v2(self, coor_lists, rescale_fac = 1.0):
        gtboxes = []
        for coor_list in coor_lists:
            coors_x = [int(coor_list[2 * i]) for i in range(4)]
            coors_y = [int(coor_list[2 * i + 1]) for i in range(4)]
            xmin = min(coors_x)
            xmax = max(coors_x)
            ymin = min(coors_y)
            ymax = max(coors_y)
            if rescale_fac > 1.0:
                xmin = int(xmin / rescale_fac)
                xmax = int(xmax / rescale_fac)
                ymin = int(ymin / rescale_fac)
                ymax = int(ymax / rescale_fac)
            prev = xmin
            for i in range(xmin // 16 + 1, xmax // 16 + 1):
                next = 16*i-0.5
                gtboxes.append((prev, ymin, next, ymax))
                prev = next
            gtboxes.append((prev, ymin, xmax, ymax))
        return np.array(gtboxes)

    def parse_gtfile(self, gt_path, rescale_fac = 1.0):
        coor_lists = list()
        with open(gt_path, 'r', encoding="utf-8-sig") as f:
            content = f.readlines()
            for line in content:
                coor_list = line.split(',')[:8]
                if len(coor_list) == 8:
                    coor_lists.append(coor_list)
        return self.box_transfer_v2(coor_lists, rescale_fac)

    def draw_boxes(self,img,cls,base_anchors,gt_box):
        for i in range(len(cls)):
            if cls[i]==1:
                pt1 = (int(base_anchors[i][0]),int(base_anchors[i][1]))
                pt2 = (int(base_anchors[i][2]),int(base_anchors[i][3]))
                img = cv2.rectangle(img,pt1,pt2,(200,100,100))
        for i in range(gt_box.shape[0]):
            pt1 = (int(gt_box[i][0]),int(gt_box[i][1]))
            pt2 = (int(gt_box[i][2]),int(gt_box[i][3]))
            img = cv2.rectangle(img, pt1, pt2, (100, 200, 100))
        return img

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.datadir, img_name)
        img = cv2.imread(img_path)

        h, w, c = img.shape
        rescale_fac = max(h, w) / 1000
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            img = cv2.resize(img,(w,h))

        gt_path = os.path.join(self.labelsdir, img_name.split('.')[0]+'.txt')
        gtbox = self.parse_gtfile(gt_path, rescale_fac)

        # random flip image
        if np.random.randint(2) == 1:
            img = img[:, ::-1, :]
            newx1 = w - gtbox[:, 2] - 1
            newx2 = w - gtbox[:, 0] - 1
            gtbox[:, 0] = newx1
            gtbox[:, 2] = newx2

        [cls, regr] = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gtbox)
        regr = np.hstack([cls.reshape(cls.shape[0], 1), regr])
        cls = np.expand_dims(cls, axis=0)

        m_img = img - IMAGE_MEAN
        m_img = torch.from_numpy(m_img.transpose([2, 0, 1])).float()
        cls = torch.from_numpy(cls).float()
        regr = torch.from_numpy(regr).float()

        return m_img, cls, regr