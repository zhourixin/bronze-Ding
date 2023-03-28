import torch
import torch.utils.data as data
import torch.nn as nn
from PIL import Image
import math
import os
import numpy as np
from lxml import etree
import pandas as pd
import sys
import time
import math


dating_tree_dict = {
           '1': [1,1],
           '2': [1,2],
           '3': [2,3],
           '4': [2,4],
           '5': [2,5],
           '6': [3,6],
           '7': [3,7],
           '8': [3,8],
           '9': [4,9],
           '10': [4,10],
           '11': [4,11]
}


complete_attribute={
        'sdlq	獸面紋帶 列旗脊/雲雷紋':0,
        'sdsb	獸面紋帶 省變':1,
        'sdst	獸面紋帶 雙體軀幹':2,
        'sd`	獸面紋帶 其他或看不清的':3,
        'sws	獸面紋 尾上卷':4,
        'swx	獸面紋 尾下卷':5,
        'sdt	獸面紋 獨體':6,
        'sfj	獸面紋 分解':7,
        'sh	獸面紋 有火紋':8,
        's`	獸面紋 其他':9,
        'ssz	足部獸首':10,
        'drz	夔龍紋 直身':11,
        'drd	夔龍紋 低頭卷尾':12,
        'drq	夔龍紋 曲身拱背':13,
        'drjb	夔龍紋 卷鼻':14,
        'drqq	夔龍紋 捲曲':15,
        'drj	交龍紋':16,
        'drds	龍紋 單首雙身':17,
        'drjd	卷龍紋':18,
        'tdzs	顧龍紋 折身':19,
        'td`	顧龍紋 其他（斜身、拱背、雙首）':20,
        'tdft	顧龍紋 分體':21,
        'pc	蟠螭紋':22,
        'ph	蟠虺紋':23,
        'vb	直立鳥紋':24,
        'sb	小鳥紋':25,
        'tsb	回首的小鳥紋':26,
        'bchw	長尾鳥紋':27,
        'bcw	分離C形尾長鳥紋':28,
        'bsw	分離S形尾長鳥紋':29,
        'bb	大鳥紋（昂首）':30,
        'tbb	大鳥紋（回首）':31,
        'sbd	鳥首龍身紋':32,
        'tsbd	鳥首龍身紋（回首）':33,
        'db	龍首鳥身':34,
        'snakes	蛇紋':35,
        'vc	蟬紋':36,
        'sc	連續的蟬紋':37,
        'cz	蟬紋（足）':38,
        'elephants	象紋':39,
        'crescents	四瓣目紋':40,
        'ead	斜角目紋':41,
        'eas	目雲紋':42,
        'ch	重環紋':43,
        'scales	鱗紋':44,
        'qqfl	分離的分解獸面竊曲紋':45,
        'qqsb	有省變的分解獸面紋':46,
        'qqkl	夔龍紋演變的竊曲紋':47,
        'qqs	S形竊曲紋':48,
        'qqu	U形竊曲紋':49,
        'qqg	G形竊曲紋':50,
        'qq`	其他竊曲紋':51,
        's	普通雲雷紋':52,
        't	勾連雷紋':53,
        'rd	乳釘雷紋':54,
        'xjy	斜角雲紋':55,
        'lg	菱格雷紋':56,
        'wc	圓渦紋':57,
        'rd	乳釘紋':58,
        'sj	三角紋':59,
        'jy	蕉葉紋':60,
        'zl	直棱紋':61,
        'scb	圓圈紋':62,
        'shan	山紋':63,
        'xw	弦紋':64,
        'sw	繩紋':65,
        'wa	瓦紋':66,
        'sdfw	獸面紋帶 分尾':67,
        'sc	散螭紋':68,
        'tiger	虎紋':69,
        'fish	魚紋':70,
        'qqw	其他紋飾':71 ,
        'tp	托盘':72,
        'zmc	门窗型炉灶':73,
        'z1	普通炉灶':74,

        'g	蓋':75,
        'qcn	曲尺紐':76,
        'shn	三环钮':77,

        'f	平直扉棱':78,
        'ff	F形扉棱':79,
        'f`	其他扉棱':80,

        've	立耳':81,
        've1	立耳（外撇）':82,
        'be	附耳':83,
        'be1	附耳（彎曲）':84,
        'be2	附耳（S形）':85,
        'ce	环耳':86,

        'bzk	夔龍形扁足':87,
        'bzn	鳥形扁足':88,
        'zhz	錐足':89,
        'zz	柱足':90,
        'tz	蹄足':91,
        'tzdc	蹄足（短粗）':92,
        'tzwc	蹄足（外侈）':93,
        'tzxc	蹄足（细长）':94,
        'zzx	柱足（上粗下细）':95


}


class BronzeWare_Dataset(data.Dataset):
    def __init__(self, img_dir, xml_dir, excel_dir, input_transform=None, train=None, size=None):
        self.root_dir = img_dir
        self.annotations_root=xml_dir
        self.input_transform = input_transform

        ware_img_name_for_3, _, ware_age_for3, _, ware_shape_for3, _, _ = self.load_xlsx_table(excel_dir)

        self.ware_img_name = ware_img_name_for_3
        self.ware_age = ware_age_for3
        self.ware_shape=ware_shape_for3
        self.ware_img = []
        self.ware_xml=[]
        self.input_size = size
        self.encoder = DataEncoder()
        self.train = train

        self.front_img = []
        self.back_img = []

        for img_name in self.ware_img_name:
            png_name = img_name + '.png' 
            xml_name = img_name + '.xml'
            png_name = os.path.join(self.root_dir, png_name)
            xml_name = os.path.join(self.annotations_root, xml_name)
            self.ware_img.append(png_name)
            self.ware_xml.append(xml_name)

    
    
    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xml_path=self.ware_xml[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        xml_data = self.parse_xml_to_dict(xml)["annotation"]


        labels_level1 = []
        labels_level2 = []
        iscrowd = []
        attributes = [0]*96
        shape_label = []


        level_1 = dating_tree_dict[str(self.ware_age[idx])][0]-1
        level_2 = dating_tree_dict[str(self.ware_age[idx])][1]-1
        labels_level1.append(level_1)
        labels_level2.append(level_2)
        shape_label.append(float(self.ware_shape[idx]))
        for obj in xml_data["object"]:
            if obj["name"] in complete_attribute:
                attribute_id = complete_attribute[obj["name"]]
                attributes[attribute_id] = 1
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)
        
        image = Image.open(self.ware_img[idx]).convert('RGB')



        labels_level1 = torch.from_numpy(np.asarray(labels_level1).astype('int64'))
        labels_level2 = torch.from_numpy(np.asarray(labels_level2).astype('int64'))
        shape_label = torch.from_numpy(np.asarray(shape_label).astype('int64'))
        attributes = torch.from_numpy(np.asarray(attributes).astype('int64'))
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["image_id"] = image_id
        target["labels_level1"] = labels_level1
        target["labels_level2"] = labels_level2
        target["attributes"] = attributes
        target["shape_label"] = shape_label

        if self.input_transform is not None:
            image = self.input_transform(image)

        


        return image, None, None, attributes, target["labels_level2"][0], target["shape_label"][0]


    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.ware_xml[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def load_xlsx_table(self, file_path):
        age_table = pd.read_excel(file_path, engine='openpyxl')

        ware_id = np.asarray(age_table.iloc[:, 1],dtype=np.str)
        ware_name = np.asarray(age_table.iloc[:, 2])
        ware_age = np.asarray(age_table.iloc[:, 3])
        ware_book = np.asarray(age_table.iloc[:, 4])
        ware_shape = np.asarray(age_table.iloc[:, 5])
        now_location = np.asarray(age_table.iloc[:, 6])
        out_location = np.asarray(age_table.iloc[:, 7])

        return ware_id, ware_name, ware_age, ware_book, ware_shape, now_location, out_location



    def parse_xml_to_dict(self, xml):
        """
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result: 
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):

        xml_path = self.ware_xml[idx]
        with open(xml_path, encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])

        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            a = float(self.ware_age[idx])
            a = int(a)
            labels.append(a)
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.from_numpy(np.asarray(labels).astype('int64'))
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        level1_label = [x[3] for x in batch]
        level2_label = [x[4] for x in batch]
        shape_label = [x[5] for x in batch]
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        for i in range(num_imgs):
            inputs[i] = imgs[i]
        return inputs, None, None, torch.stack(level1_label), torch.stack(level2_label), torch.stack(shape_label)






class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (tensor) model input size of (w,h).

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                        where #anchors = fmw * fmh * #anchors_per_cell
        '''
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w,fm_h) + 0.5  
            xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
            box = torch.cat([xy,wh], 3)  
            boxes.append(box.view(-1,4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:,2:]
        loc_wh = torch.log(boxes[:,2:]/anchor_boxes[:,2:])
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)  
        cls_targets[ignore] = -1  
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.

        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 4].
          cls_preds: (tensor) predicted class labels, sized [#anchors, #classes].
          input_size: (int/tuple) model input size of (w,h).

        Returns:
          boxes: (tensor) decode box locations, sized [#obj,4].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''
        CLS_THRESH = 0.5
        NMS_THRESH = 0.5

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:,:2]
        loc_wh = loc_preds[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]
        boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  

        score, labels = cls_preds.sigmoid().max(1)          
        ids = score > CLS_THRESH
        ids = ids.nonzero().squeeze()             
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
        return boxes[ids][keep], labels[ids][keep]



def get_mean_and_std(dataset, max_load=10000):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i in range(N):
        print(i)
        im,_,_ = dataset.load(1)
        for j in range(3):
            mean[j] += im[:,j,:,:].mean()
            std[j] += im[:,j,:,:].std()
    mean.div_(N)
    std.div_(N)
    return mean, std

def mask_select(input, mask, dim=0):
    '''Select tensor rows/cols using a mask tensor.

    Args:
      input: (tensor) input tensor, sized [N,M].
      mask: (tensor) mask tensor, sized [N,] or [M,].
      dim: (tensor) mask dim.

    Returns:
      (tensor) selected rows/cols.

    Example:
    >>> a = torch.randn(4,2)
    >>> a
    -0.3462 -0.6930
     0.4560 -0.7459
    -0.1289 -0.9955
     1.7454  1.9787
    [torch.FloatTensor of size 4x2]
    >>> i = a[:,0] > 0
    >>> i
    0
    1
    0
    1
    [torch.ByteTensor of size 4]
    >>> masked_select(a, i, 0)
    0.4560 -0.7459
    1.7454  1.9787
    [torch.FloatTensor of size 2x2]
    '''
    index = mask.nonzero().squeeze(1)
    return input.index_select(dim, index)

def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]

    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)

def change_box_order(boxes, order):
    '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).

    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].
    '''
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a+1], 1)
    return torch.cat([a-b/2,a+b/2], 1)

def box_iou(box1, box2, order='xyxy'):
    '''Compute the intersection over union of two set of boxes.

    The default box order is (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
      order: (str) box order, either 'xyxy' or 'xywh'.

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    '''
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  
    rb = torch.min(box1[:,None,2:], box2[:,2:])  

    wh = (rb-lt+1).clamp(min=0)     
    inter = wh[:,:,0] * wh[:,:,1]  

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  
    iou = inter / (area1[:,None] + area2 - inter)
    return iou

def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) bbox scores, sized [N,].
      threshold: (float) overlap threshold.
      mode: (str) 'union' or 'min'.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1+1) * (y2-y1+1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def softmax(x):
    '''Softmax along a specific dimension.

    Args:
      x: (tensor) input tensor, sized [N,D].

    Returns:
      (tensor) softmaxed tensor, sized [N,D].
    '''
    xmax, _ = x.max(1)
    x_shift = x - xmax.view(-1,1)
    x_exp = x_shift.exp()
    return x_exp / x_exp.sum(1).view(-1,1)

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  
    return y[labels]            

def msr_init(net):
    '''Initialize layer parameters.'''
    for layer in net:
        if type(layer) == nn.Conv2d:
            n = layer.kernel_size[0]*layer.kernel_size[1]*layer.out_channels
            layer.weight.data.normal_(0, math.sqrt(2./n))
            layer.bias.data.zero_()
        elif type(layer) == nn.BatchNorm2d:
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif type(layer) == nn.Linear:
            layer.bias.data.zero_()


term_width = 143
TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time() 

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')


    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


trees_bronze = [
    [1, 1],
    [2, 1],
    [3, 2],
    [4, 2],
    [5, 2],
    [6, 3],
    [7, 3],
    [8, 3],
    [9, 4],
    [10, 4],
    [11, 4],
]
def get_order_family_target(targets, device, shape_labels):

    order_target_list = []
    target_list_sig = []

    shape_targets_list = []

    for i in range(targets.size(0)):
        if targets[i] < 4:
            order_target_list.append(int(targets[i]))
        elif targets[i] > 3:
            order_target_list.append(trees_bronze[targets[i]][1]-1)


        target_list_sig.append(int(targets[i])+4)
        shape_targets_list.append(int(shape_labels[i])+15)

    order_target_list = torch.from_numpy(np.array(order_target_list)).to(device)
    target_list_sig = torch.from_numpy(np.array(target_list_sig)).to(device)
    shape_targets_list = torch.from_numpy(np.array(shape_targets_list)).to(device)

    return order_target_list, target_list_sig, shape_targets_list

