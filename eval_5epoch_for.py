"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import sys
import os
import os.path as osp
import time
import argparse
import numpy as np
import pickle
import cv2
import shutil
from ssd import build_ssd
from data import *

labelmap = SIXray_CLASSES


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "a1")


EPOCH = 5
GPUID = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUID

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default="", type=str,
                    help='Trained state_dict file path to open')
parser.add_argument(  # '--save_folder', default='/media/dsg3/husheng/eval/', type=str,
    '--save_folder',
    default="", type=str,
    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.2, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--SIXray_root', default=SIXray_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--imagesetfile',
                    # default='/media/dsg3/datasets/SIXray/dataset-test.txt', type=str,
                    default="/media/trs2/Xray20190723/train_test_txt/battery_sub/sub_test_core_coreless.txt", type=str,
                    help='imageset file path to open')
parser.add_argument('--my_local', default=False, type=str2bool,
                    help='for my local test')

args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        # print("WARNING: It looks like you have a CUDA device, but aren't using \
        #         CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.SIXray_root, 'Annotation', '%s.txt')
imgpath = os.path.join(args.SIXray_root, 'Image', '%s.jpg')

# imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')

YEAR = '2007'

devkit_path = args.save_folder
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename,imgpath):
    """ Parse a PASCAL VOC xml file """
    # tree = ET.parse(filename)
    # filename = filename[:-3] + 'txt'

    filename = filename.replace('.xml', '.txt')

    #imagename0 = filename.replace('Anno_core_coreless_battery_sub_2000_500', 'cut_Image_core_coreless_battery_sub_2000_500')
    img_fold_name = imgpath.split('/')[-2]
    imagename0 = filename.replace('Annotation', img_fold_name)
    imagename1 = imagename0.replace('.txt', '.jpg')  # jpg form
    imagename2 = imagename0.replace('.txt', '.jpg')
    objects = []
    # 还需要同时打开图像，读入图像大小
    # print(imagename1)
    img = cv2.imread(imagename1)
    if img is None:
        img = cv2.imread(imagename2)
    height, width, channels = img.shape
    with open(filename, "r", encoding='utf-8') as f1:
        dataread = f1.readlines()
        for annotation in dataread:
            obj_struct = {}
            temp = annotation.split()
            name = temp[1]
            if name != '带电芯充电宝' and name != '不带电芯充电宝':
                continue
            xmin = int(temp[2])
            # 只读取V视角的
            if int(xmin) > width:
                continue
            if xmin < 0:
                xmin = 1
            ymin = int(temp[3])
            if ymin < 0:
                ymin = 1
            xmax = int(temp[4])
            if xmax > width:
                xmax = width - 1
            ymax = int(temp[5])
            if ymax > height:
                ymax = height - 1
            ##name
            obj_struct['name'] = name
            obj_struct['pose'] = 'Unspecified'
            obj_struct['truncated'] = 0
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [float(xmin) - 1,
                                  float(ymin) - 1,
                                  float(xmax) - 1,
                                  float(ymax) - 1]
            objects.append(obj_struct)

    '''
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text.lower().strip()
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [float(bbox.find('xmin').text) - 1,
                              float(bbox.find('ymin').text) - 1,
                              float(bbox.find('xmax').text) - 1,
                              float(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)
    '''
    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        # print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects a1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=False):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        rec, prec, ap = voc_eval(
            filename, annopath,imgpath, args.imagesetfile, cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric, my_local=args.my_local)
        aps += [ap]
        # print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print("EPOCH, {:d}, mAP, {:.4f}, core_AP, {:.4f}, coreless_AP, {:.4f}".format(EPOCH, np.mean(aps), aps[0], aps[1]))
    # print('Mean AP = {:.4f}'.format(np.mean(aps)))
    '''
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')
    '''


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
			       imgpath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True,
             my_local=False):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
       annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    # with open(imagesetfile, 'r') as f:
    #     lines = f.readlines()
    # imagenames = [x.strip() for x in lines]


    imagenames = []
    if my_local:
        listdir = os.listdir(osp.join('%s' % args.SIXray_root, 'Annotation'))
        testdir = listdir[0:6000:5]
        testdir = ['coreless_battery00004025.txt', 'coreless_battery00004028.txt', 'coreless_battery00004041.txt',
             'coreless_battery00004049.txt', 'coreless_battery00004057.txt', 'coreless_battery00004053.txt',
             'coreless_battery00004083.txt', 'coreless_battery00004088.txt', 'coreless_battery00004106.txt',
             'coreless_battery00004122.txt', 'coreless_battery00004114.txt', 'coreless_battery00004128.txt',
             'coreless_battery00004133.txt', 'coreless_battery00004144.txt', 'coreless_battery00004172.txt',
             'coreless_battery00004170.txt', 'coreless_battery00004179.txt', 'coreless_battery00004210.txt',
             'coreless_battery00004188.txt', 'coreless_battery00004219.txt', 'coreless_battery00004234.txt',
             'coreless_battery00004229.txt', 'coreless_battery00004242.txt', 'coreless_battery00004245.txt',
             'coreless_battery00004255.txt', 'coreless_battery00004278.txt', 'coreless_battery00004289.txt',
             'coreless_battery00004285.txt', 'coreless_battery00004127.txt', 'coreless_battery00004182.txt',
             'coreless_battery00004307.txt', 'coreless_battery00004314.txt', 'coreless_battery00004327.txt',
             'coreless_battery00004346.txt', 'coreless_battery00004336.txt', 'coreless_battery00004351.txt',
             'coreless_battery00004366.txt', 'coreless_battery00004378.txt', 'coreless_battery00004383.txt',
             'coreless_battery00004393.txt', 'coreless_battery00004411.txt', 'coreless_battery00004421.txt',
             'coreless_battery00004428.txt', 'coreless_battery00004423.txt', 'coreless_battery00004461.txt',
             'coreless_battery00004454.txt', 'coreless_battery00004466.txt', 'coreless_battery00004477.txt',
             'coreless_battery00004514.txt', 'coreless_battery00004484.txt', 'coreless_battery00004528.txt',
             'coreless_battery00004533.txt', 'coreless_battery00004534.txt', 'coreless_battery00004570.txt',
             'coreless_battery00004566.txt', 'coreless_battery00004579.txt', 'coreless_battery00004600.txt',
             'coreless_battery00004594.txt', 'coreless_battery00004609.txt', 'coreless_battery00004619.txt',
             'coreless_battery00004621.txt', 'coreless_battery00004633.txt', 'coreless_battery00004636.txt',
             'coreless_battery00004661.txt', 'coreless_battery00004667.txt', 'coreless_battery00004674.txt',
             'coreless_battery00004377.txt', 'coreless_battery00004690.txt', 'coreless_battery00004551.txt',
             'coreless_battery00004698.txt', 'coreless_battery00004734.txt', 'coreless_battery00004712.txt',
             'coreless_battery00004738.txt', 'coreless_battery00004752.txt', 'coreless_battery00004740.txt',
             'coreless_battery00004778.txt', 'coreless_battery00004766.txt', 'coreless_battery00004783.txt',
             'coreless_battery00004788.txt', 'coreless_battery00004796.txt', 'coreless_battery00004820.txt',
             'coreless_battery00004809.txt', 'coreless_battery00004843.txt', 'coreless_battery00004861.txt',
             'coreless_battery00004880.txt', 'coreless_battery00004911.txt', 'coreless_battery00004903.txt',
             'coreless_battery00004921.txt', 'coreless_battery00004938.txt', 'coreless_battery00004940.txt',
             'coreless_battery00004956.txt', 'coreless_battery00004968.txt', 'coreless_battery00004982.txt',
             'coreless_battery00005006.txt', 'coreless_battery00005000.txt', 'coreless_battery00005011.txt',
             'coreless_battery00005040.txt', 'coreless_battery00005024.txt', 'coreless_battery00005061.txt',
             'coreless_battery00005071.txt', 'coreless_battery00005083.txt', 'coreless_battery00005096.txt',
             'coreless_battery00005104.txt', 'coreless_battery00005114.txt', 'coreless_battery00004793.txt',
             'coreless_battery00004751.txt', 'coreless_battery00004942.txt', 'coreless_battery00005129.txt',
             'coreless_battery00005135.txt', 'coreless_battery00005165.txt', 'coreless_battery00005157.txt',
             'coreless_battery00005173.txt', 'coreless_battery00005200.txt', 'coreless_battery00005183.txt',
             'coreless_battery00005210.txt', 'coreless_battery00005218.txt', 'coreless_battery00005221.txt',
             'coreless_battery00005233.txt', 'coreless_battery00005252.txt', 'coreless_battery00005244.txt',
             'coreless_battery00005260.txt', 'coreless_battery00005289.txt', 'coreless_battery00005283.txt',
             'coreless_battery00005297.txt', 'coreless_battery00005306.txt', 'coreless_battery00005301.txt',
             'coreless_battery00005328.txt', 'coreless_battery00005335.txt', 'coreless_battery00005346.txt',
             'coreless_battery00005364.txt', 'coreless_battery00005355.txt', 'coreless_battery00005383.txt',
             'coreless_battery00005406.txt', 'coreless_battery00005393.txt', 'coreless_battery00005411.txt',
             'coreless_battery00005419.txt', 'coreless_battery00005426.txt', 'coreless_battery00005447.txt',
             'coreless_battery00005431.txt', 'coreless_battery00005458.txt', 'coreless_battery00005460.txt',
             'coreless_battery00005485.txt', 'coreless_battery00005521.txt', 'coreless_battery00005498.txt',
             'coreless_battery00005189.txt', 'coreless_battery00005373.txt', 'coreless_battery00005483.txt',
             'coreless_battery00005542.txt', 'coreless_battery00005532.txt', 'coreless_battery00005550.txt',
             'coreless_battery00005556.txt', 'coreless_battery00005563.txt', 'coreless_battery00005577.txt',
             'coreless_battery00005601.txt', 'coreless_battery00005595.txt', 'coreless_battery00005608.txt',
             'coreless_battery00005614.txt', 'coreless_battery00005635.txt', 'coreless_battery00005638.txt',
             'coreless_battery00005645.txt', 'coreless_battery00005664.txt', 'coreless_battery00005687.txt',
             'coreless_battery00005671.txt', 'coreless_battery00005693.txt', 'coreless_battery00005696.txt',
             'coreless_battery00005704.txt', 'coreless_battery00005726.txt', 'coreless_battery00005739.txt',
             'coreless_battery00005734.txt', 'coreless_battery00005760.txt', 'coreless_battery00005766.txt',
             'coreless_battery00005778.txt', 'coreless_battery00005793.txt', 'coreless_battery00005801.txt',
             'coreless_battery00005825.txt', 'coreless_battery00005816.txt', 'coreless_battery00005832.txt',
             'coreless_battery00005838.txt', 'coreless_battery00005843.txt', 'coreless_battery00005854.txt',
             'coreless_battery00005884.txt', 'coreless_battery00005876.txt', 'coreless_battery00005888.txt',
             'coreless_battery00005546.txt', 'coreless_battery00005646.txt', 'coreless_battery00005841.txt',
             'coreless_battery00005920.txt', 'core_battery00002538.txt', 'coreless_battery00005901.txt',
             'core_battery00005134.txt', 'coreless_battery00005914.txt', 'coreless_battery00005916.txt',
             'coreless_battery00005935.txt', 'coreless_battery00005941.txt', 'coreless_battery00005955.txt',
             'coreless_battery00005963.txt', 'coreless_battery00005969.txt', 'coreless_battery00005978.txt',
             'coreless_battery00005986.txt', 'coreless_battery00005998.txt', 'coreless_battery00002052.txt',
             'coreless_battery00002067.txt', 'coreless_battery00002092.txt', 'coreless_battery00002077.txt',
             'coreless_battery00002103.txt', 'coreless_battery00002108.txt', 'coreless_battery00002109.txt',
             'coreless_battery00002151.txt', 'coreless_battery00002139.txt', 'coreless_battery00002160.txt',
             'coreless_battery00002179.txt', 'coreless_battery00002166.txt', 'coreless_battery00002189.txt',
             'coreless_battery00002193.txt', 'coreless_battery00002209.txt', 'coreless_battery00002229.txt',
             'coreless_battery00002241.txt', 'coreless_battery00002258.txt', 'coreless_battery00002290.txt',
             'coreless_battery00002272.txt', 'coreless_battery00002297.txt', 'coreless_battery00002310.txt',
             'coreless_battery00002331.txt', 'coreless_battery00002332.txt', 'coreless_battery00002345.txt',
             'coreless_battery00002352.txt', 'coreless_battery00002355.txt', 'coreless_battery00002370.txt',
             'coreless_battery00002389.txt', 'coreless_battery00002393.txt', 'coreless_battery00002381.txt',
             'coreless_battery00002404.txt', 'coreless_battery00002085.txt', 'coreless_battery00002249.txt',
             'coreless_battery00002416.txt', 'coreless_battery00002409.txt', 'coreless_battery00002422.txt',
             'coreless_battery00002446.txt', 'coreless_battery00002430.txt', 'coreless_battery00002453.txt',
             'coreless_battery00002480.txt', 'coreless_battery00002461.txt', 'coreless_battery00002488.txt',
             'coreless_battery00002496.txt', 'coreless_battery00002509.txt', 'coreless_battery00002521.txt',
             'coreless_battery00002527.txt', 'coreless_battery00002539.txt', 'coreless_battery00002585.txt',
             'coreless_battery00002572.txt', 'coreless_battery00002592.txt', 'coreless_battery00002609.txt',
             'coreless_battery00002601.txt', 'coreless_battery00002620.txt', 'coreless_battery00002631.txt',
             'coreless_battery00002625.txt', 'coreless_battery00002644.txt', 'coreless_battery00002651.txt',
             'coreless_battery00002661.txt', 'coreless_battery00002675.txt', 'coreless_battery00002681.txt',
             'coreless_battery00002679.txt', 'coreless_battery00002701.txt', 'coreless_battery00002720.txt',
             'coreless_battery00002708.txt', 'coreless_battery00002730.txt', 'coreless_battery00002748.txt',
             'coreless_battery00002759.txt', 'coreless_battery00002781.txt', 'coreless_battery00002769.txt',
             'coreless_battery00002790.txt', 'coreless_battery00002634.txt', 'coreless_battery00002610.txt',
             'coreless_battery00002794.txt', 'coreless_battery00002806.txt', 'coreless_battery00002812.txt',
             'coreless_battery00002833.txt', 'coreless_battery00002841.txt', 'coreless_battery00002849.txt',
             'coreless_battery00002870.txt', 'coreless_battery00002860.txt', 'coreless_battery00002872.txt',
             'coreless_battery00002886.txt', 'coreless_battery00002889.txt', 'coreless_battery00002904.txt',
             'coreless_battery00002912.txt', 'coreless_battery00002928.txt', 'coreless_battery00002936.txt',
             'coreless_battery00002957.txt', 'coreless_battery00002953.txt', 'coreless_battery00002967.txt',
             'coreless_battery00002981.txt', 'coreless_battery00002984.txt', 'coreless_battery00003002.txt',
             'coreless_battery00003010.txt', 'coreless_battery00003020.txt', 'coreless_battery00003039.txt',
             'coreless_battery00003030.txt', 'coreless_battery00003046.txt', 'coreless_battery00003059.txt',
             'coreless_battery00003064.txt', 'coreless_battery00003080.txt', 'coreless_battery00003106.txt',
             'coreless_battery00003097.txt', 'coreless_battery00003112.txt', 'coreless_battery00003120.txt',
             'coreless_battery00003129.txt', 'coreless_battery00003133.txt', 'coreless_battery00003146.txt',
             'coreless_battery00003149.txt', 'coreless_battery00002873.txt', 'coreless_battery00003011.txt',
             'coreless_battery00002952.txt', 'coreless_battery00003165.txt', 'coreless_battery00003180.txt',
             'coreless_battery00003190.txt', 'coreless_battery00003208.txt', 'coreless_battery00003203.txt',
             'coreless_battery00003233.txt', 'coreless_battery00003262.txt', 'coreless_battery00003249.txt',
             'coreless_battery00003268.txt', 'coreless_battery00003295.txt', 'coreless_battery00003289.txt',
             'coreless_battery00003303.txt', 'coreless_battery00003313.txt', 'coreless_battery00003316.txt',
             'coreless_battery00003331.txt', 'coreless_battery00003360.txt', 'coreless_battery00003335.txt',
             'coreless_battery00003371.txt', 'coreless_battery00003389.txt', 'coreless_battery00003384.txt',
             'coreless_battery00003415.txt', 'coreless_battery00003406.txt', 'coreless_battery00003421.txt',
             'coreless_battery00003438.txt', 'coreless_battery00003440.txt', 'coreless_battery00003465.txt',
             'coreless_battery00003452.txt', 'coreless_battery00003474.txt', 'coreless_battery00003500.txt',
             'coreless_battery00003497.txt', 'coreless_battery00003505.txt', 'coreless_battery00003523.txt',
             'coreless_battery00003517.txt', 'coreless_battery00003530.txt', 'coreless_battery00003555.txt',
             'coreless_battery00003545.txt', 'coreless_battery00003259.txt', 'coreless_battery00003338.txt',
             'coreless_battery00003529.txt', 'coreless_battery00003579.txt', 'coreless_battery00003574.txt',
             'coreless_battery00003588.txt', 'coreless_battery00003606.txt', 'coreless_battery00003594.txt',
             'coreless_battery00003619.txt', 'coreless_battery00003644.txt', 'coreless_battery00003633.txt',
             'coreless_battery00003654.txt', 'coreless_battery00003659.txt', 'coreless_battery00003660.txt',
             'coreless_battery00003690.txt', 'coreless_battery00003699.txt', 'coreless_battery00003683.txt',
             'coreless_battery00003725.txt', 'coreless_battery00003711.txt', 'coreless_battery00003737.txt',
             'coreless_battery00003746.txt', 'coreless_battery00003743.txt', 'coreless_battery00003760.txt',
             'coreless_battery00003767.txt', 'coreless_battery00003792.txt', 'coreless_battery00003816.txt',
             'coreless_battery00003808.txt', 'coreless_battery00003822.txt', 'coreless_battery00003836.txt',
             'coreless_battery00003829.txt', 'coreless_battery00003845.txt', 'coreless_battery00003860.txt',
             'coreless_battery00003867.txt', 'coreless_battery00003884.txt', 'coreless_battery00003873.txt',
             'coreless_battery00003917.txt', 'coreless_battery00003897.txt', 'coreless_battery00003927.txt',
             'coreless_battery00003936.txt', 'coreless_battery00003642.txt', 'coreless_battery00003719.txt',
             'coreless_battery00003903.txt', 'coreless_battery00003954.txt', 'coreless_battery00003956.txt',
             'coreless_battery00003968.txt', 'coreless_battery00003973.txt', 'coreless_battery00003981.txt',
             'coreless_battery00003992.txt', 'coreless_battery00004000.txt', 'coreless_battery00004015.txt',
             'coreless_battery00004009.txt', 'core_battery00000004.txt', 'core_battery00000010.txt',
             'core_battery00000021.txt', 'coreless_battery00000043.txt', 'coreless_battery00000071.txt',
             'coreless_battery00000062.txt', 'coreless_battery00000081.txt', 'coreless_battery00000097.txt',
             'coreless_battery00000089.txt', 'coreless_battery00000128.txt', 'coreless_battery00000119.txt',
             'coreless_battery00000134.txt', 'coreless_battery00000167.txt', 'coreless_battery00000155.txt',
             'coreless_battery00000175.txt', 'coreless_battery00000183.txt', 'coreless_battery00000182.txt',
             'coreless_battery00000208.txt', 'coreless_battery00000216.txt', 'coreless_battery00000230.txt',
             'coreless_battery00000237.txt', 'coreless_battery00000245.txt', 'coreless_battery00000264.txt',
             'coreless_battery00000273.txt', 'coreless_battery00000279.txt', 'coreless_battery00000286.txt',
             'coreless_battery00000288.txt', 'coreless_battery00000305.txt', 'coreless_battery00000334.txt',
             'coreless_battery00000321.txt', 'coreless_battery00000341.txt', 'coreless_battery00000355.txt',
             'coreless_battery00000359.txt', 'coreless_battery00000383.txt', 'coreless_battery00000375.txt',
             'coreless_battery00000392.txt', 'coreless_battery00000403.txt', 'coreless_battery00000406.txt',
             'coreless_battery00000423.txt', 'coreless_battery00000428.txt', 'core_battery00000135.txt',
             'core_battery00000250.txt', 'core_battery00000384.txt', 'coreless_battery00000450.txt',
             'coreless_battery00000474.txt', 'coreless_battery00000468.txt', 'coreless_battery00000479.txt',
             'coreless_battery00000500.txt', 'coreless_battery00000485.txt', 'coreless_battery00000524.txt',
             'coreless_battery00000530.txt', 'coreless_battery00000543.txt', 'coreless_battery00000551.txt',
             'coreless_battery00000569.txt', 'coreless_battery00000565.txt', 'coreless_battery00000582.txt',
             'coreless_battery00000592.txt', 'coreless_battery00000599.txt', 'coreless_battery00000608.txt',
             'coreless_battery00000618.txt', 'coreless_battery00000615.txt', 'coreless_battery00000657.txt',
             'coreless_battery00000655.txt', 'coreless_battery00000670.txt', 'coreless_battery00000678.txt',
             'coreless_battery00000691.txt', 'coreless_battery00000726.txt', 'coreless_battery00000720.txt',
             'coreless_battery00000735.txt', 'coreless_battery00000763.txt', 'coreless_battery00000761.txt',
             'coreless_battery00000781.txt', 'coreless_battery00000801.txt', 'coreless_battery00000791.txt',
             'coreless_battery00000811.txt', 'coreless_battery00000466.txt', 'coreless_battery00000818.txt',
             'coreless_battery00000625.txt', 'coreless_battery00000844.txt', 'coreless_battery00000836.txt',
             'coreless_battery00000851.txt', 'coreless_battery00000867.txt', 'coreless_battery00000877.txt',
             'coreless_battery00000883.txt', 'coreless_battery00000895.txt', 'coreless_battery00000904.txt',
             'coreless_battery00000921.txt', 'coreless_battery00000936.txt', 'coreless_battery00000938.txt',
             'coreless_battery00000956.txt', 'coreless_battery00000974.txt', 'coreless_battery00000964.txt',
             'coreless_battery00000989.txt', 'coreless_battery00001000.txt', 'coreless_battery00000999.txt',
             'coreless_battery00001023.txt', 'coreless_battery00001053.txt', 'coreless_battery00001043.txt',
             'coreless_battery00001059.txt', 'coreless_battery00001074.txt', 'coreless_battery00001067.txt',
             'coreless_battery00001081.txt', 'coreless_battery00001090.txt', 'coreless_battery00001092.txt',
             'coreless_battery00001108.txt', 'coreless_battery00001122.txt', 'coreless_battery00001134.txt',
             'coreless_battery00001147.txt', 'coreless_battery00001167.txt', 'coreless_battery00001150.txt',
             'coreless_battery00001174.txt', 'coreless_battery00001186.txt', 'coreless_battery00001181.txt',
             'coreless_battery00001204.txt', 'coreless_battery00001213.txt', 'coreless_battery00001208.txt',
             'coreless_battery00001064.txt', 'coreless_battery00001223.txt', 'coreless_battery00001114.txt',
             'coreless_battery00001242.txt', 'coreless_battery00001245.txt', 'coreless_battery00001279.txt',
             'coreless_battery00001301.txt', 'coreless_battery00001288.txt', 'coreless_battery00001310.txt',
             'coreless_battery00001334.txt', 'coreless_battery00001323.txt', 'coreless_battery00001341.txt',
             'coreless_battery00001343.txt', 'coreless_battery00001357.txt', 'coreless_battery00001361.txt',
             'coreless_battery00001376.txt', 'coreless_battery00001389.txt', 'coreless_battery00001392.txt',
             'coreless_battery00001406.txt', 'coreless_battery00001412.txt', 'coreless_battery00001429.txt',
             'coreless_battery00001449.txt', 'coreless_battery00001455.txt', 'coreless_battery00001476.txt',
             'coreless_battery00001480.txt', 'coreless_battery00001495.txt', 'coreless_battery00001494.txt',
             'coreless_battery00001510.txt', 'coreless_battery00001532.txt', 'coreless_battery00001520.txt',
             'coreless_battery00001541.txt', 'coreless_battery00001545.txt', 'coreless_battery00001559.txt',
             'coreless_battery00001571.txt', 'coreless_battery00001576.txt', 'coreless_battery00001592.txt',
             'coreless_battery00001606.txt', 'coreless_battery00001619.txt', 'coreless_battery00001487.txt',
             'coreless_battery00001457.txt', 'coreless_battery00001591.txt', 'coreless_battery00001637.txt',
             'coreless_battery00001636.txt', 'coreless_battery00001667.txt', 'coreless_battery00001664.txt',
             'coreless_battery00001674.txt', 'coreless_battery00001695.txt', 'coreless_battery00001684.txt',
             'coreless_battery00001707.txt', 'coreless_battery00001722.txt', 'coreless_battery00001718.txt',
             'coreless_battery00001737.txt', 'coreless_battery00001750.txt', 'coreless_battery00001757.txt',
             'coreless_battery00001780.txt', 'coreless_battery00001777.txt', 'coreless_battery00001787.txt',
             'coreless_battery00001807.txt', 'coreless_battery00001801.txt', 'coreless_battery00001817.txt',
             'coreless_battery00001849.txt', 'coreless_battery00001845.txt', 'coreless_battery00001859.txt',
             'coreless_battery00001882.txt', 'coreless_battery00001890.txt', 'coreless_battery00001883.txt',
             'coreless_battery00001912.txt', 'coreless_battery00001910.txt', 'coreless_battery00001922.txt',
             'coreless_battery00001932.txt', 'coreless_battery00001925.txt', 'coreless_battery00001962.txt',
             'coreless_battery00001983.txt', 'coreless_battery00001976.txt', 'coreless_battery00001993.txt',
             'coreless_battery00002005.txt', 'coreless_battery00002012.txt', 'coreless_battery00001676.txt',
             'coreless_battery00001992.txt', 'coreless_battery00001878.txt', 'coreless_battery00002031.txt',
             'coreless_battery00002027.txt', 'coreless_battery00002038.txt', 'coreless_battery00002060.txt',
             'core_battery00004092.txt', 'core_battery00004098.txt', 'core_battery00004102.txt', 'core_battery00004131.txt',
             'core_battery00004143.txt', 'core_battery00004146.txt', 'core_battery00004162.txt', 'core_battery00004168.txt',
             'core_battery00004192.txt', 'core_battery00004173.txt', 'core_battery00004201.txt', 'core_battery00004207.txt',
             'core_battery00004220.txt', 'core_battery00004256.txt', 'core_battery00004253.txt', 'core_battery00004261.txt',
             'core_battery00004265.txt', 'core_battery00004271.txt', 'core_battery00004288.txt', 'core_battery00004296.txt',
             'core_battery00004295.txt', 'core_battery00004322.txt', 'core_battery00004328.txt', 'core_battery00004329.txt',
             'core_battery00004362.txt', 'core_battery00004118.txt', 'core_battery00003926.txt', 'core_battery00004273.txt',
             'core_battery00004375.txt', 'core_battery00004387.txt', 'core_battery00004396.txt', 'core_battery00004406.txt',
             'core_battery00004414.txt', 'core_battery00004432.txt', 'core_battery00004445.txt', 'core_battery00004439.txt',
             'core_battery00004465.txt', 'core_battery00004488.txt', 'core_battery00004481.txt', 'core_battery00004496.txt',
             'core_battery00004502.txt', 'core_battery00004509.txt', 'core_battery00004527.txt', 'core_battery00004525.txt',
             'core_battery00004524.txt', 'core_battery00004564.txt', 'core_battery00004559.txt', 'core_battery00004573.txt',
             'core_battery00004598.txt', 'core_battery00004586.txt', 'core_battery00004608.txt', 'core_battery00004620.txt',
             'core_battery00004644.txt', 'core_battery00004649.txt', 'core_battery00004654.txt', 'core_battery00004666.txt',
             'core_battery00004688.txt', 'core_battery00004680.txt', 'core_battery00004693.txt', 'core_battery00004706.txt',
             'core_battery00004721.txt', 'core_battery00004710.txt', 'core_battery00004745.txt', 'core_battery00004742.txt',
             'core_battery00004756.txt', 'core_battery00004764.txt', 'core_battery00004782.txt', 'core_battery00004368.txt',
             'core_battery00004540.txt', 'core_battery00004720.txt', 'core_battery00004816.txt', 'core_battery00004824.txt',
             'core_battery00004807.txt', 'core_battery00004830.txt', 'core_battery00004845.txt', 'core_battery00004836.txt',
             'core_battery00004851.txt', 'core_battery00004857.txt', 'core_battery00004864.txt', 'core_battery00004884.txt',
             'core_battery00004866.txt', 'core_battery00004893.txt', 'core_battery00004908.txt', 'core_battery00004901.txt',
             'core_battery00004919.txt', 'core_battery00004935.txt', 'core_battery00004941.txt', 'core_battery00004962.txt',
             'core_battery00004957.txt', 'core_battery00004966.txt', 'core_battery00004972.txt', 'core_battery00004975.txt',
             'core_battery00004994.txt', 'core_battery00005019.txt', 'core_battery00005013.txt', 'core_battery00005026.txt',
             'core_battery00005037.txt', 'core_battery00005031.txt', 'core_battery00005048.txt', 'core_battery00005060.txt',
             'core_battery00005076.txt', 'core_battery00005079.txt', 'core_battery00005105.txt', 'core_battery00005090.txt',
             'core_battery00005112.txt', 'core_battery00005142.txt', 'core_battery00005123.txt', 'core_battery00005147.txt',
             'core_battery00005150.txt', 'core_battery00004860.txt', 'core_battery00004964.txt', 'core_battery00005159.txt',
             'core_battery00005170.txt', 'core_battery00005182.txt', 'core_battery00005196.txt', 'core_battery00005201.txt',
             'core_battery00005229.txt', 'core_battery00005212.txt', 'core_battery00005241.txt', 'core_battery00005247.txt',
             'core_battery00005267.txt', 'core_battery00005274.txt', 'core_battery00005281.txt', 'core_battery00005282.txt',
             'core_battery00005309.txt', 'core_battery00005320.txt', 'core_battery00005329.txt', 'core_battery00005362.txt',
             'core_battery00005367.txt', 'core_battery00005341.txt', 'core_battery00005374.txt', 'core_battery00005384.txt',
             'core_battery00005390.txt', 'core_battery00005402.txt', 'core_battery00005427.txt', 'core_battery00005438.txt',
             'core_battery00005444.txt', 'core_battery00005461.txt', 'core_battery00005468.txt', 'core_battery00005471.txt',
             'core_battery00005479.txt', 'core_battery00005495.txt', 'core_battery00005512.txt', 'core_battery00005505.txt',
             'core_battery00005516.txt', 'core_battery00005545.txt', 'core_battery00005540.txt', 'core_battery00005554.txt',
             'core_battery00005586.txt', 'core_battery00005582.txt', 'core_battery00005590.txt', 'core_battery00005257.txt',
             'core_battery00005220.txt', 'core_battery00005594.txt', 'core_battery00005620.txt', 'core_battery00005615.txt',
             'core_battery00005632.txt', 'core_battery00005648.txt', 'core_battery00005653.txt', 'core_battery00005678.txt',
             'core_battery00005670.txt', 'core_battery00005683.txt', 'core_battery00005708.txt', 'core_battery00005697.txt',
             'core_battery00005721.txt', 'core_battery00005729.txt', 'core_battery00005732.txt', 'core_battery00005753.txt',
             'core_battery00005758.txt', 'core_battery00005779.txt', 'core_battery00005790.txt', 'core_battery00005797.txt',
             'core_battery00005806.txt', 'core_battery00005821.txt', 'core_battery00005837.txt', 'core_battery00005827.txt',
             'core_battery00005860.txt', 'core_battery00005882.txt', 'core_battery00005864.txt', 'core_battery00005893.txt',
             'core_battery00005921.txt', 'core_battery00005912.txt', 'core_battery00005929.txt', 'core_battery00005931.txt',
             'core_battery00005944.txt', 'core_battery00005977.txt', 'core_battery00005968.txt', 'core_battery00005993.txt',
             'core_battery00006004.txt', 'core_battery00006008.txt', 'core_battery00006027.txt', 'core_battery00006023.txt',
             'core_battery00005592.txt', 'core_battery00005720.txt', 'core_battery00005813.txt', 'core_battery00006034.txt',
             'core_battery00006041.txt', 'core_battery00006051.txt', 'core_battery00006068.txt', 'core_battery00006060.txt',
             'core_battery00001968.txt', 'core_battery00001984.txt', 'core_battery00002015.txt', 'core_battery00002006.txt',
             'core_battery00002022.txt', 'core_battery00002053.txt', 'core_battery00002054.txt', 'core_battery00002063.txt',
             'core_battery00002080.txt', 'core_battery00002072.txt', 'core_battery00002094.txt', 'core_battery00002110.txt',
             'core_battery00002096.txt', 'core_battery00002134.txt', 'core_battery00002131.txt', 'core_battery00001759.txt',
             'core_battery00001921.txt', 'core_battery00001946.txt', 'core_battery00002149.txt', 'core_battery00002162.txt',
             'core_battery00002175.txt', 'core_battery00002197.txt', 'core_battery00002187.txt', 'core_battery00002208.txt',
             'core_battery00002223.txt', 'core_battery00002224.txt', 'core_battery00002228.txt', 'core_battery00002233.txt',
             'core_battery00002235.txt', 'core_battery00002263.txt', 'core_battery00002259.txt', 'core_battery00002261.txt',
             'core_battery00002281.txt', 'core_battery00002305.txt', 'core_battery00002294.txt', 'core_battery00002322.txt',
             'core_battery00002315.txt', 'core_battery00002330.txt', 'core_battery00002350.txt', 'core_battery00002327.txt',
             'core_battery00002391.txt', 'core_battery00002383.txt', 'core_battery00002397.txt', 'core_battery00002436.txt',
             'core_battery00002414.txt', 'core_battery00002442.txt', 'core_battery00002448.txt', 'core_battery00002464.txt',
             'core_battery00002476.txt', 'core_battery00002492.txt', 'core_battery00002500.txt', 'core_battery00002511.txt',
             'core_battery00002533.txt', 'core_battery00002544.txt', 'core_battery00002517.txt', 'core_battery00002559.txt',
             'core_battery00002553.txt', 'core_battery00002225.txt', 'core_battery00002444.txt', 'core_battery00002301.txt',
             'core_battery00002567.txt', 'core_battery00002583.txt', 'core_battery00002573.txt', 'core_battery00002594.txt',
             'core_battery00002629.txt', 'core_battery00002604.txt', 'core_battery00002637.txt', 'core_battery00002647.txt',
             'core_battery00002665.txt', 'core_battery00002674.txt', 'core_battery00002692.txt', 'core_battery00002707.txt',
             'core_battery00002717.txt', 'core_battery00002740.txt', 'core_battery00002726.txt', 'core_battery00002749.txt',
             'core_battery00002753.txt', 'core_battery00002771.txt', 'core_battery00002797.txt', 'core_battery00002777.txt',
             'core_battery00002801.txt', 'core_battery00002824.txt', 'core_battery00002819.txt', 'core_battery00002825.txt',
             'core_battery00002863.txt', 'core_battery00002840.txt', 'core_battery00002867.txt', 'core_battery00002884.txt',
             'core_battery00002898.txt', 'core_battery00002903.txt', 'core_battery00002913.txt', 'core_battery00002930.txt',
             'core_battery00002937.txt', 'core_battery00002955.txt', 'core_battery00002963.txt', 'core_battery00002973.txt',
             'core_battery00002985.txt', 'core_battery00003008.txt', 'core_battery00003000.txt', 'core_battery00002687.txt',
             'core_battery00002882.txt', 'core_battery00003016.txt', 'core_battery00003045.txt', 'core_battery00003027.txt',
             'core_battery00003057.txt', 'core_battery00003071.txt', 'core_battery00003068.txt', 'core_battery00003104.txt',
             'core_battery00003100.txt', 'core_battery00003119.txt', 'core_battery00003135.txt', 'core_battery00003152.txt',
             'core_battery00003166.txt', 'core_battery00003168.txt', 'core_battery00003185.txt', 'core_battery00003196.txt',
             'core_battery00003199.txt', 'core_battery00003209.txt', 'core_battery00003217.txt', 'core_battery00003227.txt',
             'core_battery00003245.txt', 'core_battery00003243.txt', 'core_battery00003253.txt', 'core_battery00003258.txt',
             'core_battery00003277.txt', 'core_battery00003280.txt', 'core_battery00003310.txt', 'core_battery00003300.txt',
             'core_battery00003318.txt', 'core_battery00003342.txt', 'core_battery00003330.txt', 'core_battery00003348.txt',
             'core_battery00003355.txt', 'core_battery00003361.txt', 'core_battery00003379.txt', 'core_battery00003390.txt',
             'core_battery00003388.txt', 'core_battery00003413.txt', 'core_battery00003419.txt', 'core_battery00003443.txt',
             'core_battery00003214.txt', 'core_battery00003121.txt', 'core_battery00003329.txt', 'core_battery00003464.txt',
             'core_battery00003454.txt', 'core_battery00003471.txt', 'core_battery00003478.txt', 'core_battery00003492.txt',
             'core_battery00003527.txt', 'core_battery00003518.txt', 'core_battery00003535.txt', 'core_battery00003547.txt',
             'core_battery00003542.txt', 'core_battery00003567.txt', 'core_battery00003580.txt', 'core_battery00003589.txt',
             'core_battery00003615.txt', 'core_battery00003611.txt', 'core_battery00003622.txt', 'core_battery00003634.txt',
             'core_battery00003649.txt', 'core_battery00003661.txt', 'core_battery00003684.txt', 'core_battery00003676.txt',
             'core_battery00003696.txt', 'core_battery00003718.txt', 'core_battery00003700.txt', 'core_battery00003726.txt',
             'core_battery00003751.txt', 'core_battery00003745.txt', 'core_battery00003762.txt', 'core_battery00003779.txt',
             'core_battery00003777.txt', 'core_battery00003787.txt', 'core_battery00003798.txt', 'core_battery00003805.txt',
             'core_battery00003819.txt', 'core_battery00003834.txt', 'core_battery00003850.txt', 'core_battery00003885.txt',
             'core_battery00003855.txt', 'core_battery00003875.txt', 'core_battery00003638.txt', 'core_battery00003510.txt',
             'core_battery00003766.txt', 'core_battery00003906.txt', 'core_battery00003912.txt', 'core_battery00003918.txt',
             'core_battery00003928.txt', 'core_battery00003925.txt', 'core_battery00003947.txt', 'core_battery00003960.txt',
             'core_battery00003958.txt', 'core_battery00004021.txt', 'core_battery00004012.txt', 'core_battery00004027.txt',
             'core_battery00004045.txt', 'core_battery00004063.txt', 'core_battery00004067.txt', 'core_battery00004078.txt',
             'coreless_battery00000001.txt', 'coreless_battery00000011.txt', 'coreless_battery00000023.txt',
             'coreless_battery00000017.txt', 'core_battery00000036.txt', 'core_battery00000054.txt',
             'core_battery00000046.txt', 'core_battery00000058.txt', 'core_battery00000073.txt', 'core_battery00000066.txt',
             'coreless_battery00000075.txt', 'core_battery00000091.txt', 'coreless_battery00000165.txt',
             'core_battery00000105.txt', 'coreless_battery00000353.txt', 'core_battery00000125.txt',
             'core_battery00000123.txt', 'core_battery00000139.txt', 'core_battery00000158.txt', 'core_battery00000154.txt',
             'core_battery00000164.txt', 'core_battery00000177.txt', 'core_battery00000186.txt', 'core_battery00000200.txt',
             'core_battery00000224.txt', 'core_battery00000211.txt', 'core_battery00000235.txt', 'core_battery00000244.txt',
             'core_battery00000255.txt', 'core_battery00000263.txt', 'core_battery00000281.txt', 'core_battery00000274.txt',
             'core_battery00000315.txt', 'core_battery00000311.txt', 'core_battery00000324.txt', 'core_battery00000336.txt',
             'core_battery00000348.txt', 'core_battery00000378.txt', 'core_battery00000367.txt', 'core_battery00000385.txt',
             'core_battery00000401.txt', 'core_battery00000396.txt', 'core_battery00000416.txt', 'core_battery00000437.txt',
             'core_battery00000425.txt', 'core_battery00000449.txt', 'core_battery00000455.txt', 'core_battery00000469.txt',
             'core_battery00000487.txt', 'core_battery00000499.txt', 'core_battery00000504.txt', 'core_battery00000521.txt',
             'core_battery00000520.txt', 'core_battery00000535.txt', 'core_battery00000554.txt', 'core_battery00000547.txt',
             'core_battery00000572.txt', 'core_battery00000578.txt', 'core_battery00000590.txt', 'core_battery00000616.txt',
             'core_battery00000598.txt', 'core_battery00000643.txt', 'core_battery00000638.txt', 'core_battery00000647.txt',
             'core_battery00000653.txt', 'core_battery00000673.txt', 'core_battery00000665.txt', 'core_battery00000686.txt',
             'core_battery00000704.txt', 'core_battery00000692.txt', 'core_battery00000718.txt', 'core_battery00000717.txt',
             'core_battery00000713.txt', 'core_battery00000746.txt', 'core_battery00000745.txt', 'core_battery00000751.txt',
             'core_battery00000766.txt', 'core_battery00000757.txt', 'core_battery00000779.txt', 'core_battery00000804.txt',
             'core_battery00000808.txt', 'core_battery00000800.txt', 'core_battery00000827.txt', 'core_battery00000830.txt',
             'core_battery00000840.txt', 'core_battery00000733.txt', 'core_battery00000571.txt', 'core_battery00000846.txt',
             'core_battery00000870.txt', 'core_battery00000856.txt', 'core_battery00000880.txt', 'core_battery00000893.txt',
             'core_battery00000890.txt', 'core_battery00000912.txt', 'core_battery00000923.txt', 'core_battery00000928.txt',
             'core_battery00000939.txt', 'core_battery00000961.txt', 'core_battery00000944.txt', 'core_battery00000970.txt',
             'core_battery00000987.txt', 'core_battery00000978.txt', 'core_battery00001004.txt', 'core_battery00001009.txt',
             'core_battery00001022.txt', 'core_battery00001034.txt', 'core_battery00001042.txt', 'core_battery00001052.txt',
             'core_battery00001083.txt', 'core_battery00001075.txt', 'core_battery00001095.txt', 'core_battery00001109.txt',
             'core_battery00001124.txt', 'core_battery00001128.txt', 'core_battery00001141.txt', 'core_battery00001151.txt',
             'core_battery00001161.txt', 'core_battery00001182.txt', 'core_battery00001180.txt', 'core_battery00001199.txt',
             'core_battery00001226.txt', 'core_battery00001227.txt', 'core_battery00001233.txt', 'core_battery00001241.txt',
             'core_battery00001251.txt', 'core_battery00000841.txt', 'core_battery00001260.txt', 'core_battery00000960.txt',
             'core_battery00001272.txt', 'core_battery00001269.txt', 'core_battery00001277.txt', 'core_battery00001280.txt',
             'core_battery00001296.txt', 'core_battery00001305.txt', 'core_battery00001309.txt', 'core_battery00001330.txt',
             'core_battery00001354.txt', 'core_battery00001351.txt', 'core_battery00001368.txt', 'core_battery00001385.txt',
             'core_battery00001395.txt', 'core_battery00001382.txt', 'core_battery00001424.txt', 'core_battery00001423.txt',
             'core_battery00001434.txt', 'core_battery00001439.txt', 'core_battery00001444.txt', 'core_battery00001460.txt',
             'core_battery00001483.txt', 'core_battery00001470.txt', 'core_battery00001490.txt', 'core_battery00001504.txt',
             'core_battery00001506.txt', 'core_battery00001522.txt', 'core_battery00001552.txt', 'core_battery00001548.txt',
             'core_battery00001563.txt', 'core_battery00001572.txt', 'core_battery00001583.txt', 'core_battery00001605.txt',
             'core_battery00001600.txt', 'core_battery00001611.txt', 'core_battery00001631.txt', 'core_battery00001627.txt',
             'core_battery00001639.txt', 'core_battery00001668.txt', 'core_battery00001648.txt', 'core_battery00001680.txt',
             'core_battery00001362.txt', 'core_battery00001686.txt', 'core_battery00001505.txt', 'core_battery00001626.txt',
             'coreless_battery00002793.txt', 'coreless_battery00004293.txt', 'core_battery00001698.txt',
             'core_battery00001710.txt', 'core_battery00001735.txt', 'core_battery00001730.txt', 'core_battery00001741.txt',
             'core_battery00001758.txt', 'core_battery00001769.txt', 'core_battery00001778.txt', 'core_battery00001791.txt',
             'core_battery00001805.txt', 'core_battery00001812.txt', 'core_battery00001819.txt', 'core_battery00001830.txt',
             'core_battery00001846.txt', 'core_battery00001839.txt', 'core_battery00001858.txt', 'core_battery00001870.txt',
             'core_battery00001864.txt', 'core_battery00001885.txt', 'core_battery00001894.txt', 'core_battery00001907.txt',
             'core_battery00001930.txt', 'core_battery00001935.txt', 'core_battery00001950.txt', 'core_battery00001942.txt',
             'core_battery00001961.txt']

        traindir = set(listdir).difference(set(testdir))
        # traindir = list(traindir)[0:1200]
        for name in testdir:
            imagenames.append(osp.splitext(name)[0])
    else:
        listdir = os.listdir(osp.join('%s' % args.SIXray_root, 'Annotation'))
        for name in listdir:
            imagenames.append(osp.splitext(name)[0])

    if not os.path.isfile(cachefile):
        # print('not os.path.isfile')
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename),imgpath)
            '''
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            '''
        # save
        # print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # print('no,no,no')
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # print (recs)
    # print (classname)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]

        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # print (class_recs)

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    # //
    # //
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('result/ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w, og_im = dataset.pull_item(i)
        # 这里im的颜色偏暗，因为BaseTransform减去了一个mean
        # im_saver = cv2.resize(im[(a2,a1,0),:,:].permute((a1,a2,0)).numpy(), (w,h))

        im_det = og_im.copy()
        im_gt = og_im.copy()

        # print(im_det)
        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        # //
        # //
        # print(detections)
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            # print(boxes)
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

            # print(all_boxes)
            for item in cls_dets:
                # print(item)
                # print(item[5])
                if item[4] > thresh:
                    # print(item)
                    chinese = labelmap[j - 1] + str(round(item[4], 2))
                    print(chinese+'det\n\n')
                    if chinese[0] == '带':
                        chinese = 'P_Battery_Core' + chinese[6:]
                    else:
                        chinese = 'P_Battery_No_Core' + chinese[7:]
                    cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
                    cv2.putText(im_det, chinese, (int(item[0]), int(item[1]) - 5), 0,
                                0.6, (0, 0, 255), 2)
        real = 0
        if gt[0][4] == 3:
            real = 0
        else:
            real = 1

        for item in gt:
            if real == 0:
                print('this pic dont have the obj:', dataset.ids[i])
                break
            # print(chinese+'gt\n\n')
            if chinese[0] == '带':
                chinese = 'P_Battery_Core'
            else:
                chinese = 'P_Battery_No_Core'
            cv2.rectangle(im_det, (int(item[0] * w), int(item[1] * h)), (int(item[2] * w), int(item[3] * h)),
                          (0, 255, 255), 2)
            cv2.putText(im_det, chinese, (int(item[0] * w), int(item[1] * h) - 5), 0, 0.6, (0, 255, 255), 2)
            # print(labelmap[int(item[4])])

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))

        # cv2.imwrite(f'{SIXray_ROOT}/{dataset.ids[i]}_det.jpg', im_det)
        # cv2.imwrite(f'{SIXray_ROOT}/{dataset.ids[i]}_gt.jpg', im_gt)
        # cv2.imwrite('/media/dsg3/husheng/eval/{0}_det.jpg'.format(dataset.ids[i]), im_det)
        # cv2.imwrite('/media/dsg3/husheng/eval/{0}_gt.jpg'.format(dataset.ids[i]), im_gt)
    #     break
    # return
    #
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset)
    do_python_eval(output_dir)


def reset_args(EPOCH):
    global args
    root = os.getcwd()
    args.trained_model = root + "/weights/sixray3000.pth"
    saver_root = root
    if not os.path.exists(saver_root):
        os.mkdir(saver_root)
    args.save_folder = saver_root + '/result/{:d}epoeich_500/'.format(EPOCH)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    else:
        shutil.rmtree(args.save_folder)
        os.mkdir(args.save_folder)

    global devkit_path
    devkit_path = args.save_folder


def xavier(param):
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    # EPOCHS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    # EPOCHS = [85, 90, 95, 100, 105, 110, 115, 120]
    # EPOCHS = [90, 95, 100, 105, 110, 115, 120, 125]
    EPOCHS = [x for x in range(145, 205, 5)]
    for EPOCH in EPOCHS:
        reset_args(EPOCH)


        # load net
        num_classes = len(labelmap) + 1  # +a1 for background
        net = build_ssd('test', 300, num_classes)  # initialize SSD
        # from ssd_net_vgg import SSD
        # net = SSD()
        # net.apply(weights_init)
        # print(dir(net))
        # print(net.state_dict)
        print(args.trained_model)
        # print(torch.load(args.trained_model))
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(args.trained_model))
        else:
            net.load_state_dict(torch.load(args.trained_model, map_location='cpu'))
        net.eval()

        # print('Finished loading model!')
        # load data
        dataset = SIXrayDetection(args.SIXray_root, args.imagesetfile,
                                  BaseTransform(300, dataset_mean),
                                  SIXrayAnnotationTransform(), phase='test',
                                  my_local=args.my_local)
        print(args)

        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        # evaluation

        # with open("./ssd300_120000/test/detections.pkl", 'rb') as fo:  # 读取pkl文件数据
        #     all_boxes = pickle.load(fo, encoding='bytes')
        # output_dir = get_output_dir('ssd300_120000', set_type)
        # evaluate_detections(all_boxes, output_dir, dataset)

        test_net(args.save_folder, net, args.cuda, dataset,
                 BaseTransform(net.size, dataset_mean), args.top_k, 300,
                 thresh=args.confidence_threshold)
        # if EPOCH == 150:
        break
