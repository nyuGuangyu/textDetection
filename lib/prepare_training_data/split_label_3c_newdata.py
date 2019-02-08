import os
import numpy as np
import math
import cv2 as cv
import re

path = '/home/gl00/text-detection-ctpn-master/lib/prepare_training_data/ocr_images'
gt_path = '/home/gl00/text-detection-ctpn-master/lib/prepare_training_data/gt.txt'
train_out_path = 'ocr_image_newdata_train'
test_out_path = 'ocr_image_newdata_test'

if not os.path.exists(train_out_path):
    os.makedirs(train_out_path)
if not os.path.exists(test_out_path):
    os.makedirs(test_out_path)
files = os.listdir(path)
files.sort()
count_latin = 0
count_asian = 0
ind = 0

gt_dict = {}

with open(gt_path, 'r') as f:
    lines = f.readlines()
train_test_cutoff = 2500

p = r"[{\[].*?[}\]]"
keywords = ['left', 'top', 'width', 'height', 'label']

for line in lines[1:]:
    line = line.split('\t')
    basename, annotation = line[0], line[1].strip()
    anno_list = re.findall(p, annotation)
# every image contains a bounding box list, every bounding box in this list is dict:{"left","top","width","height","label"}
    anno_dict_list = []
    for e in anno_list: # for each bounding box in one image
        e = e.strip('[')
        e = e.strip(']')
        e = e.strip('{')
        e = e.strip('}')
        if "left" not in e or "top" not in e or "width" not in e or "height" not in e or "label" not in e:
            continue
        anno_dict = {}
        for item in e.split(','): # left,top,width,height,label must be in item
            if ':' not in item:
                continue
            # eg k:'left',v:172
            try:
                k,v = item.split(':')[0].strip('"'), int(item.split(':')[1])
            except:
                k, v = item.split(':')[0].strip('"'), item.split(':')[1].strip('"')
            anno_dict[k] = v
        anno_dict_list.append(anno_dict)
            
        # anno_dict = dict((k.strip('"'), v.strip('"')) for k,v in (item.split(':') for item in e.split(',')))
        # anno_dict_list.append(anno_dict)
    gt_dict[basename] = anno_dict_list

for basename in files:
    if basename.lower().split('.')[-1] not in ['jpg', 'png']:
        continue
    stem, ext = os.path.splitext(basename)
    img_path = os.path.join(path, basename)
    img = cv.imread(img_path)
    # if type(img) == 'NoneType':
    #     continue
    try:
        img_size = img.shape
    except:
        continue
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
    re_size = re_im.shape
    #cv.imwrite(os.path.join(out_path, stem) + '.jpg', re_im)
    found = 0
    if basename not in gt_dict:
        print('no corresponding gt info found.')
        continue
    boxes = gt_dict[basename]

    for box in boxes:
        #find all languages
        chinese_chars = re.findall(ur'[\u4e00-\u9fff]+', box['label'].decode('utf-8'))
        found = 1
        if chinese_chars:
            language = 'asian'
            count_asian += 1
        else:
            language = 'latin'
            count_latin += 1
        pt_x = np.zeros((4, 1))
        pt_y = np.zeros((4, 1))
        # clockwise
        x1 = box['left'] - int(box['width']/2.)
        x2 = box['left'] + int(box['width']/2.)
        x3 = x2
        x4 = x1
        y1 = box['top'] - int(box['height']/2.)
        y2 = y1
        y3 = box['top'] + int(box['height']/2.)
        y4 = y3
        pt_x[0, 0] = int(float(x1) / img_size[1] * re_size[1])
        pt_y[0, 0] = int(float(y1) / img_size[0] * re_size[0])
        pt_x[1, 0] = int(float(x2) / img_size[1] * re_size[1])
        pt_y[1, 0] = int(float(y2) / img_size[0] * re_size[0])
        pt_x[2, 0] = int(float(x3) / img_size[1] * re_size[1])
        pt_y[2, 0] = int(float(y3) / img_size[0] * re_size[0])
        pt_x[3, 0] = int(float(x4) / img_size[1] * re_size[1])
        pt_y[3, 0] = int(float(y4) / img_size[0] * re_size[0])

        ind_x = np.argsort(pt_x, axis=0)
        pt_x = pt_x[ind_x]
        pt_y = pt_y[ind_x]

        if pt_y[0] < pt_y[1]:
            pt1 = (pt_x[0], pt_y[0])
            pt3 = (pt_x[1], pt_y[1])
        else:
            pt1 = (pt_x[1], pt_y[1])
            pt3 = (pt_x[0], pt_y[0])

        if pt_y[2] < pt_y[3]:
            pt2 = (pt_x[2], pt_y[2])
            pt4 = (pt_x[3], pt_y[3])
        else:
            pt2 = (pt_x[3], pt_y[3])
            pt4 = (pt_x[2], pt_y[2])

        xmin = int(min(pt1[0], pt2[0]))
        ymin = int(min(pt1[1], pt2[1]))
        xmax = int(max(pt2[0], pt4[0]))
        ymax = int(max(pt3[1], pt4[1]))

        if xmin < 0:
            xmin = 0
        if xmax > re_size[1] - 1:
            xmax = re_size[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > re_size[0] - 1:
            ymax = re_size[0] - 1

        width = xmax - xmin
        height = ymax - ymin

        # reimplement
        step = 16.0
        x_left = []
        x_right = []
        x_left.append(xmin)
        x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
        if x_left_start == xmin:
            x_left_start = xmin + 16
        for i in np.arange(x_left_start, xmax, 16):
            x_left.append(i)
        x_left = np.array(x_left)

        x_right.append(x_left_start - 1)
        for i in range(1, len(x_left) - 1):
            x_right.append(x_left[i] + 15)
        x_right.append(xmax)
        x_right = np.array(x_right)

        idx = np.where(x_left == x_right)
        x_left = np.delete(x_left, idx, axis=0)
        x_right = np.delete(x_right, idx, axis=0)

        if not os.path.exists('train_label_ocr'):
            os.makedirs('train_label_ocr')
        if not os.path.exists('test_label_ocr'):
            os.makedirs('test_label_ocr')
        if ind <= train_test_cutoff:
            with open(os.path.join('train_label_ocr', stem) + '.txt', 'a') as f:
                for i in range(len(x_left)):
                    f.writelines(language+"\t")
                    f.writelines(str(int(x_left[i])))
                    f.writelines("\t")
                    f.writelines(str(int(ymin)))
                    f.writelines("\t")
                    f.writelines(str(int(x_right[i])))
                    f.writelines("\t")
                    f.writelines(str(int(ymax)))
                    f.writelines("\n")
        else:
            with open(os.path.join('test_label_ocr', stem) + '.txt', 'a') as f:
                for i in range(len(x_left)):
                    f.writelines(language+"\t")
                    f.writelines(str(int(x_left[i])))
                    f.writelines("\t")
                    f.writelines(str(int(ymin)))
                    f.writelines("\t")
                    f.writelines(str(int(x_right[i])))
                    f.writelines("\t")
                    f.writelines(str(int(ymax)))
                    f.writelines("\n")

    if found == 1:
        print("count latin "+str(count_latin))
        print("count asian " + str(count_asian))
        #re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_LINEAR)
        #re_size = re_im.shape
        if ind <= train_test_cutoff:
            cv.imwrite(os.path.join(train_out_path, stem) + '.jpg', re_im)
            print("wrote {0}/{1}".format(train_out_path, stem))
        else:
            cv.imwrite(os.path.join(test_out_path, stem) + '.jpg', re_im)
            print("wrote {0}/{1}".format(test_out_path, stem))
        ind += 1