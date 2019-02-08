# -*- coding:utf-8 -*-
import numpy as np
from .generate_anchors import generate_anchors
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from lib.fast_rcnn.nms_wrapper import nms
import sys


DEBUG = False
LANG1_FRAC = 0.85
LANG2_FRAC = 1. - LANG1_FRAC

"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""
def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride = [16,], anchor_scales = [16,]):
    """
    Parameters
    ----------
    rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)

    """
    # cfg_key=cfg_key.decode('ascii')
    _anchors = generate_anchors(scales=np.array(anchor_scales))#生成基本的9个anchor
    _num_anchors = _anchors.shape[0]#9个anchor

    im_info = im_info[0]#原始图像的高宽、缩放尺度

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'

    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N#12000,在做nms之前，最多保留的候选box数目
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N#2000，做完nms之后，最多保留的box的数目
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH#nms用参数，阈值是0.7
    min_size      = cfg[cfg_key].RPN_MIN_SIZE#候选box的最小尺寸，目前是16，高宽均要大于16
    #TODO 后期需要修改这个最小尺寸，改为8？

    height, width = rpn_cls_prob_reshape.shape[1:3]#feature-map的高宽

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    # (1, H, W, A)
    print(rpn_cls_prob_reshape.shape)
    n_classes = cfg.NCLASSES
    #scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, n_classes])[:,:,:,:,1],
    #                    [1, height, width, _num_anchors])

    scores_lang0 = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, n_classes])[:,:,:,:,0],
                        [1, height, width, _num_anchors])
    scores_lang1 = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, n_classes])[:,:,:,:,1],
                        [1, height, width, _num_anchors])
    scores_lang2 = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, _num_anchors, n_classes])[:,:,:,:,2],
                        [1, height, width, _num_anchors])
    # print(scores_lang0)
    # print(len(scores_lang2[scores_lang2>0.55]))
    # sys.exit()

    #提取到object的分数，non-object的我们不关心, update: scores 分成两个 scores_lang1: latin；scores_lang2：asian
    #并reshape到1*H*W*9

    bbox_deltas = rpn_bbox_pred#模型输出的pred是相对值，需要进一步处理成真实图像中的坐标
    #im_info = bottom[2].data[0, :]

    if DEBUG:
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))

    # 1. Generate proposals from bbox deltas and shifted anchors
    if DEBUG:
        print('score_lang1 map size: {}'.format(scores_lang1.shape))

    # Enumerate all shifts
    # 同anchor-target-layer-tf这个文件一样，生成anchor的shift，进一步得到整张图像上的所有anchor
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))#这里得到的anchor就是整张图像上的所有anchor

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.reshape((-1, 4)) #(HxWxA, 4)

    # Same story for the scores:
    #scores = scores.reshape((-1, 1))
    scores_lang1 = scores_lang1.reshape((-1, 1))
    scores_lang2 = scores_lang2.reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)#做逆变换，得到box在图像上的真实坐标

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])#将所有的proposal修建一下，超出图像范围的将会被修剪掉

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size * im_info[2])#移除那些proposal小于一定尺寸的proposal
    proposals = proposals[keep, :]#保留剩下的proposal
    #scores = scores[keep]
    scores_lang1 = scores_lang1[keep]
    scores_lang2 = scores_lang2[keep]

    bbox_deltas=bbox_deltas[keep,:]


    # # remove irregular boxes, too fat too tall
    keep = _filter_irregular_boxes(proposals)
    proposals = proposals[keep, :]
    #scores = scores[keep]
    scores_lang1 = scores_lang1[keep]
    scores_lang2 = scores_lang2[keep]


    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    #order = scores.ravel().argsort()[::-1]#score按得分的高低进行排序
    order_lang1 = scores_lang1.ravel().argsort()[::-1] # 分别sort lang1和lang2 的 scores,然后取并集决定保留proposal
    order_lang2 = scores_lang2.ravel().argsort()[::-1]
    order = []

    if pre_nms_topN > 0:                #保留12000个proposal进去做nms
        #order = order[:pre_nms_topN]
        # order_lang1 = list(order_lang1[:int(LANG1_FRAC*pre_nms_topN)]) # 分别取一部分出来组成order
        # order_lang2 = list(order_lang2[:int(LANG2_FRAC*pre_nms_topN)])
        # order = list(set(order_lang1) | set(order_lang2))
        
        # 去除重合
        order_lang2 = order_lang2[:int(LANG2_FRAC*pre_nms_topN)]
        # lang2 is first priority.
        temp = list(order_lang1)
        for e in list(order_lang2):
            if e in temp:
                temp.pop(temp.index(e))
        order_lang1 = np.array(temp)
        order_lang1 = order_lang1[:int(LANG1_FRAC*pre_nms_topN)]
        set_a = set(list(order_lang1))
        set_b = set(list(order_lang2))
        order = list(order_lang1) + list(order_lang2)
        assert len(set_a&set_b) == 0
        
 
    proposals = proposals[order, :]
    #scores = scores[order]
    a = np.zeros((len(order),1))
    a[:len(order_lang1)] = scores_lang1[order_lang1]
    scores_lang1 = a

    b = np.zeros((len(order),1))
    b[len(order_lang1):] = scores_lang2[order_lang2]
    scores_lang2 = b

    bbox_deltas=bbox_deltas[order,:]


    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    #keep = nms(np.hstack((proposals, scores)), nms_thresh)#进行nms操作，保留2000个proposal
    keep_lang1 = nms(np.hstack((proposals, scores_lang1.astype(np.float32, copy=False))), nms_thresh)
    keep_lang2 = nms(np.hstack((proposals, scores_lang2.astype(np.float32, copy=False))), nms_thresh)

    if post_nms_topN > 0:
        #keep = keep[:post_nms_topN]
        # keep_lang1 = keep_lang1[:int(LANG1_FRAC*post_nms_topN)]
        # keep_lang2 = keep_lang2[:int(LANG2_FRAC*post_nms_topN)]

        # 去除重合
        keep_lang2 = keep_lang2[:int(LANG2_FRAC*post_nms_topN)]
        # lang2 is first priority.
        temp = list(keep_lang1)
        for e in list(keep_lang2):
            if e in temp:
                temp.pop(temp.index(e))
        keep_lang1 = np.array(temp)
        keep_lang1 = keep_lang1[:int(LANG1_FRAC*post_nms_topN)]
        set_a = set(list(keep_lang1))
        set_b = set(list(keep_lang2))
        keep = list(keep_lang1) + list(keep_lang2)
        assert len(set_a&set_b) == 0

    # keep = list(set(keep_lang1) | set(keep_lang2))
    proposals = proposals[keep, :]
    #scores = scores[keep]
    # scores_lang1 = scores_lang1[keep]
    # scores_lang2 = scores_lang2[keep]
    a = np.zeros((len(keep),1))
    a[:len(keep_lang1)] = scores_lang1[keep_lang1]
    scores_lang1 = a

    b = np.zeros((len(keep),1))
    b[len(keep_lang1):] = scores_lang2[keep_lang2]
    scores_lang2 = b

    # print(scores_lang1)
    # print(scores_lang2)
    # sys.exit()

    bbox_deltas=bbox_deltas[keep,:]


    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    #blob = np.hstack((scores.astype(np.float32, copy=False), proposals.astype(np.float32, copy=False)))]
    blob = np.hstack((scores_lang1.astype(np.float32, copy=False), scores_lang2.astype(np.float32, copy=False)))
    blob = np.hstack((blob, proposals.astype(np.float32, copy=False)))

    return blob,bbox_deltas


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def _filter_irregular_boxes(boxes, min_ratio = 0.2, max_ratio = 5):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    rs = ws / hs
    keep = np.where((rs <= max_ratio) & (rs >= min_ratio))[0]
    return keep
