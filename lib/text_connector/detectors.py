#coding:utf-8
import numpy as np
from lib.fast_rcnn.nms_wrapper import nms
from lib.fast_rcnn.config import cfg
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented
from .text_connect_cfg import Config as TextLineCfg
import sys


class TextDetector:
    def __init__(self):
        self.mode= cfg.TEST.DETECT_MODE
        if self.mode == "H":
            self.text_proposal_connector=TextProposalConnector()
        elif self.mode == "O":
            self.text_proposal_connector=TextProposalConnectorOriented()

        
    def detect(self, text_proposals,scores,size, lang=0):
        # 删除得分较低的proposal
        #keep_inds=np.where(scores>TextLineCfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        # print(len(scores))
        if lang == 1:
            TEXT_PROPOSALS_MIN_SCORE = 0.10
        elif lang == 2:
            TEXT_PROPOSALS_MIN_SCORE = 0.10
        else:
            TEXT_PROPOSALS_MIN_SCORE = 0.7
        keep_inds=np.where(scores>TEXT_PROPOSALS_MIN_SCORE)[0]
        # print(len(keep_inds))

        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

        # 按得分排序
        sorted_indices=np.argsort(scores.ravel())[::-1]
        text_proposals, scores=text_proposals[sorted_indices], scores[sorted_indices]

        # 对proposal做nms
        keep_inds=nms(np.hstack((text_proposals, scores)), TextLineCfg.TEXT_PROPOSALS_NMS_THRESH)
        # print(len(keep_inds))

        text_proposals, scores=text_proposals[keep_inds], scores[keep_inds]

        # 获取检测结果
        text_recs=self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        keep_inds=self.filter_boxes(text_recs,lang)
        # print(len(keep_inds))

        return text_recs[keep_inds]

    def filter_boxes(self, boxes,lang=0):
        heights=np.zeros((len(boxes), 1), np.float)
        widths=np.zeros((len(boxes), 1), np.float)
        scores=np.zeros((len(boxes), 1), np.float)
        index=0
        for box in boxes:
            heights[index]=(abs(box[5]-box[1])+abs(box[7]-box[3]))/2.0+1
            widths[index]=(abs(box[2]-box[0])+abs(box[6]-box[4]))/2.0+1
            scores[index] = box[8]
            index += 1
        if lang == 1:
            LINE_MIN_SCORE = 0.1
        elif lang == 2:
            LINE_MIN_SCORE = 0.1
        else:
            LINE_MIN_SCORE = 0.5

        #return np.where((widths/heights>TextLineCfg.MIN_RATIO) & (scores>TextLineCfg.LINE_MIN_SCORE) &
                          #(widths>(TextLineCfg.TEXT_PROPOSALS_WIDTH*TextLineCfg.MIN_NUM_PROPOSALS)))[0]
        return np.where((widths/heights>TextLineCfg.MIN_RATIO) & (scores>LINE_MIN_SCORE) &
                          (widths>(TextLineCfg.TEXT_PROPOSALS_WIDTH*TextLineCfg.MIN_NUM_PROPOSALS)))[0]
