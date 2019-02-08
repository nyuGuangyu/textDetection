import numpy as np
from lib.fast_rcnn.config import cfg
from lib.roi_data_layer.minibatch import get_minibatch
import sys

class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self,isTrain=True):
        """Return the roidb indices for the next minibatch."""
        if isTrain:
            if cfg.TRAIN.HAS_RPN:
                if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
                    self._shuffle_roidb_inds()

                db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
                self._cur += cfg.TRAIN.IMS_PER_BATCH
            else:
                # sample images
                db_inds = np.zeros((cfg.TRAIN.IMS_PER_BATCH), dtype=np.int32)
                i = 0
                while (i < cfg.TRAIN.IMS_PER_BATCH):
                    ind = self._perm[self._cur]
                    num_objs = self._roidb[ind]['boxes'].shape[0]
                    if num_objs != 0:
                        db_inds[i] = ind
                        i += 1

                    self._cur += 1
                    if self._cur >= len(self._roidb):
                        self._shuffle_roidb_inds()
        else:
            if self._cur + cfg.TRAIN.IMS_PER_BATCH < len(self._roidb):
                db_inds = range(len(self._roidb))[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
                self._cur += cfg.TRAIN.IMS_PER_BATCH
            else:
                print 'Done evaluating all testing data, now check the last shown accumulated confusion matrix as the final result.'
                sys.exit()

        return db_inds

    def _get_next_minibatch(self,isTrain=True):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds(isTrain=isTrain)
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)
            
    def forward(self, isTrain=True):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch(isTrain=isTrain)
        return blobs
