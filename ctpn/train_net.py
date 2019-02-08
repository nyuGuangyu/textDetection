import os.path
import pprint
import sys
#sys.path.append(os.getcwd())
sys.path += ['/home/gl00/text-detection-ctpn-master','/home/gl00/miniconda2/lib/python2.7/site_packages']
print(sys.path)
import pyximport
pyximport.install()
from lib.fast_rcnn.train import get_training_roidb, train_net
from lib.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib.datasets.factory import get_imdb
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg


if __name__ == '__main__':
    cfg_from_file('text.yml')
    print('Using config:')
    pprint.pprint(cfg)
    imdb = get_imdb('voc_2007_trainval')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    log_dir = get_log_dir(imdb)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))

    device_name = '/gpu:0'
    print(device_name)

    network = get_network('VGGnet_train')

    train_net(network, imdb, roidb,
              output_dir=output_dir,
              log_dir=log_dir,
              pretrained_model='../data/pretrain/VGG_imagenet.npy',
              max_iters=int(cfg.TRAIN.max_steps),
              restore=True) # bool(int(cfg.TRAIN.restore))
