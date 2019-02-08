from .pascal_voc import pascal_voc
import sys
__sets = {}
def _selective_search_IJCV_top_k(split, year, top_k):
    imdb = pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb
# Set up voc_<year>_<split> using selective search "fast" mode
global_isTrain = True
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                pascal_voc(split, year, global_isTrain))

def get_imdb(name, isTrain=True):
    """Get an imdb (image database) by name."""
    global global_isTrain
    global_isTrain = isTrain
    if name not in __sets:
        print((list_imdbs()))
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
