import argparse
import math

def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    parser.add_argument('-mod', type=str, required=True, help='mod type:seg,cls,val_ad')
    parser.add_argument('-exp_name', type=str, required=True, help='net type')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=bool, default=False, help='visualization')
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=1,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=1, help='start gpu of model splition')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=256, help='image_size')
    parser.add_argument('-patch_size', type=int, default=7, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=1, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-shuffle_weights', type=float, default = 0.3, help='weight of shuffle loss')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use(data parallel)')
    parser.add_argument(
    '-data_path',
    type=str,
    default='../dataset',
    help='The path of segmentation data')
    opt = parser.parse_args()

    return opt

class Config:
    def __init__(self, imsize="128"):
        larger_dim = max(list(map(int, imsize.split(','))))
        model_image_size = larger_dim if (larger_dim & (larger_dim - 1)) == 0 else (2 ** math.ceil(math.log2(larger_dim)))

        self.net = 'transunet'
        self.baseline = 'unet'
        self.seg_net = 'transunet'
        self.mod = 'rec'
        self.exp_name = 'test_train'
        self.type = 'map'
        self.vis = False
        self.reverse = False
        self.val_freq = 1
        self.gpu = True
        self.gpu_device = 0
        self.sim_gpu = 0
        self.epoch_ini = 1
        self.image_size = model_image_size
        self.patch_size = 7
        self.dim = 512
        self.depth = 1
        self.heads = 16
        self.mlp_dim = 1024
        self.w = 4
        self.b = 16
        self.s = True
        self.warm = 1
        self.lr = 1e-4
        self.imp_lr = 3e-4
        self.weights = '0'
        self.base_weights = '0'
        self.shuffle_weights = 0.3
        self.distributed = 'none'
        self.data_path = '../dataset'