import numpy as np
import scipy.io as sio
import warnings
import data_provider.image as dataset
import model.dqn.dqn as model
import sys
from pprint import pprint
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define input arguments
lr = float(sys.argv[1])
output_dim = int(sys.argv[2])
iter_num = int(sys.argv[3])
cq_lambda = float(sys.argv[4])
subspace_num = int(sys.argv[5])
n_subcenter = int(sys.argv[6])
_dataset = sys.argv[7]
gpu = sys.argv[8]
log_dir = sys.argv[9]
data_root = sys.argv[10]

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 21, 'coco': 91}
Rs = {'cifar10': 59000, 'nuswide_81': 5000, 'coco': 5000,
      'nuswide_21': 5000, 'imagenet': 5000, 'cifar10_zero_shot': 15000}

config = {
    'device': '/gpu:' + gpu,
    'max_iter': iter_num,
    'batch_size': 128,
    'val_batch_size': 100,
    'decay_step': 500,                   # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.5,   # Learning rate decay factor.
    'learning_rate': lr,                 # Initial learning rate img.

    'output_dim': output_dim,

    # 'R': 5,
    'R': Rs[_dataset],
    'model_weights': '../../DeepHash/architecture/pretrained_model/reference_pretrain.npy',

    'img_model': 'alexnet',

    # if only finetune last layer
    'finetune_all': True,

    # CQ params
    'max_iter_update_b': 3,
    'max_iter_update_Cb': 1,
    'cq_lambda': cq_lambda,
    'code_batch_size': 500,
    'n_subspace': subspace_num,
    'n_subcenter': n_subcenter,

    'label_dim': label_dims[_dataset],
    'img_tr': "../../data/{}/train.txt".format(_dataset),
    'img_te': "../../data/{}/test.txt".format(_dataset),
    'img_db': "../../data/{}/database.txt".format(_dataset),
    'save_dir': "./models/",
    'log_dir': log_dir,
    'dataset': _dataset
}

pprint(config)

train_img = dataset.import_train(data_root, config['img_tr'])
model_weights = model.train(train_img, config)

config['model_weights'] = model_weights
query_img, database_img = dataset.import_validation(data_root, config['img_te'], config['img_db'])
maps = model.validation(database_img, query_img, config)

# for key in maps:
#     print(("{}: {}".format(key, maps[key])))
result_txt = './result.txt'
f = open(result_txt, 'a')
for key in maps:
    print(("{}\t{}".format(key, maps[key])))
    f.write(("{}\t{}\n".format(key, maps[key])))
f.close()

pprint(config)
