import argparse
import itertools
import time
import os
import random
import numpy as np

import scipy
import torch
from scipy.sparse import csr_matrix

from model import HSACC
from utils.get_mask import get_mask
from utils.util import cal_std
from utils.logger_ import get_logger
from utils.datasets import *
from configure.configure import get_default_config
import collections
import warnings

warnings.simplefilter("ignore")

dataset = {
    0: "Caltech101-20",
    1: "NoisyMNIST",
    2: "LandUse_21",
    3: "100leaves",
    4: "Hdigit"
}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='2', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0.5', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['missing_rate'] = args.missing_rate
    config['print_num'] = args.print_num
    config['dataset'] = dataset
    logger, plt_name = get_logger(config)

    logger.info('Dataset:' + str(dataset))
    for (k, v) in config.items():
        if isinstance(v, dict):
            logger.info("%s={" % (k))
            for (g, z) in v.items():
                logger.info("          %s = %s" % (g, z))
        else:
            logger.info("%s = %s" % (k, v))

    # Load data
    X_list, Y_list = load_data(config)
    x1_train_raw = X_list[0]
    #x1_train_raw_dense = x1_train_raw.toarray() if scipy.sparse.issparse(x1_train_raw) else x1_train_raw  #BBCSport数据集要，其余数据集删掉

    x2_train_raw = X_list[1]
    #x2_train_raw_dense = x2_train_raw.toarray() if scipy.sparse.issparse(x2_train_raw) else x2_train_raw  #BBCSport数据集要

    fold_acc, fold_nmi, fold_ari = [], [], []

    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)

        # Get Mask
        mask = get_mask(2, x1_train_raw.shape[0], config['missing_rate'])

        # print(mask.shape)

        x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
        x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)
        mask = torch.from_numpy(mask).long().to(device)

        # Accumulated metrics
        accumulated_metrics = collections.defaultdict(list)

        # Set random seeds
        if config['missing_rate'] == 0:
            seed = data_seed
        else:
            seed = config['seed']

        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Build model
        HSACC_model = HSACC(config)
        optimizer = torch.optim.Adam(
            itertools.chain(HSACC_model.autoencoder1.parameters(), HSACC_model.autoencoder2.parameters(),
                            HSACC_model.img2txt.parameters(), HSACC_model.txt2img.parameters()),
            lr=config['training']['lr'])

        # Print the models
        logger.info(HSACC_model.autoencoder1)
        logger.info(HSACC_model.img2txt)
        logger.info(optimizer)

        HSACC_model.autoencoder1.to(device), HSACC_model.autoencoder2.to(device)
        HSACC_model.img2txt.to(device), HSACC_model.txt2img.to(device)

        # Training
        acc, nmi, ari = HSACC_model.train(config, logger, accumulated_metrics, x1_train, x2_train, Y_list, mask,
                                             optimizer, device)
        fold_acc.append(acc)
        fold_nmi.append(nmi)
        fold_ari.append(ari)

        print(time.time() - start)

    logger.info('--------------------Training over--------------------')
    acc, nmi, ari = cal_std(logger, fold_acc, fold_nmi, fold_ari)


if __name__ == '__main__':
    main()
