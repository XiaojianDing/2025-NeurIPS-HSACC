import argparse
import itertools
import time
import torch

from model_multiview import HSACCMultiView
from utils.get_mask import get_mask
from utils.util import cal_std
from utils.logger_ import get_logger
from utils.datasets import *
from configure.configure_multiview import get_default_config
import collections
import warnings

warnings.simplefilter("ignore")

dataset = {
    1:"Mfeat"
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=int, default='1', help='dataset id')
parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
parser.add_argument('--print_num', type=int, default='50', help='gap of print evaluations')
parser.add_argument('--test_time', type=int, default='5', help='number of test times')
parser.add_argument('--missing_rate', type=float, default='0.5', help='missing rate')

args = parser.parse_args()
dataset = dataset[args.dataset]


def main():
    total_start = time.time()
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
    X_list, Y_list = load_multiview_data(config)

    fold_acc, fold_nmi, fold_ari = [], [], []
    for data_seed in range(1, args.test_time + 1):
        start = time.time()
        np.random.seed(data_seed)

        # Get Mask
        mask = get_mask(config['view'], X_list[0].shape[0], config['missing_rate'])

        X_list_new = []
        for i in range(config['view']):
            X_list_new.append(X_list[i] * mask[:, i][:, np.newaxis])
            # 假设 X_list[i] 的大小为 (4659, 685)，mask[:, i] 的大小为 (685,)
            # 需要将 mask[:, i] 转换为 (685, 1) 来与 X_list[i] 相乘

            #X_list_new.append(X_list[i].multiply(mask[:, i][:, np.newaxis]))  #BBC4view需要，删除上面那个

        # 将 X_list_new[i] 从稀疏矩阵转换为稠密的 numpy 数组，再转换为 Torch 张量
        #X_list_new = [torch.from_numpy(X_list_new[i].toarray()).float().to(device) for i in range(config['view'])] #BBC4view需要。
        X_list_new = [torch.from_numpy(X_list_new[i]).float().to(device) for i in range(config['view'])]

        mask = torch.from_numpy(mask).long().to(device)

        # Accumulated metrics
        accumulated_metrics = collections.defaultdict(list)

        # Set random seeds
        if config['missing_rate'] == 0:
            seed = data_seed * config['seed']
        else:
            seed = config['seed']

        np.random.seed(seed)
        random.seed(seed + 1)
        torch.manual_seed(seed + 2)
        torch.cuda.manual_seed(seed + 3)
        torch.backends.cudnn.deterministic = True

        # Build model
        HSACC_model = HSACCMultiView(config)
        optimizer = torch.optim.Adam(HSACC_model.parameters(), lr=config['training']['lr'])

        logger.info(HSACC_model.autoencoder1)

        logger.info(optimizer)
        HSACC_model.to(device)

        acc, nmi, ari = HSACC_model.train_multiview(config, logger, accumulated_metrics, X_list_new,
                                                   Y_list, mask, optimizer, device)
        fold_acc.append(acc)
        fold_nmi.append(nmi)
        fold_ari.append(ari)

        print(time.time() - start)

    logger.info('--------------------Training over--------------------')

    acc, nmi, ari = cal_std(logger, fold_acc, fold_nmi, fold_ari)



if __name__ == '__main__':
    main()
