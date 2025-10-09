from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate._bvp import EPS
from sklearn.utils import shuffle

from loss import instance_contrastive_Loss, MMD
from utils import clustering
from utils.next_batch import next_batch


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent
    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat
    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent
class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)
    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent

def target_l2(q):
    return ((q ** 2).t() / (q ** 2).sum(1)).t()

def compute_view_value(rs, H, view):
    N = H.shape[0]
    w = []
    # all features are normalized
    global_sim = torch.matmul(H, H.t())
    for v in range(view):
        view_sim = torch.matmul(rs[v], rs[v].t())
        related_sim = torch.matmul(rs[v], H.t())
        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N * N)
        w_exp = torch.exp(-w_v)
        w.append(torch.exp(-w_v))
    w = torch.stack(w)
    w = w / torch.sum(w)
    return w.squeeze()

class HSACC(nn.Module):

    def __init__(self, config):
        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        super(HSACC, self).__init__()
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])
        #self.attention_layer = AttentionLayer(config['Autoencoder']['arch1'][-1])

        self.img2txt = Prediction(self._dims_view1)
        self.txt2img = Prediction(self._dims_view2)


    def train(self, config, logger, accumulated_metrics, x1_train,
                         x2_train, Y_list, mask, optimizer, device):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information
              accumulated_metrics: list of metrics
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari

        """

        self.to(device)
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']
        classes=config['training']['class_num']
        best_acc, best_nmi, best_ari = 0.0, 0.0, 0.0
        # Get complete data for training
        flag = (torch.LongTensor([1, 1]).to(device) == mask).int()
        flag = (flag[:, 1] + flag[:, 0]) == 2
        train_view1 = x1_train[flag].to(device).float()  # Ensure data is on the same device
        train_view2 = x2_train[flag].to(device).float()

        for k in range(epochs_total):
            X1, X2 = shuffle(train_view1, train_view2)
            all0 = 0.0
            all1 = 0.0
            all2 = 0.0
            map1 = 0.0
            map2 = 0.0
            all_icl1 = 0.0
            all_icl2 = 0.0
            all_icl = 0.0
            for batch_x1, batch_x2, batch_No in next_batch(X1, X2, batch_size):
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)
                z_half1 = z_half1.to(device).float()  # Ensure z_half1 is on the same device
                z_half2 = z_half2.to(device).float()


                # Within-view Reconstruction Loss
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_half1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_half2), batch_x2)
                reconstruction_loss = recon1 + recon2
                # Cross-view Contrastive_Loss
                z_1, z_2 = z_half1, z_half2

                view_both = torch.add(z_1, z_2).div(2)
                w = compute_view_value(rs=[z_1, z_2], H=view_both, view=2)
                Hnew = z_1 * w[0] + z_2 * w[1]

                loss_icl1 = MMD(z_1, Hnew,kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num'])+\
                     MMD(z_2, Hnew,kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num'])
                # loss_ccl = category_contrastive_loss(H, per_labels,classes, flag_gt)
                loss_icl2 = instance_contrastive_Loss(z_1, z_2, config['training']['alpha'])
                loss_icl=loss_icl1*config['training']['lambda4']+loss_icl2*config['training']['lambda3']
                # Cross-view Dual-Prediction Loss
                img2txt, _ = self.img2txt(z_half1)
                txt2img, _ = self.txt2img(z_half2)
                recon3 = F.mse_loss(img2txt, z_half2)
                recon4 = F.mse_loss(txt2img, z_half1)

                dualprediction_loss = (recon3 + recon4)

                all_loss = loss_icl+ reconstruction_loss * config['training']['lambda1']

                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda2'] * dualprediction_loss


                optimizer.zero_grad()
                all_loss.backward()

                optimizer.step()
                
                all0 += all_loss.item()
                all1 += recon1.item()
                all2 += recon2.item()
                map1 += recon3.item()
                map2 += recon4.item()
                all_icl1 += loss_icl1.item()
                all_icl2 += loss_icl2.item()
                all_icl += loss_icl.item()
            output = "Epoch : {:.0f}/{:.0f} ===> Total loss = {:.4e}".format(k + 1, epochs_total, all0)
            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.autoencoder1.eval(), self.autoencoder2.eval()
                    self.img2txt.eval(), self.txt2img.eval()

                    img_idx_eval = mask[:, 0] == 1  #完整部分
                    txt_idx_eval = mask[:, 1] == 1
                    img_missing_idx_eval = mask[:, 0] == 0 #缺失部分
                    txt_missing_idx_eval = mask[:, 1] == 0

                    imgs_latent_eval = self.autoencoder1.encoder(x1_train[img_idx_eval]) #对各自视图的完整部分进行编码
                    txts_latent_eval = self.autoencoder2.encoder(x2_train[txt_idx_eval])

                    latent_code_img_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                        device)
                    latent_code_txt_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                        device)

                    if x2_train[img_missing_idx_eval].shape[0] != 0:

                        img_missing_latent_eval = self.autoencoder2.encoder(x2_train[img_missing_idx_eval])

                        txt_missing_latent_eval = self.autoencoder1.encoder(x1_train[txt_missing_idx_eval])

                        txt2img_recon_eval, _ = self.txt2img(img_missing_latent_eval)
                        img2txt_recon_eval, _ = self.img2txt(txt_missing_latent_eval)

                        latent_code_img_eval[img_missing_idx_eval] = txt2img_recon_eval
                        latent_code_txt_eval[txt_missing_idx_eval] = img2txt_recon_eval

                    latent_code_img_eval[img_idx_eval] = imgs_latent_eval
                    latent_code_txt_eval[txt_idx_eval] = txts_latent_eval
                    latent_fusion = torch.cat([latent_code_img_eval, latent_code_txt_eval], dim=1).cpu().numpy()

                    scores = clustering.get_score([latent_fusion], Y_list,
                                                  accumulated_metrics['acc'],
                                                  accumulated_metrics['nmi'],
                                                  accumulated_metrics['ARI'])

                    selected_scores = scores['kmeans']
                    if selected_scores['accuracy'] >= best_acc:
                        best_acc = selected_scores['accuracy']
                        best_nmi = selected_scores['NMI']
                        best_ari = selected_scores['ARI']

                    self.autoencoder1.train(), self.autoencoder2.train()
                    self.img2txt.train(), self.txt2img.train()

        return best_acc, best_nmi, best_ari

