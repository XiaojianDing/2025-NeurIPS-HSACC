from __future__ import print_function, absolute_import, division
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle

from loss import instance_contrastive_Loss,MMD
from utils import clustering
from utils.next_batch import next_batch_multiview
from model import Autoencoder

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # The implementation of MMD
        w_v = (torch.sum(view_sim) + torch.sum(global_sim) - 2 * torch.sum(related_sim)) / (N * N)
        w_exp = torch.exp(-w_v)
        w.append(torch.exp(-w_v))
    w = torch.stack(w)
    w = w / torch.sum(w)

    return w.squeeze()
class SharedPredictionBase(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(SharedPredictionBase, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.shared_net = nn.Sequential(*layers)

    def forward(self, x):
        return self.shared_net(x)
class SpecificHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpecificHead, self).__init__()
        self.head = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.head(x)
class PartialSharedPredictor(nn.Module):
    def __init__(self, input_dim, shared_hidden_dims, output_dim, num_views):
        super(PartialSharedPredictor, self).__init__()
        self.shared_base = SharedPredictionBase(input_dim, shared_hidden_dims)
        self.specific_heads = nn.ModuleList([SpecificHead(shared_hidden_dims[-1], output_dim) for _ in range(num_views)])

    def forward(self, x, target_view_idx):
        h = self.shared_base(x)
        out = self.specific_heads[target_view_idx](h)
        return out, h
class HSACCMultiView(torch.nn.Module):
    # Dual contrastive prediction for multi-view
    def __init__(self, config):
        super(HSACCMultiView, self).__init__()

        """Constructor.

        Args:
            config: parameters defined in configure.py.
        """
        self._config = config
        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self.view_num = config['view']
        n_clusters = config['training']['class_num']


        for i in range(self.view_num):
            autoencoder = Autoencoder(config['Autoencoder'][f'arch{i + 1}'], config['Autoencoder']['activations'], config['Autoencoder']['batchnorm'])
            self.add_module('autoencoder{}'.format(i), autoencoder)
            dims_view = [self._latent_dim] + self._config['Prediction'][f'arch{i + 1}']

        self.shared_pred_hidden_dims = self._config['Prediction']['shared_hidden']
        self.pred_output_dim = self._latent_dim
        self.partial_predictors = nn.ModuleList([
            PartialSharedPredictor(
                input_dim=self._latent_dim,
                shared_hidden_dims=self.shared_pred_hidden_dims,
                output_dim=self.pred_output_dim,
                num_views=self.view_num
            ) for _ in range(self.view_num)
        ])

    def train_multiview(self, config, logger, accumulated_metrics, X_list, Y_list, mask, optimizer, device):

        """Training the model with cove view for clustering

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              accumulated_metrics: list of metrics
              X_list: list data of all view
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """
        epochs_total = config['training']['epoch']
        batch_size = config['training']['batch_size']

        # select the complete samples
        flag = torch.ones(len(X_list)).to(device)
        flag = (mask == flag.long())
        flag = torch.all(flag == 1, dim=1)
        train_views = [x[flag] for x in X_list]
        best_acc, best_nmi, best_ari = 0, 0, 0
        for k in range(epochs_total):
            shuffled_views = shuffle(*train_views)
            loss_all, rec, dul, icl = 0, 0, 0, 0
            for batch_view, batch_No in next_batch_multiview(shuffled_views, batch_size):
                # (Todo) Currently, view 0 is the core view as default
                latent_view_z = []
                reconstruction_loss = 0
                # Within-view Reconstruction Loss
                for i in range(self.view_num):
                    autoencoder = getattr(self, f'autoencoder{i}')
                    # Get the hidden states for each view
                    latent_view_z.append(autoencoder.encoder(batch_view[i]))
                    reconstruction_loss += F.mse_loss(autoencoder.decoder(latent_view_z[i]), batch_view[i])
                reconstruction_loss /= self.view_num

                # Instance-level Contrastive Loss
                icl_loss1 = 0
                pair_count = 0
                for i in range(self.view_num):
                    for j in range(i + 1, self.view_num):
                        icl_loss1 += instance_contrastive_Loss(latent_view_z[i], latent_view_z[j],
                                                               config['training']['alpha'])
                        pair_count += 1
                icl_loss1 /= pair_count
                with torch.no_grad():
                    view_both = torch.stack(latent_view_z, dim=0).mean(dim=0)
                w = compute_view_value(rs=latent_view_z, H=view_both, view=self.view_num)
                H = sum(latent_view_z[i] * w[i] for i in range(self.view_num))  # shape: [batch_size, dim]
                icl_loss2 = 0
                for i in range(self.view_num):
                    icl_loss2 += MMD(latent_view_z[i], H,
                                     kernel_mul=config['training']['kernel_mul'],
                                     kernel_num=config['training']['kernel_num'])
                icl_loss2 /= self.view_num
                icl_loss = icl_loss1 * config['training']['lambda2'] + icl_loss2 * config['training']['lambda3']

                dualprediction_loss = 0
                count = 0
                for i in range(self.view_num):
                    for j in range(self.view_num):
                        if i != j:
                            prediction_net = self.partial_predictors[i]
                            pred_z_j, _ = prediction_net(latent_view_z[i], j)
                            dualprediction_loss += F.mse_loss(pred_z_j, latent_view_z[j])
                            count += 1
                dualprediction_loss /= count

                all_loss = icl_loss+ reconstruction_loss * config['training']['lambda1']
                if k >= config['training']['start_dual_prediction']:
                    all_loss += config['training']['lambda4'] * dualprediction_loss

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                loss_all += all_loss.item()


            output = f"Epoch : {k + 1}/{epochs_total} ===> Total loss = {loss_all:.4e}"

            if (k + 1) % config['print_num'] == 0:
                logger.info("\033[2;29m" + output + "\033[0m")

            if (k + 1) % config['print_num'] == 0:
                with torch.no_grad():
                    self.eval()
                    latent_codes_eval = [torch.zeros(X_list[i].shape[0], self._latent_dim).to(device) for i in
                                         range(self.view_num)]
                    for i in range(self.view_num):
                        existing_idx_eval = mask[:, i] == 1
                        latent_codes_eval[i][existing_idx_eval] = getattr(self, f'autoencoder{i}').encoder(
                            X_list[i][existing_idx_eval])
                    for i in range(self.view_num):
                        if i == 0:
                            missing_idx_eval = mask[:, i] == 0
                            accumulated = (mask[:, i] == 1).float().to(device)
                            if missing_idx_eval.sum() != 0:
                                for j in range(1, self.view_num):
                                    jhas_idx = missing_idx_eval * (mask[:, j] == 1)
                                    accumulated += jhas_idx.float()
                                    if jhas_idx.sum() != 0:
                                        jhas_latent = latent_codes_eval[j][jhas_idx]
                                        predicted_latent, _ = self.partial_predictors[j](jhas_latent, 0)
                                        latent_codes_eval[i][jhas_idx] += predicted_latent
                            latent_codes_eval[i] = latent_codes_eval[i] / torch.unsqueeze(accumulated, 1)
                        else:

                            missing_idx_eval = mask[:, i] == 0
                            if missing_idx_eval.sum() != 0:
                                core_latent = latent_codes_eval[0][
                                    missing_idx_eval]
                                predicted_latent, _ = self.partial_predictors[0](core_latent, i)
                                latent_codes_eval[i][missing_idx_eval] = predicted_latent

                    latent_fusion = torch.cat(latent_codes_eval, dim=1).cpu().numpy()

                    scores = clustering.get_score(
                        [latent_fusion], Y_list,
                        accumulated_metrics['acc'], accumulated_metrics['nmi'],
                        accumulated_metrics['ARI']
                    )

                    selected_scores = scores['kmeans']
                    current_acc = selected_scores['accuracy']
                    current_nmi = selected_scores['NMI']
                    current_ari = selected_scores['ARI']

                    if current_acc >= best_acc:
                        best_acc = current_acc
                        best_nmi = current_nmi
                        best_ari = current_ari

                    self.train()
        return best_acc, best_nmi, best_ari

