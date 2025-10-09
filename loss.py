import sys
import torch

import torch

def guassian_kernel_mmd(source, target, kernel_mul=2, kernel_num=6, fix_sigma=2**15):
    """计算高斯核矩阵（Gram Kernel Matrix），并返回MMD损失的核矩阵。

    source: source样本，形状为 (sample_size_1, feature_size)
    target: target样本，形状为 (sample_size_2, feature_size)
    kernel_mul: 高斯核带宽的缩放因子
    kernel_num: 核的数量
    fix_sigma: 如果提供了sigma值，则固定带宽
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并source和target, (n_samples, feature_size)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1))) # (n_samples, n_samples, feature_size)
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0 - total1) ** 2).sum(2)  #计算任意两个样本的距离
   # 计算带宽（bandwidth），如果fix_sigma不是None，则把它的值赋值给带宽，否则的话动态计算带宽
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)  # 默认通过L2距离来计算带宽
    bandwidth /= kernel_mul ** (kernel_num // 2)
    # 生成多个带宽值，形成 kernel_num 个不同尺度的高斯核
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    # 计算不同尺度的核函数，并累加，最后返回核矩阵
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)
def MMD(source, target, kernel_mul=2, kernel_num=6, fix_sigma=None):
    """计算最大均值差异（MMD）损失

    source: source样本，形状为 (sample_size_1, feature_size)
    target: target样本，形状为 (sample_size_2, feature_size)
    kernel_mul: 高斯核带宽的缩放因子
    kernel_num: 核的数量
    fix_sigma: 如果提供了sigma值，则固定带宽
    """
    n = int(source.size()[0])
    m = int(target.size()[0])

    # 获取高斯核矩阵 (n_samples, n_samples)，每个元素代表样本i-j之间的相似度
    kernels = guassian_kernel_mmd(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=2**15)

    # 获取kernel矩阵的各部分
    XX = kernels[:n, :n]  # Source <-> Source
    YY = kernels[n:, n:]  # Target <-> Target
    XY = kernels[:n, n:]  # Source <-> Target
    YX = kernels[n:, :n]  # Target <-> Source

    # 对每个部分进行归一化
    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)
    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)

    # 计算最终的MMD损失
    loss = XX.sum() + XY.sum() + YX.sum() + YY.sum()

    return loss

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def instance_contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


