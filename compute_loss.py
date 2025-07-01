import torch
import numpy as np
# from torcheval.metrics.functional import r2_score
def compute_mae(y_true, y_pred):
    """
    计算 MAE (Mean Absolute Error)
    """
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae

def compute_mape(y_true, y_pred):
    """
    计算 MAPE (Mean Absolute Percentage Error)
    """
    mask = (y_true!=0)
    mape = torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / (y_true[mask]))) * 100
    return mape
def RMSE(hat_x, true_x):
    # pos = np.where(true_x != 0)
    return np.sqrt(np.average((true_x - hat_x) ** 2))
# def R_Square(y_true, y_pred):
#     return r2_score(y_pred, y_true)

def R_Square(y_true, y_pred):
    y_true_mean = torch.mean(y_true)

    # 计算分子部分： (y_true - y_pred) 的平方和
    ss_residual = torch.sum((y_true - y_pred) ** 2)

# 计算分母部分： (y_true - y_true_mean) 的平方和
    ss_total = torch.sum((y_true - y_true_mean) ** 2)

# 计算 R²
    r_squared = 1 - (ss_residual / ss_total)
    return r_squared

def kl_loss_function(mu, log_var):
    """
    Calculate the loss. Note that the loss includes two parts.
    :param x_hat:
    :param x:
    :param mu:
    :param log_var:
    :return: total loss, BCE and KLD of our model
    """
    # 1. the reconstruction loss.
    # We regard the MNIST as binary classification
    # print(torch.sum(x))
   

    # 2. KL-divergence
    # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
    # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
    KLD = 0.5 * torch.sum(torch.exp(log_var) + torch.pow(mu, 2) - 1. - log_var)
    # KLD = 0.5 * torch.sum(torch.exp(log_var)  - 1. - log_var)
    # 3. total loss
    # loss = BCE + KLD
    return KLD
# def kl_loss_function(mu, sigma):
#     """
#     Calculate the loss. Note that the loss includes two parts.
#     :param x_hat:
#     :param x:
#     :param mu:
#     :param log_var:
#     :return: total loss, BCE and KLD of our model
#     """
#     # 1. the reconstruction loss.
#     # We regard the MNIST as binary classification
#     # print(torch.sum(x))
   

#     # 2. KL-divergence
#     # D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
#     # here we assume that \Sigma is a diagonal matrix, so as to simplify the computation
#     KLD = 0.5 * torch.sum(sigma**2 + torch.pow(mu, 2) - 1. - 2*torch.log(sigma.clamp(min=1e-8)))

#     # 3. total loss
#     # loss = BCE + KLD
#     return KLD
