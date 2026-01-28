from __future__ import print_function, with_statement, division, absolute_import

import random

import torch
import torchsnooper
from torch import nn, optim
from torch.distributions.normal import Normal
from common import config
import numpy as np
from scipy import sparse as sp
from collections import Counter
import math
import logging
from tqdm import trange, tqdm
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, auc
from risker.pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import LambdaLR

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

device = [0,1]  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection)
class_num = cfg.get_class_num()

LEARN_VARIANCE = cfg.learn_variance
APPLY_WEIGHT_FUNC = cfg.apply_function_to_weight_classifier_output

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(module)s:%(levelname)s] - %(message)s")

class RiskCalibrator:
    def __init__(self, method='platt'):
        if method == 'platt':
            self.calibrator = LogisticRegression()
        elif method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError("Unsupported calibration method: choose 'platt' or 'isotonic'")
    
    def fit(self, probabilities, true_labels):
        self.calibrator.fit(probabilities, true_labels)
    
    def predict(self, probabilities):
        return self.calibrator.predict_proba(probabilities)[:, 1]
        
def preprocess_data(data, nan_replace=0.0, clip_min=None, clip_max=None):
    # Convert to numpy array if it's a PyTorch tensor
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Convert to float64 to handle high precision
    data = data.astype(np.float64)
    
    # Replace NaN values
    data = np.nan_to_num(data, nan=nan_replace)
    
    # Clip values if clip_min and clip_max are provided
    if clip_min is not None and clip_max is not None:
        data = np.clip(data, clip_min, clip_max)
    
    return data
        
def compute_class_weights(labels):
    # Compute the number of samples per class
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)


def swap_sigma_and_mu(y_true, sigma, mu):
    for i in range(y_true.shape[0]):
        class_number = y_true.shape[1]
        if class_number > 1: 
            true_label_index = torch.argmax(y_true[i])
        else:
            true_label_index = y_true[i].item()
        # Swap for sigma (ensure lowest value at true label index)
        min_sigma_index = torch.argmin(sigma[i])
        sigma[i, true_label_index], sigma[i, min_sigma_index] = sigma[i, min_sigma_index].item(), sigma[i, true_label_index].item()
        
        # Swap for mu (ensure highest value at true label index)
        max_mu_index = torch.argmax(mu[i])
        mu[i, true_label_index], mu[i, max_mu_index] = mu[i, max_mu_index].item(), mu[i, true_label_index].item()
    
    return sigma, mu


def my_truncated_normal_ppf(confidence, a, b, mean, stddev):
    x = torch.zeros_like(mean)
    mean = torch.reshape(mean, (-1, 1))
    stddev = torch.reshape(stddev, (-1, 1))
    norm = Normal(mean, stddev)
    _nb = norm.cdf(b)
    _na = norm.cdf(a)
    _sb = 1. - norm.cdf(b)
    _sa = 1. - norm.cdf(a)

    y = torch.where(a > 0,
                    -norm.icdf(confidence * _sb + _sa * (1.0 - confidence)),
                    norm.icdf(confidence * _nb + _na * (1.0 - confidence)))
    return torch.reshape(y, (-1, class_num))

#def my_truncated_normal_ppf(confidence, a, b, mean, stddev):
    #print('confidence')
    #print(confidence)
    #print('a')
    #print(a)
    #print('b')
    #print(b)
    #print('mean')
    #print(mean)
    #print('stddev')
    #print(stddev)
#    x = torch.zeros_like(mean)
#    mean = torch.reshape(mean, (-1, 1))
#   stddev = torch.reshape(stddev, (-1, 1))
#    norm = Normal(mean, stddev)
#    _nb = norm.cdf(b)
#    _na = norm.cdf(a)
#    _sb = 1. - norm.cdf(b)
#    _sa = 1. - norm.cdf(a)

#    y = torch.where(a > 0,
#                    -norm.icdf(confidence * _sb + _sa * (1.0 - confidence)),
#                    norm.icdf(confidence * _nb + _na * (1.0 - confidence)))
#    print(y)
#    return torch.reshape(y, (-1, class_num))


def gaussian_function(a, b, c, x):
    _part = (- torch.div((x - b) ** 2, 2.0 * (c ** 2)))
    _f = -torch.exp(_part) + a + 1.0
    return _f


def L1loss(model, beta=0.001):
    l1_loss = torch.tensor(0., requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l1_loss = l1_loss + beta * torch.sum(torch.abs(parma)).to(device[0])
    return l1_loss.to(device[0])


def L2loss(model, alpha=0.001):
    l2_loss = torch.tensor(0., requires_grad=True)
    for name, parma in model.named_parameters():
        if 'bias' not in name:
            l2_loss = l2_loss + (0.5 * alpha * torch.sum(parma ** 2)).to(device[0])
    return l2_loss.to(device[0])




class RiskModel(nn.Module):

    def __init__(self, rule_m, max_w, class_num, input_dim, true_label, init_variance=None, dropout_rate=0.5, update_rate=0.1):
        super(RiskModel, self).__init__()
        self.max_w = max_w
        alaph = torch.tensor(cfg.risk_confidence, dtype=torch.float32)
        a = torch.tensor(0., dtype=torch.float32)
        b = torch.tensor(1., dtype=torch.float32)

        # weight_func_a = torch.tensor(0.5, dtype=torch.float32)
        # weight_fun_b = torch.tensor([0.5] * class_num, dtype=torch.float32)
        weight_fun_b = torch.tensor([0.5], dtype=torch.float32)
        # weight_func_c = torch.tensor(0.5, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout_rate)
        self.class_weights = compute_class_weights(true_label)
        self.register_buffer('alaph', alaph)
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('weight_fun_b', weight_fun_b)

        self.rule_m = rule_m
        self.machine_m = cfg.interval_number_4_continuous_value

        self.rule_var = nn.Parameter(
            torch.empty((1, self.rule_m,), dtype=torch.float32, requires_grad=True)
        )
        torch.nn.init.uniform_(self.rule_var, 0, 1)
        self.machine_var = nn.Parameter(
            torch.empty(1, self.machine_m, dtype=torch.float32, requires_grad=True)
        )
        torch.nn.init.uniform_(self.machine_var, 0, 1)
        # self.machine_var = nn.Parameter(
        #     torch.empty(self.machine_m, dtype=torch.float32, requires_grad=True)
        # )
        # torch.nn.init.uniform_(self.machine_var, a=0.1, b=0.9)

        self.rule_w = nn.Parameter(
            torch.empty((1, self.rule_m,), dtype=torch.float32, requires_grad=True)
        )
        torch.nn.init.uniform_(self.rule_w, 0, max_w)

        # self.machine_w = nn.Parameter(
        #     torch.empty((class_num, self.machine_m,), dtype=torch.float32)
        # )
        # torch.nn.init.uniform(self.machine_w, a=0, b=max_w)

        self.learn2rank_sigma = nn.Parameter(torch.tensor(1., dtype=torch.float32, requires_grad=True))
        self.risk_weight_learn = nn.Parameter(torch.tensor(0.1, dtype=torch.float32, requires_grad=True))
        # self.weight_fun_a = nn.Parameter(torch.tensor([1.] * class_num, dtype=torch.float32, requires_grad=True))
        # self.weight_fun_c = nn.Parameter(torch.tensor([0.5] * class_num, dtype=torch.float32, requires_grad=True))
        self.weight_fun_a = nn.Parameter(torch.tensor([1.], dtype=torch.float32, requires_grad=True))
        self.weight_fun_c = nn.Parameter(torch.tensor([0.5], dtype=torch.float32, requires_grad=True))
        # Attention mechanism
        #self.attention = nn.Linear(input_dim, class_num)
        self.attention = nn.Linear(input_dim, class_num)
        self.mean_adjustment = nn.Parameter(torch.zeros(class_num))  # Adaptive mean
        # self.register_parameter('rule_vars', self.rule_var)
        # self.register_parameter('machine_var', self.machine_var)
        # self.register_parameter('rule_w', self.rule_w)
        # self.register_parameter('learn2rank_sigma', self.learn2rank_sigma)
        # self.register_parameter('weight_fun_a', self.weight_fun_a)
        # self.register_parameter('weight_fun_c', self.weight_fun_c)

    def forward(self, machine_labels, rule_mus, machine_mus, rule_feature_matrix, machine_feature_matrix,
                machine_one, y_risk, y_mul_risk, init_rule_mu, init_machine_mu):
        
        #print('labels')
        #print(machine_labels)
        # rule_w = self.rule_w.clamp(0, )
        # rule_var = self.rule_var.clamp(0., 1.)
        # machine_var = self.machine_var.clamp(0., 1.)
        # weight_fun_a = self.weight_fun_a.clamp(1e-10, )
        # weight_fun_c = self.weight_fun_c.clamp(1e-10, )
        # rule_w = torch.relu(self.rule_w)
        # rule_var = torch.sigmoid(self.rule_var)
        # machine_var = torch.sigmoid(self.machine_var)
        # weight_fun_a = torch.relu(self.weight_fun_a)
        # weight_fun_c = torch.relu(self.weight_fun_c)
        #print(machine_mus)
        machine_mus_vector = torch.reshape(torch.sum(machine_mus, 2), (-1, class_num))
        #print(machine_mus_vector)
        # machine_w = gaussian_function(self.weight_fun_a, self.weight_fun_b, self.weight_fun_c,
        #                               machine_mus_vector.reshape((-1, class_num)))
        # if self.weight_fun_a < 0.9:
        #     self.weight_fun_a[0] = self.weight_fun_a[0]+(0.9-self.weight_fun_a[0])
        # if self.weight_fun_b < 0.4:
        #     self.weight_fun_b[0] = self.weight_fun_b[0]+(0.4-self.weight_fun_b[0])
        # if self.weight_fun_c < 0.4:
        #     self.weight_fun_c[0] = self.weight_fun_c[0]+(0.4-self.weight_fun_c[0])
        scores = self.attention(machine_mus)
        machine_w = torch.softmax(scores, dim=-1)
        #print(machine_w)
        machine_w = machine_w[:, :, 0]
        #print(machine_w)
        #machine_w = gaussian_function(self.weight_fun_a, self.weight_fun_b, self.weight_fun_c,
        #                             machine_mus_vector.reshape((-1, 1)))
        # print(self.weight_fun_c)
        # print('---funa')
        # print(self.weight_fun_a)
        # print('----func')
        # print(self.weight_fun_c)
        #machine_w = machine_w.reshape((-1, class_num))
        # print('mac_w')
        # print(machine_w)
        # print(self.weight_fun_a, self.weight_fun_b, self.weight_fun_c)
        # print('---------------a')
        # print(self.weight_fun_a)
        # print('--------------c')
        # print(self.weight_fun_c)
        # è®¡ç® big_mu
        # r_mu = torch.sum(rule_mus * self.rule_w, 2)
        # mac_mu = machine_mus_vector * machine_w
        # print(torch.sum(rule_mus * self.rule_w, 2)/4)
        big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * machine_w + 1e-10
        
        risk_weight_learn = self.risk_weight_learn
        #print(risk_weight_learn)
        #print('big_mu before')
        #print(big_mu.shape)
        #print(self.mean_adjustment)
        #big_mu = self.dropout(big_mu)
        big_mu = big_mu + self.mean_adjustment
        #print('big_mu after')
        #print(big_mu)
        
        m_m= machine_mus_vector * machine_w
        # big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * torch.sum(rule_feature_matrix * self.rule_w, 2) + 1e-10
        # print('mu')
        # print(big_mu)

        # RSD
        
        new_rule_mus = torch.from_numpy(init_rule_mu).clone().to(device[0])
        new_mac_mus = torch.from_numpy(init_machine_mu).clone().to(device[0])
        new_rule_mus = new_rule_mus.float()
        new_mac_mus = new_mac_mus.float()
        new_rule_mus[torch.where(new_rule_mus < 0.1)] = 0.1
        new_mac_mus[torch.where(new_mac_mus < 0.1)] = 0.1
        rule_standard_deviation = new_rule_mus * self.rule_var
        mac_standard_deviation = new_mac_mus * self.machine_var
        rule_var = rule_standard_deviation ** 2
        machine_var = mac_standard_deviation ** 2
        
        
        # print(rule_var.shape)
        # print(machine_var.shape)
        # print(rule_var)
        # print(machine_var)
        # è®¡ç® big_sigma
        rule_sigma = rule_feature_matrix * rule_var
        machine_sigma = machine_feature_matrix * machine_var
        machine_sigma_vector = torch.sum(machine_sigma, 2).reshape((-1, class_num))

        # r_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2)
        # mac_sigma = machine_sigma_vector * (machine_w ** 2)
        big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (machine_w ** 2) + 1e-10
        r_sig=torch.sum(rule_sigma * (self.rule_w ** 2), 2)
        m_sig=machine_sigma_vector * (machine_w ** 2)
        # big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (torch.sum(rule_feature_matrix * self.rule_w, 2) ** 2) + 1e-10
        # print('--sigma')
        # print(big_sigma)
        # è®¡ç®æé
        r_w = self.rule_w
        r_all_w = torch.sum(rule_feature_matrix * self.rule_w, 2)
        # print(r_all_w[0][0:10])
        # print('----------------')
        # print(machine_w[0][0:10])
        weight_vector = torch.sum(rule_feature_matrix * self.rule_w, 2) + machine_w + 1e-10
        #print('weights')
        #print(weight_vector)
        #weight_vector = torch.sum(rule_feature_matrix * self.rule_w, 2) * 2 + 1e-10
        big_mu = big_mu / (weight_vector + 1e-10)
        big_sigma = big_sigma / (weight_vector ** 2 + 1e-10)
        #big_sigma, big_mu = swap_sigma_and_mu(machine_labels, big_sigma, big_mu)
        # print('------norm')
        # print(big_mu)
        # print('---')
        # print(big_sigma)
        machine_one = machine_one.float()
        if torch.isnan(big_mu).any():
            print(1)
        # print(big_mu)
        # print(torch.sqrt(big_sigma))
        ## Big MU is the expectation, and big sigma is the variance
        Fr_alpha = my_truncated_normal_ppf(self.alaph, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alaph, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        # print(Fr_alpha)
        # print(Fr_alpha_bar)
        # prob_mul = Fr_alpha * (torch.ones_like(machine_one) - machine_one) + (
        #         torch.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_one
        #risk_weight_learn = self.risk_weight_learn
        #print(risk_weight_learn)
        #print(risk_weight_learn)
        prob_mul = 1 - Fr_alpha_bar
        #print(prob_mul)
        
        true_risk=torch.sum(prob_mul*machine_labels,1)
        #print(machine_w.cuda())
        #weighted_attention = machine_w.cuda() * self.class_weights.cuda()
        weighted_attention = machine_w.cuda() 
        prob_mul = prob_mul * weighted_attention
        #print(prob_mul)
        #print(machine_one)
        prob = torch.sum(prob_mul * machine_one, 1)
        #print(machine_one)
        #print(prob)
        prob = torch.reshape(prob, (-1, 1))
        #print(prob.shape)
        # np.save('rule_mu', r_mu.cpu().detach().numpy())
        # np.save('mac_mu', mac_mu.cpu().detach().numpy())
        # np.save('rule_sigma', r_sigma.cpu().detach().numpy())
        # np.save('mac_sigma', mac_sigma.cpu().detach().numpy())
        # np.save('rule_w', r_w.cpu().detach().numpy())
        # np.save('rule_all_w', r_all_w.cpu().detach().numpy())
        # np.save('mac_w', machine_w.cpu().detach().numpy())
        # np.save('big_mu.npy', big_mu.cpu().detach().numpy())
        # np.save('big_sigma.npy', big_sigma.cpu().detach().numpy())
        # np.save('Fr.npy', Fr_alpha.cpu().detach().numpy())
        # np.save('Fr_bar.npy', Fr_alpha_bar.cpu().detach().numpy())
        # print(self.weight_fun_a.data, self.weight_fun_b.data, self.weight_fun_c.data)
        return prob_mul, prob, [self.weight_fun_a.data, self.weight_fun_b.data, self.weight_fun_c.data], \
               self.rule_w.data, rule_var.data, machine_var.data, big_mu, big_sigma, r_all_w, m_m, [Fr_alpha, Fr_alpha_bar],r_sig,m_sig




class PairwiseLoss(nn.Module):
    def __init__(self, learn2rank_sigma, risk_weight_learn):
        super(PairwiseLoss, self).__init__()
        self.learn2rank_sigma = learn2rank_sigma
        self.result = torch.empty((0, 2), dtype=torch.float32)
        self.init_result = self.result

    # @torchsnooper.snoop()
    def forward(self, input, target):
        #print('input')
        #print(input)
        #print('target')
        #print(target)
        pairwise_probs = self.get_pairwise_combinations(input).to(device[0])
        # print(pairwise_probs.shape)
        pairwise_labels = self.get_pairwise_combinations(target.float()).to(device[0])
        # print(pairwise_labels.shape)

        p_target_ij = 0.5 * (1.0 + pairwise_labels[:, 0] - pairwise_labels[:, 1])
        o_ij = pairwise_probs[:, 0] - pairwise_probs[:, 1]

        diff_label_indices = torch.nonzero(p_target_ij != 0.5)  # .squeeze()
        # print(diff_label_indices.shape)
        new_p_target_ij = p_target_ij[diff_label_indices]
        # print(new_p_target_ij.shape)
        new_o_ij = o_ij[diff_label_indices] * self.learn2rank_sigma
        #app = torch.sum(- new_p_target_ij * new_o_ij + torch.log(1.0 + torch.exp(new_o_ij))).to(device[0])
        #print('hello')
        #print(app)
        # print(self.learn2rank_sigma)
        return torch.sum(- new_p_target_ij * new_o_ij + torch.log(1.0 + torch.exp(new_o_ij))).to(device[0])

    # @torchsnooper.snoop()
    def get_pairwise_combinations(self, input):
        self.result = self.init_result
        for i in range(input.shape[0] - 1):
            tensor = torch.stack(
                torch.meshgrid(input[i, 0], input[i + 1:, 0]), dim=-1
            ).reshape((-1, 2))
            self.result = torch.cat((self.result.to(device[0]), tensor.to(device[0])), dim=0)

        return self.result



def train(model, val, test, init_rule_mu, init_machine_mu, epoch_cnn=0, epoches=20,
          suffle_data=True, shots_per_class=None):  # ADDED: shots_per_class parameter
    # rule_m = val._rule_mus.shape[1]
    # machine_m = cfg.interval_number_4_continuous_value
    # val, test = test, val
    #print(val.risk_labels)
    
    # ============== FEW-SHOT SAMPLING ==============
    if shots_per_class is not None:
        print(f"=== FEW-SHOT MODE: Selecting {shots_per_class} samples per class ===")
        
        # Get all indices
        all_indices = np.arange(len(val.risk_labels))
        
        # Group indices by true class labels
        class_to_indices = {}
        for idx in all_indices:
            true_label = val.true_labels[idx]
            if true_label not in class_to_indices:
                class_to_indices[true_label] = []
            class_to_indices[true_label].append(idx)
        
        # Select K samples per class
        selected_indices = []
        for class_idx in sorted(class_to_indices.keys()):
            class_indices = class_to_indices[class_idx]
            if len(class_indices) >= shots_per_class:
                selected = np.random.choice(class_indices, shots_per_class, replace=False)
            else:
                selected = class_indices
                print(f"  Class {class_idx}: Only {len(class_indices)} samples available")
            selected_indices.extend(selected)
        
        selected_indices = np.array(selected_indices)
        print(f"  Total selected samples: {len(selected_indices)} (expected: {shots_per_class} x {len(class_to_indices)} = {shots_per_class * len(class_to_indices)})")
        
        # Create a mask for selected indices
        selected_mask = np.zeros(len(val.risk_labels), dtype=bool)
        selected_mask[selected_indices] = True
    else:
        # Use all data (original behavior)
        selected_mask = np.ones(len(val.risk_labels), dtype=bool)
        print("=== USING ALL AVAILABLE DATA ===")
    
    # ============== ORIGINAL CODE WITH FEW-SHOT FILTERING ==============
    data_len = np.sum(selected_mask)  # Use only selected samples
    
    # random.seed(2020)
    # np.random.seed(2020)
    # torch.manual_seed(2020)
    if epoch_cnn > 1:
        epoches = 10
    #print(val.get_risk_mean_X_discrete())
    
    # Apply selected_mask to all data arrays
    machine_labels = torch.tensor(val.machine_labels[selected_mask].reshape([data_len, 1]), dtype=torch.int) # n * 1
    rule_mus = torch.tensor(val.get_risk_mean_X_discrete()[selected_mask], dtype=torch.float32) # n * class_num * m 
    #print(rule_mus)
    machine_mus = torch.tensor(val.get_risk_mean_X_continue()[selected_mask], dtype=torch.float32)# n * class_num
    rule_feature_activate = torch.tensor(val.get_rule_activation_matrix()[selected_mask], dtype=torch.float32) # n * class_num * m
    machine_feature_activate = torch.tensor(val.get_prob_activation_matrix()[selected_mask], dtype=torch.float32) # n * class_num
    machine_one = torch.tensor(val.machine_label_2_one[selected_mask].reshape([data_len, class_num]), dtype=torch.float32) 
    y_risk = torch.tensor(val.risk_labels[selected_mask].reshape([data_len, 1]), dtype=torch.float32) 
    y_mul_risk = torch.tensor(val.risk_mul_labels[selected_mask].reshape([data_len, class_num]), dtype=torch.float32)
    y_true_one = torch.tensor(val.true_label_2_one[selected_mask].reshape([data_len, class_num]), dtype=torch.float32)
    
    risk_weight = 0
    learning_rate = 0.00005
    l2_reg = 0.001
    bs = 32
    batch_num = data_len // bs + (1 if data_len % bs else 0)

    for name, p in model.named_parameters():
        print(name)

    criterion = PairwiseLoss(model.learn2rank_sigma, model.risk_weight_learn).to(device[0])
    loss2 = torch.nn.BCELoss()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # lambda2 = lambda epoch: 0.99 ** (epoch // 1)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda2)

    
    early_stopping = EarlyStopping(patience=100, verbose=True)

    for epoch in tqdm(range(epoches), desc="train of each class"):
        # scheduler.step()
        # print(scheduler.get_lr()[0])
        model.train()
        if suffle_data:
            index = np.random.permutation(np.arange(data_len))
            # print(index)
        else:
            index = np.arange(data_len)
        accuracy = 0.
        loss_total = 0.
        n_total = 0.
        outputs_all = torch.empty((0, 1), dtype=torch.float32, requires_grad=False)
        outputs_mul_all = torch.empty((0, class_num), dtype=torch.float32, requires_grad=False)
        right = 0
        tot = 0
        fr_right = 0
        r_right=0
        m_right=0
        # if (epoch_cnn + 1) % 3 == 1:
        #     b_stop = batch_num / 3
        #     b_sta = 0
        # if (epoch_cnn + 1) % 3 == 2:
        #     b_stop = (batch_num / 3) * 2
        #     b_sta = batch_num / 3
        # if (epoch_cnn + 1) % 3 == 0:
        #     b_stop = batch_num
        #     b_sta = (batch_num / 3) * 2
        # for i in range(batch_num):
        # print(int(b_sta),int(b_stop))
        print(batch_num)
        #for i in range(1,2):
        for i in range(batch_num):

            machine_labels_batch = machine_labels[index][bs * i: bs * i + bs].to(device[0])
            rule_mus_batch = rule_mus[index][bs * i: bs * i + bs].to(device[0])
            machine_mus_batch = machine_mus[index][bs * i: bs * i + bs].to(device[0])
            rule_feature_activate_batch = rule_feature_activate[index][bs * i: bs * i + bs].to(device[0])
            machine_feature_activate_batch = machine_feature_activate[index][bs * i: bs * i + bs].to(device[0])
            machine_one_batch = machine_one[index][bs * i: bs * i + bs].to(device[0])
            y_risk_batch = y_risk[index][bs * i: bs * i + bs].to(device[0])
            y_mul_risk_batch = y_mul_risk[index][bs * i: bs * i + bs].to(device[0])
            y_true_one_batch=y_true_one[index][bs * i: bs * i + bs].to(device[0])
            #print(y_true_one_batch)
            outputs_mul, outputs, func_params, rule_w, rule_var, machine_var, big_mu, big_sigma, r_w, m_w, fr,_,_ = model(y_true_one_batch,
                                                                                     rule_mus_batch,
                                                                                     machine_mus_batch,
                                                                                     rule_feature_activate_batch,
                                                                                     machine_feature_activate_batch,
                                                                                     machine_one_batch,
                                                                                     y_risk_batch,
                                                                                     y_mul_risk_batch,
                                                                                     init_rule_mu,
                                                                                     init_machine_mu,)
            
            #print('big_sigma')
            #print(big_sigma)
            #print('big_mu')
            #print(big_mu)
            #print('y_true_one_batch')
            #print(y_true_one_batch)
            
            
            #print('Updated big_sigma')
            #print(big_sigma)
            #print('Updated big_mu')
            #print(big_mu)

            l1_loss = torch.tensor(0.).to(device[0])
            l2_loss = torch.tensor(0.).to(device[0])
            beta = torch.tensor(0.001).to(device[0])

            # l1, l2 æ­£åï¼?æ¹å·®ä¸å¤ç?
            for name, param in model.named_parameters():
                if 'var' not in name:
                    l1_loss += beta * torch.sum(torch.abs(param))
                    l2_loss += (0.5 * beta * torch.sum(param ** 2))
            #print('output')
            #print(outputs)
            #print('y_label')
            #print(y_risk_batch)
            optimizer.zero_grad()
            # print('output')
            # print(outputs.reshape((-1, 1)))
            # print('y_label')
            # print(y_risk_batch.reshape((-1, 1)))
            rank_loss1 = criterion(outputs.reshape((-1, 1)), y_risk_batch.reshape((-1, 1)))
            # rank_loss2 = criterion(outputs_mul.reshape((-1, 1)), y_mul_risk_batch.reshape((-1, 1)))
            loss = rank_loss1 + l1_loss + l2_loss
            #loss = torch.tensor(0, dtype=torch.float32).to(device[0])
            #for j in range(class_num):
            #    loss += criterion(outputs_mul[:, j].reshape((-1, 1)), y_mul_risk_batch[:, j].reshape((-1, 1)))
            #loss += l1_loss + l2_loss

            # loss = rank_loss + l2_loss + l1_loss
            # print(rank_loss.item(), l1_loss.item(), l2_loss.item())
            # loss = criterion(outputs.reshape((-1, 1)), y_risk_batch.reshape((-1, 1))) + l2_loss + l1_loss
            # loss = l2_loss + l1_loss
            # for j in range(class_num):
            #     loss = criterion(outputs_mul[:, j].reshape((-1, 1)), y_mul_risk_batch[:, j].reshape((-1, 1)))
            #     print(loss.item())
            #     loss.backward()
            # optimizer.step()

            # loss = loss2(outputs_mul.reshape((-1, 1)), y_mul_risk_batch.reshape((-1, 1))) + l2_loss + l1_loss
            # loss = criterion(outputs.reshape((-1, 1)), y_risk_batch.reshape((-1, 1)))
            # loss = torch.tensor(0, dtype=torch.float32).to(device[0])
            # for j in range(class_num):
            #     loss += criterion(outputs_mul[:, j].reshape((-1, 1)), y_mul_risk_batch[:, j].reshape((-1, 1)))
            # loss += l1_loss + l2_loss
            loss.backward()
            optimizer.step()
            # # model.module.rule_w.data.clamp_(0., )
            # model.module.rule_var.data.clamp_(0., 1.)
            # model.module.machine_var.data.clamp_(0., 1.)
            # model.module.weight_fun_a.data.clamp_(0., )
            # model.module.weight_fun_c.data.clamp_(0., )

            # æªæ­ï¼?ä¿è¯å¶çå®æä¹?
            model.rule_w.data.clamp_(0., )
            model.rule_var.data.clamp_(0., 1.)
            model.machine_var.data.clamp_(0., 1.)
            model.weight_fun_a.data.clamp_(1e-10, )
            model.weight_fun_c.data.clamp_(1e-10, )

            loss_total += loss.item()
            n_total += len(outputs_mul)
            # logging.info("[%d][%d/%d], loss=%f" % (epoch, i, batch_num, loss.item()))
            outputs_all = torch.cat((outputs_all.to(device[0]), outputs), dim=0)
            # print('outputs_all')
            #print(outputs_all)
            outputs_mul_all = torch.cat((outputs_mul_all.to(device[0]), outputs_mul), dim=0)
            #print('big mu')
            #print(big_mu)
            #print('fr')
            #print(fr[0])
            #print('rw')
            #print(r_w)
            #print('mw')
           
            right += torch.sum(torch.argmax(big_mu, 1).reshape(-1, 1) == torch.tensor(val.true_labels[selected_mask][index][bs * i: bs * i + bs], dtype=torch.float32).reshape((-1, 1)).to(device[0]))
            tot += len(big_mu)
            fr_right += torch.sum(
                torch.argmax(fr[0], 1).reshape(-1, 1) == torch.tensor(val.true_labels[selected_mask][index][bs * i: bs * i + bs],
                                                                      dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))
            r_right += torch.sum(
                torch.argmax(r_w, 1).reshape(-1, 1) == torch.tensor(val.true_labels[selected_mask][index][bs * i: bs * i + bs],
                                                                    dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))
            m_right += torch.sum(
                torch.argmax(m_w, 1).reshape(-1, 1) == torch.tensor(val.true_labels[selected_mask][index][bs * i: bs * i + bs],
                                                                    dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))
        
            
        print("racc:", float(r_right) / float(tot))
        print("macc:", float(m_right) / float(tot))
            # print(right, fr_right)
            # print(outputs_mul_all)
        scheduler.step()

        # # è®¡ç®roc
        # _machine_mul_pro = np.abs(val.machine_label_2_one - val.machine_mul_probs).reshape(-1, 1)
        # fpr, tpr, _ = roc_curve(val.risk_mul_labels.reshape((-1)), _machine_mul_pro.reshape((-1)))
        # baseline_mul_roc_auc = auc(fpr, tpr)
        # baseline_mul_roc_auc = baseline_mul_roc_auc * 100

        # _machine_pro = 1 - val.machine_probs.reshape((-1, 1))
        # fpr, tpr, _ = roc_curve(val.risk_labels, _machine_pro.reshape((-1)))
        # baseline_roc_auc = auc(fpr, tpr)
        # baseline_roc_auc = baseline_roc_auc * 100
        # logging.info(
        #     "epoch=%d, baseline loss=%f val_mul auc = %f val auc = %f" % (epoch, loss_total, baseline_mul_roc_auc, baseline_roc_auc))
        #
        # fpr, tpr, _ = roc_curve(val.risk_mul_labels[index].reshape((-1, 1)),
        #                         outputs_mul_all.reshape((-1, 1)).cpu().detach().numpy())
        # val_mul_roc_auc = auc(fpr, tpr)
        # fpr, tpr, _ = roc_curve(val.risk_labels[index].reshape((-1)), outputs_all.reshape((-1)).cpu().detach().numpy())
        #
        # early_stopping(loss_total, model)
        #
        # val_roc_auc = auc(fpr, tpr)
        # logging.info("epoch=%d, loss=%f val_mul auc = %f val auc = %f" % (epoch, loss_total, val_mul_roc_auc, val_roc_auc))
        # acc = float(right) / float(tot)
        # fr_acc = float(fr_right) / float(tot)
        # logging.info("val right {} tot {} acc {:.4f}".format(right, tot, acc))
        # logging.info("val fr_right {} tot {} acc {:.4f}".format(fr_right, tot, fr_acc))
        logging.info("loss=%f" % (loss_total))
        if epoch % 1 == 0:
            # print('---------------val predict')
            # predict(model, val, epoch, init_rule_mu,  init_machine_mu, epoch_cnn,)
            print('-----------------test predict')
            predict(model, test, epoch, init_rule_mu, init_machine_mu, epoch_cnn, True)

        if early_stopping.early_stop:
            logging.info('early stopping')
            break

    return func_params, rule_w, rule_var, machine_var


def predict(model, test, epoch, init_rule_mu, init_machine_mu, epoch_cnn=0, is_print=False):
    data_len = len(test.risk_labels)

    machine_labels = torch.tensor(test.machine_labels.reshape([test.data_len, 1]), dtype=torch.int)
    
    # �?    
    rule_mus = torch.tensor(test.get_risk_mean_X_discrete(), dtype=torch.float32)
    machine_mus = torch.tensor(test.get_risk_mean_X_continue(), dtype=torch.float32)
    rule_feature_activate = torch.tensor(test.get_rule_activation_matrix(), dtype=torch.float32)
    machine_feature_activate = torch.tensor(test.get_prob_activation_matrix(), dtype=torch.float32)
    machine_one = torch.tensor(test.machine_label_2_one.reshape([data_len, class_num]), dtype=torch.int)
    y_risk = torch.tensor(test.risk_labels.reshape([data_len, 1]), dtype=torch.int)
    y_mul_risk = torch.tensor(test.risk_mul_labels.reshape([data_len, class_num]), dtype=torch.int)

    bs = 4
    batch_num = data_len // bs + (1 if data_len % bs else 0)
    outputs_all = torch.empty((0, 1), dtype=torch.long)
    outputs_mul_all = torch.empty((0, class_num), dtype=torch.long)
    '''
    big_mu_all = torch.empty((0, class_num), dtype=torch.float32)
    big_sigma_all = torch.empty((0, class_num), dtype=torch.float32)
    fr_all = torch.empty((0, class_num), dtype=torch.float32)
    fr_bar_all = torch.empty((0, class_num), dtype=torch.float32)
    r_w_all = torch.empty((0, class_num), dtype=torch.float32)
    m_w_all = torch.empty((0, class_num), dtype=torch.float32)
    r_every_w_all = None#torch.empty((0, class_num), dtype=torch.float32)
    '''
    model.eval()
    right = 0
    right_fr = 0
    right_fr_bar = 0
    right_entropy = 0
    tot = 0
    tot_2 = 0
    rs_right=0
    ms_right=0
    with torch.no_grad():
        for i in range(batch_num):
            machine_labels_batch = machine_labels[bs * i: bs * i + bs].to(device[0])
            rule_mus_batch = rule_mus[bs * i: bs * i + bs].to(device[0])
            machine_mus_batch = machine_mus[bs * i: bs * i + bs].to(device[0])
            rule_feature_activate_batch = rule_feature_activate[bs * i: bs * i + bs].to(device[0])
            machine_feature_activate_batch = machine_feature_activate[bs * i: bs * i + bs].to(device[0])
            machine_one_batch = machine_one[bs * i: bs * i + bs].to(device[0])
            y_risk_batch = y_risk[bs * i: bs * i + bs].to(device[0])
            y_mul_risk_batch = y_mul_risk[bs * i: bs * i + bs].to(device[0])

            outputs_mul, outputs, _, r_every_w, r_var, m_var, big_mu, big_sigma, r_w, m_w, fr,r_s,m_s = model(machine_labels_batch, rule_mus_batch, machine_mus_batch,
                                                     rule_feature_activate_batch, machine_feature_activate_batch,
                                                     machine_one_batch,
                                                     y_risk_batch, y_mul_risk_batch, init_rule_mu, init_machine_mu)
                                                     
                                                     # �?                                                     
            # logging.info(torch.reshape(outputs_mul, (-1, class_num)).shape)
            outputs_mul_all = torch.cat((outputs_mul_all.to(device[0]), torch.reshape(outputs_mul, (-1, class_num))),
                                        dim=0)
            # logging.info(outputs_mul_all.shape)
            # output = torch.reshape(torch.sum(torch.reshape(outputs, (-1, class_num)) * y_activate_batch, 1), (-1, 1))

            outputs_all = torch.cat((outputs_all.to(device[0]), outputs), dim=0)
            '''
            big_mu_all = torch.cat((big_mu_all, big_mu.cpu()), dim=0)
            big_sigma_all = torch.cat((big_sigma_all, big_sigma.cpu()), dim=0)
            fr_all = torch.cat((fr_all, fr[0].cpu()), dim=0)
            fr_bar_all = torch.cat((fr_bar_all, fr[1].cpu()), dim=0)
            r_w_all = torch.cat((r_w_all, r_w.cpu()), dim=0)
            m_w_all = torch.cat((m_w_all, m_w.cpu()), dim=0)
            r_every_w_all = r_every_w.cpu()
            '''
            right += torch.sum(
                torch.argmax(big_mu, 1).reshape(-1, 1) == torch.tensor(test.true_labels[bs * i: bs * i + bs], dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))
            right_fr += torch.sum(
                torch.argmax(fr[0], 1).reshape(-1, 1) == torch.tensor(test.true_labels[bs * i: bs * i + bs],
                                                                       dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))

            right_fr_bar += torch.sum(
                torch.argmin(1 - fr[1], 1).reshape(-1, 1) == torch.tensor(test.true_labels[bs * i: bs * i + bs],
                                                                      dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))
            rs_right += torch.sum(
                torch.argmax(r_s, 1).reshape(-1, 1) == torch.tensor(test.true_labels[bs * i: bs * i + bs],
                                                                    dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))
            ms_right += torch.sum(
                torch.argmax(m_s, 1).reshape(-1, 1) == torch.tensor(test.true_labels[bs * i: bs * i + bs],
                                                                    dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))


            tot += len(big_mu)
        torch.cuda.empty_cache()
    
    # Calibrate the outputs
    #('output all before')
    #print(outputs_all)
    calibrator = RiskCalibrator(method='platt')  # or 'isotonic'
    outputs_all= np.nan_to_num(outputs_all.cpu().numpy(), nan=0.0)
    calibrator.fit(outputs_all, test.risk_labels)
    outputs_all = calibrator.predict(outputs_all)
    #('output all after')
    #print(outputs_all)
    #print('risk_mul_labels')
    #print(test.risk_mul_labels.reshape((-1)))
    if (np.isnan(outputs_mul_all.reshape((-1)).cpu().numpy()).any()):
        print(outputs_mul_all.reshape((-1)).cpu().numpy())
    else:print("Fale")
    outputs_mul_all = preprocess_data(outputs_mul_all)
    risk_labels = preprocess_data(test.risk_mul_labels)
    fpr, tpr, _ = roc_curve(risk_labels.reshape((-1)), outputs_mul_all.reshape((-1)))
    risk_mul_roc_auc = auc(fpr, tpr)
    risk_mul_roc_auc = risk_mul_roc_auc * 100
    #print('machine mul prob')
    #print(test.machine_mul_probs)
    #print('Risk Labels')
    #print(test.risk_labels)
    #print('output all')
    #print(outputs_all)
    #print('test.machine_probs')
    #print(test.machine_probs)
    # risk --> abs(mac_label - mac_prob)
    _machine_mul_pro = (1 - test.machine_mul_probs).reshape(-1, 1)
    fpr, tpr, _ = roc_curve(test.risk_mul_labels.reshape((-1)), _machine_mul_pro.reshape((-1)))
    baseline_mul_roc_auc = auc(fpr, tpr)
    baseline_mul_roc_auc = baseline_mul_roc_auc * 100

    fpr, tpr, _ = roc_curve(test.risk_labels, outputs_all.reshape((-1)))
    risk_roc_auc = auc(fpr, tpr)
    risk_roc_auc = risk_roc_auc * 100

    _machine_pro = 1 - test.machine_probs.reshape((-1, 1))
    fpr, tpr, _ = roc_curve(test.risk_labels, _machine_pro.reshape((-1)))
    baseline_roc_auc = auc(fpr, tpr)
    baseline_roc_auc = baseline_roc_auc * 100
    '''
    root = './results/CAR/'
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_outputs_mul_all', outputs_mul_all.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_machine_label', np.array(test.machine_labels))
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_test_label', np.array(test.true_labels))
    
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_big_mu', big_mu_all.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_big_sigma', big_sigma_all.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_test_fr', fr_all.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_test_fr_bar', fr_bar_all.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_test_r_w', r_w_all.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_test_m_w', m_w_all.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_test_label', np.array(test.true_labels))
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_machine_label', np.array(test.machine_labels))
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_r_var', r_var.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_m_var', m_var.cpu().numpy())
    np.save(root + str(epoch_cnn) + " " + str(epoch) + '_every_r_w', r_every_w_all.cpu().numpy())
    '''
    logging.info("risk mul_roc : %f, risk_mul_baseline : %f \n risk roc : %f,  baseline roc %f," % (risk_mul_roc_auc,
                                                                                                    baseline_mul_roc_auc,risk_roc_auc, baseline_roc_auc))
    # print(torch.argmax(big_mu, 1).shape)

    acc = float(right) / float(tot) * 100.0
    fr_acc = float(right_fr) / float(tot) * 100.0
    fr_bar_acc = float(right_fr_bar) / float(tot) * 100.0
    entropy_acc = float(right_entropy) / float(tot) * 100.0
    rs_acc = float(rs_right) / float(tot) * 100.0
    ms_acc = float(ms_right) / float(tot) * 100.0
    print('big_mu right ={} tot = {}  acc = {:.4f}'.format(right, tot, acc))
    print('fr right ={} tot = {}  acc = {:.4f}'.format(right_fr, tot, fr_acc))
    print('fr_bar right ={} tot = {}  acc = {:.4f}'.format(right_fr_bar, tot, fr_bar_acc))
    print('rs_acc right ={} tot = {}  acc = {:.4f}'.format(rs_right, tot, rs_acc))
    print('ms_acc right ={} tot = {}  acc = {:.4f}'.format(ms_right, tot, ms_acc))
    # print('entropy right = {} tot = {} acc = {}'.format(right_entropy, tot, entropy_acc))
    # print(tot_2)

    if is_print:
        with open('roc.txt', 'a') as f:
            f.write('epoch {} baseline: {}, risk: {}, {}\n'.format(epoch, baseline_roc_auc, risk_roc_auc,
                                                          risk_roc_auc - baseline_roc_auc))
            f.write('epoch {} mul_baseline: {}, risk {}, {}\n'.format(epoch, baseline_mul_roc_auc, risk_mul_roc_auc, risk_mul_roc_auc - baseline_mul_roc_auc))
    # logging.info(torch.reshape(outputs, (-1, class_num)).cpu().numpy())
    # np.save('test.pro.npy', torch.reshape(outputs_mul_all, (-1, class_num)).cpu().numpy())
    return outputs_all
    # logging.info(outputs.shape)
    # return outputs.cpu().numpy()
