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
from sklearn.metrics import roc_curve, auc
from risker.pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import LambdaLR

device = [0,1]  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection)
class_num = cfg.get_class_num()

LEARN_VARIANCE = cfg.learn_variance
APPLY_WEIGHT_FUNC = cfg.apply_function_to_weight_classifier_output

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(module)s:%(levelname)s] - %(message)s")

# stddev 标准偏差;truncated 缩减的
# Truncated Normal Distribution一般翻译为截断正态分布
def my_truncated_normal_ppf(confidence, a, b, mean, stddev):
    #  torch.zeros_like:生成和括号内变量维度维度一致的全是零的内容
    x = torch.zeros_like(mean)
    mean = torch.reshape(mean, (-1, 1))
    stddev = torch.reshape(stddev, (-1, 1))
    norm = Normal(mean, stddev)
    # 累积分布函数(CDF)
    # 累积分布函数是指随机变量X小于或等于x的概率
    _nb = norm.cdf(b)
    _na = norm.cdf(a)
    _sb = 1. - norm.cdf(b)
    _sa = 1. - norm.cdf(a)
    # 合并a,b两个tensor a>0的地方保存
    y = torch.where(a > 0,
                    # icdf逆累计分布函数
                    -norm.icdf(confidence * _sb + _sa * (1.0 - confidence)),
                    norm.icdf(confidence * _nb + _na * (1.0 - confidence)))
    return torch.reshape(y, (-1, class_num))

# 对于任意的实数a,b,c，是以著名数学家Carl Friedrich Gauss的名字命名的。
# 高斯的一维图是特征对称“bell curve”形状，a是曲线尖峰的高度，b是尖峰中心的坐标，c称为标准方差，
# 这里的a好像是偏移量
def gaussian_function(a, b, c, x):
    # torch.div
    # 张量和标量做逐元素除法
    # 或者两个可广播的张量之间做逐元素除法
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

    def __init__(self, rule_m, max_w, init_variance=None):
        # 　在子类的__init__方法中使用 super().__init__() , 就相当于把父类的 __init__构造方法拿过来用,
        # 并且可以对父类的__init__方法进行补充
        super(RiskModel, self).__init__()
        self.max_w = max_w
        alaph = torch.tensor(cfg.risk_confidence, dtype=torch.float64)
        a = torch.tensor(0., dtype=torch.float64)
        b = torch.tensor(1., dtype=torch.float64)

        #weight_func_a = torch.tensor(0.5, dtype=torch.float64)
        # weight_fun_b = torch.tensor([0.5] * class_num, dtype=torch.float64)
        weight_fun_b = torch.tensor([0.5], dtype=torch.float64)
        # weight_func_c = torch.tensor(0.5, dtype=torch.float64)

        # register_buffer该方法的作用是定义一组参数，该组参数的特别之处在于：
        # 模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        # 但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        self.register_buffer('alaph', alaph)
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('weight_fun_b', weight_fun_b)

        self.rule_m = rule_m
        self.machine_m = cfg.interval_number_4_continuous_value

        self.rule_var = nn.Parameter(
            torch.empty((class_num, self.rule_m,), dtype=torch.float64, requires_grad=True)
        )
        # torch.nn.init.uniform(tensor, a=0, b=1)
        # 从均匀分布U(a, b)中生成值，填充输入的张量或变量
        #
        # 参数：
        # tensor - n维的torch.Tensor
        # a - 均匀分布的下界
        # b - 均匀分布的上界
        torch.nn.init.uniform_(self.rule_var, 0.1, 0.9)
        self.machine_var = nn.Parameter(
            torch.empty(self.machine_m, dtype=torch.float64, requires_grad=True)
        )
        torch.nn.init.uniform_(self.machine_var, 0.1, 0.9)
        # self.machine_var = nn.Parameter(
        #     torch.empty(self.machine_m, dtype=torch.float64, requires_grad=True)
        # )
        # torch.nn.init.uniform_(self.machine_var, a=0.1, b=0.9)

        # rule_w就是每个类的所有规则的权重
        self.rule_w = nn.Parameter(
            torch.empty((class_num, self.rule_m,), dtype=torch.float64, requires_grad=True)
        )

        # torch.nn.init.uniform(tensor, a=0, b=1)
        # 从均匀分布U(a, b)中生成值，填充输入的张量或变量
        #
        # 参数：
        # tensor - n维的torch.Tensor
        # a - 均匀分布的下界
        # b - 均匀分布的上界

        # 填充rule_w
        torch.nn.init.uniform_(self.rule_w, 0, max_w)

        # self.machine_w = nn.Parameter(
        #     torch.empty((class_num, self.machine_m,), dtype=torch.float64)
        # )
        # torch.nn.init.uniform(self.machine_w, a=0, b=max_w)

        self.learn2rank_sigma = nn.Parameter(torch.tensor(1., dtype=torch.float64, requires_grad=True))
        # self.weight_fun_a = nn.Parameter(torch.tensor([1.] * class_num, dtype=torch.float64, requires_grad=True))
        # self.weight_fun_c = nn.Parameter(torch.tensor([0.5] * class_num, dtype=torch.float64, requires_grad=True))
        # 高斯函数的两个参数
        self.weight_fun_a = nn.Parameter(torch.tensor([1.], dtype=torch.float64, requires_grad=True))
        self.weight_fun_c = nn.Parameter(torch.tensor([0.5], dtype=torch.float64, requires_grad=True))

        # self.register_parameter('rule_vars', self.rule_var)
        # self.register_parameter('machine_var', self.machine_var)
        # self.register_parameter('rule_w', self.rule_w)
        # self.register_parameter('learn2rank_sigma', self.learn2rank_sigma)
        # self.register_parameter('weight_fun_a', self.weight_fun_a)
        # self.register_parameter('weight_fun_c', self.weight_fun_c)
    #  machine_labels one-hot下的机器标签 rule_mus 数据对应的规则的mu machine_one机器标签  rule_feature_matrix规则匹配矩阵 机器概率矩阵  y_risk分险标签 y_mul_risk多分类风险标签
    #  init_rule_mu init_machine_mu 初始mu
    def forward(self, machine_labels, rule_mus, machine_mus, rule_feature_matrix, machine_feature_matrix,
                machine_one, y_risk, y_mul_risk, init_rule_mu, init_machine_mu):
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
        # 多分类的mu转成一个mu
        machine_mus_vector = torch.reshape(torch.sum(machine_mus, 2), (-1, class_num))
        # machine_w = gaussian_function(self.weight_fun_a, self.weight_fun_b, self.weight_fun_c,
        #                           machine_mus_vector.reshape((-1, class_num)))
        # 权重
        machine_w = gaussian_function(self.weight_fun_a, self.weight_fun_b, self.weight_fun_c,
                                 machine_mus_vector.reshape((-1, 1)))
        # print('---funa')
        # print(self.weight_fun_a)
        # print('----func')
        # print(self.weight_fun_c)

        # 可能有用
        machine_w = machine_w.reshape((-1, class_num))
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

        # 计算机器标签mu
        big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * machine_w + 1e-10
        # big_mu = torch.sum(rule_mus * self.rule_w, 2) + machine_mus_vector * torch.sum(rule_feature_matrix * self.rule_w, 2) + 1e-10
        # print('mu')
        # print(big_mu)

        # RSD
        # 相对标准方差计算 计算bigsigma
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

        # 机器标签的sigma
        big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (machine_w ** 2) + 1e-10
        # big_sigma = torch.sum(rule_sigma * (self.rule_w ** 2), 2) + machine_sigma_vector * (torch.sum(rule_feature_matrix * self.rule_w, 2) ** 2) + 1e-10
        # print('--sigma')
        # print(big_sigma)
        # è®¡ç®æé

        # 计算总权重
        r_w = self.rule_w
        r_all_w = torch.sum(rule_feature_matrix * self.rule_w, 2)
        # print(r_all_w[0][0:10])
        # print('----------------')
        # print(machine_w[0][0:10])
        weight_vector = torch.sum(rule_feature_matrix * self.rule_w, 2) + machine_w + 1e-10
        #weight_vector = torch.sum(rule_feature_matrix * self.rule_w, 2) * 2 + 1e-10

        # 这里算出正态分布的均值，方差。 归一化
        big_mu = big_mu / (weight_vector + 1e-10)
        big_sigma = big_sigma / (weight_vector ** 2 + 1e-10)
        # print('------norm')
        # print(big_mu)
        # print('---')
        # print(big_sigma)
        machine_one = machine_one.float()

        # Fr_alpha是标签为不匹配时的风险值
        Fr_alpha = my_truncated_normal_ppf(self.alaph, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        # Fr_alpha_bar是标签为匹配时的风险值VAR
        Fr_alpha_bar = my_truncated_normal_ppf(1 - self.alaph, self.a, self.b, big_mu, torch.sqrt(big_sigma))
        # print(Fr_alpha)
        # print(Fr_alpha_bar)
        # prob_mul = Fr_alpha * (torch.ones_like(machine_one) - machine_one) + (
        #         torch.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_one
        prob_mul = 1 - Fr_alpha_bar

        # machine_one就是[0....1..]标签
        prob = torch.sum(prob_mul * machine_one, 1)
        prob = torch.reshape(prob, (-1, 1))

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
               self.rule_w.data, rule_var.data, machine_var.data, big_mu, big_sigma, r_all_w, machine_w, [Fr_alpha, Fr_alpha_bar]


class PairwiseLoss(nn.Module):
    def __init__(self, learn2rank_sigma, ):
        super(PairwiseLoss, self).__init__()
        self.learn2rank_sigma = learn2rank_sigma
        self.result = torch.empty((0, 2), dtype=torch.float64)
        self.init_result = self.result

    # @torchsnooper.snoop()
    def forward(self, input, target):
        pairwise_probs = self.get_pairwise_combinations(input).to(device[0])
        # print(pairwise_probs.shape)
        pairwise_labels = self.get_pairwise_combinations(target.float()).to(device[0])
        # print(pairwise_labels.shape)

        p_target_ij = 0.5 * (1.0 + pairwise_labels[:, 0] - pairwise_labels[:, 1])
        o_ij = pairwise_probs[:, 0] - pairwise_probs[:, 1]

        diff_label_indices = torch.nonzero(p_target_ij != 0.5, as_tuple=False)  # .squeeze()
        # print(diff_label_indices.shape)
        new_p_target_ij = p_target_ij[diff_label_indices]
        # print(new_p_target_ij.shape)
        new_o_ij = o_ij[diff_label_indices] * self.learn2rank_sigma
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



def train(model, val, test, init_rule_mu, init_machine_mu, epoch_cnn=0, epoches=100
          , suffle_data=True):
    # rule_m = val._rule_mus.shape[1]
    # machine_m = cfg.interval_number_4_continuous_value
    # val, test = test, val

    data_len = len(val.risk_labels)

    # random.seed(2020)
    # np.random.seed(2020)
    # torch.manual_seed(2020)

    
    machine_labels = torch.tensor(val.machine_labels.reshape([val.data_len, 1]), dtype=torch.int) # n * 1
    rule_mus = torch.tensor(val.get_risk_mean_X_discrete(), dtype=torch.float64) # n * class_num * m --> 
    machine_mus = torch.tensor(val.get_risk_mean_X_continue(), dtype=torch.float64)# n * class_num
    # 规则激活矩阵
    rule_feature_activate = torch.tensor(val.get_rule_activation_matrix(), dtype=torch.float64) # n * class_num * m
    machine_feature_activate = torch.tensor(val.get_prob_activation_matrix(), dtype=torch.float64) # n * class_num
    machine_one = torch.tensor(val.machine_label_2_one.reshape([data_len, class_num]), dtype=torch.float64) 
    y_risk = torch.tensor(val.risk_labels.reshape([data_len, 1]), dtype=torch.float64) 
    y_mul_risk = torch.tensor(val.risk_mul_labels.reshape([data_len, class_num]), dtype=torch.float64) 

    risk_weight = 0
    learning_rate = 0.01
    l2_reg = 0.001
    bs = 16
    batch_num = data_len // bs + (1 if data_len % bs else 0)

    for name, p in model.named_parameters():
        print(name)

    criterion = PairwiseLoss(model.learn2rank_sigma).to(device[0])
    loss2 = torch.nn.BCELoss()
    # ä¼åå½æ° SGD ä¸?Adam 2é?ï¼?ç®åæµè¯è¡¨æadamæ´å¥½
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # å­¦ä¹ çè°æ´ï¼å¯é?    # lambda2 = lambda epoch: 0.99 ** (epoch // 1)
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda2)

    # æååæ­¢è®­ç»ï¼?å¯élossæèvalçrocååèæ åï¼ éä¿®æ¹æºç 
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
        outputs_all = torch.empty((0, 1), dtype=torch.float64, requires_grad=False)
        outputs_mul_all = torch.empty((0, class_num), dtype=torch.float64, requires_grad=False)
        right = 0
        tot = 0
        fr_right = 0
        for i in range(batch_num):

            machine_labels_batch = machine_labels[index][bs * i: bs * i + bs].to(device[0])
            rule_mus_batch = rule_mus[index][bs * i: bs * i + bs].to(device[0])
            machine_mus_batch = machine_mus[index][bs * i: bs * i + bs].to(device[0])
            rule_feature_activate_batch = rule_feature_activate[index][bs * i: bs * i + bs].to(device[0])
            machine_feature_activate_batch = machine_feature_activate[index][bs * i: bs * i + bs].to(device[0])
            machine_one_batch = machine_one[index][bs * i: bs * i + bs].to(device[0])
            y_risk_batch = y_risk[index][bs * i: bs * i + bs].to(device[0])
            y_mul_risk_batch = y_mul_risk[index][bs * i: bs * i + bs].to(device[0])

            outputs_mul, outputs, func_params, rule_w, rule_var, machine_var, big_mu, big_sigma, r_w, m_w, fr = model(machine_labels_batch,
                                                                                     rule_mus_batch,
                                                                                     machine_mus_batch,
                                                                                     rule_feature_activate_batch,
                                                                                     machine_feature_activate_batch,
                                                                                     machine_one_batch,
                                                                                     y_risk_batch,
                                                                                     y_mul_risk_batch,
                                                                                     init_rule_mu,
                                                                                     init_machine_mu,)

            l1_loss = torch.tensor(0.).to(device[0])
            l2_loss = torch.tensor(0.).to(device[0])
            beta = torch.tensor(0.001).to(device[0])

            # l1, l2 æ­£åï¼?æ¹å·®ä¸å¤ç?
            for name, param in model.named_parameters():
                if 'var' not in name:
                    l1_loss += beta * torch.sum(torch.abs(param))
                    l2_loss += (0.5 * beta * torch.sum(param ** 2))
            optimizer.zero_grad()
            rank_loss1 = criterion(outputs.reshape((-1, 1)), y_risk_batch.reshape((-1, 1))) 
            # rank_loss2 = criterion(outputs_mul.reshape((-1, 1)), y_mul_risk_batch.reshape((-1, 1)))
            loss = rank_loss1 + l1_loss + l2_loss
            #loss = torch.tensor(0, dtype=torch.float64).to(device[0])
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
            # loss = torch.tensor(0, dtype=torch.float64).to(device[0])
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
            # 截断 规范mu sigma范围不能超过1
            model.rule_var.data.clamp_(0., 1.)
            model.machine_var.data.clamp_(0., 1.)
            model.weight_fun_a.data.clamp_(1e-10, )
            model.weight_fun_c.data.clamp_(1e-10, )

            loss_total += loss.item()
            n_total += len(outputs_mul)
            # logging.info("[%d][%d/%d], loss=%f" % (epoch, i, batch_num, loss.item()))
            outputs_all = torch.cat((outputs_all.to(device[0]), outputs), dim=0)
            outputs_mul_all = torch.cat((outputs_mul_all.to(device[0]), outputs_mul), dim=0)

            right += torch.sum(
                torch.argmax(big_mu, 1).reshape(-1, 1) == torch.tensor(val.true_labels[index][bs * i: bs * i + bs],
                                                                       dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))
            tot += len(big_mu)
            fr_right += torch.sum(
                torch.argmax(fr[0], 1).reshape(-1, 1) == torch.tensor(val.true_labels[index][bs * i: bs * i + bs],
                                                                      dtype=torch.long).reshape(
                    (-1, 1)).to(device[0]))
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
            print('---------------val predict')
            predict(model, val, epoch, init_rule_mu,  init_machine_mu, epoch_cnn,)
            print('-----------------test predict')
            predict(model, test, epoch, init_rule_mu, init_machine_mu, epoch_cnn, True)

        if early_stopping.early_stop:
            logging.info('early stopping')
            break

    return func_params, rule_w, rule_var, machine_var


def predict(model, test, epoch, init_rule_mu, init_machine_mu, epoch_cnn=0, is_print=False):
    data_len = len(test.risk_labels)

    machine_labels = torch.tensor(test.machine_labels.reshape([test.data_len, 1]), dtype=torch.int)
    rule_mus = torch.tensor(test.get_risk_mean_X_discrete(), dtype=torch.float64)
    machine_mus = torch.tensor(test.get_risk_mean_X_continue(), dtype=torch.float64)
    rule_feature_activate = torch.tensor(test.get_rule_activation_matrix(), dtype=torch.float64)
    machine_feature_activate = torch.tensor(test.get_prob_activation_matrix(), dtype=torch.float64)
    machine_one = torch.tensor(test.machine_label_2_one.reshape([data_len, class_num]), dtype=torch.int)
    y_risk = torch.tensor(test.risk_labels.reshape([data_len, 1]), dtype=torch.int)
    y_mul_risk = torch.tensor(test.risk_mul_labels.reshape([data_len, class_num]), dtype=torch.int)

    bs = 4
    batch_num = data_len // bs + (1 if data_len % bs else 0)
    outputs_all = torch.empty((0, 1), dtype=torch.float64)
    outputs_mul_all = torch.empty((0, class_num), dtype=torch.float64)
    big_mu_all = torch.empty((0, class_num), dtype=torch.float64)
    big_sigma_all = torch.empty((0, class_num), dtype=torch.float64)
    fr_all = torch.empty((0, class_num), dtype=torch.float64)
    fr_bar_all = torch.empty((0, class_num), dtype=torch.float64)
    r_w_all = torch.empty((0, class_num), dtype=torch.float64)
    m_w_all = torch.empty((0, class_num), dtype=torch.float64)
    r_every_w_all = None#torch.empty((0, class_num), dtype=torch.float64)
    model.eval()
    right = 0
    right_fr = 0
    right_fr_bar = 0
    right_entropy = 0
    tot = 0
    tot_2 = 0
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

            outputs_mul, outputs, _, r_every_w, r_var, m_var, big_mu, big_sigma, r_w, m_w, fr = model(machine_labels_batch, rule_mus_batch, machine_mus_batch,
                                                     rule_feature_activate_batch, machine_feature_activate_batch,
                                                     machine_one_batch,
                                                     y_risk_batch, y_mul_risk_batch, init_rule_mu, init_machine_mu)
            # logging.info(torch.reshape(outputs_mul, (-1, class_num)).shape)
            outputs_mul_all = torch.cat((outputs_mul_all.to(device[0]), torch.reshape(outputs_mul, (-1, class_num))),
                                        dim=0)
            # logging.info(outputs_mul_all.shape)
            # output = torch.reshape(torch.sum(torch.reshape(outputs, (-1, class_num)) * y_activate_batch, 1), (-1, 1))

            outputs_all = torch.cat((outputs_all.to(device[0]), outputs), dim=0)
            big_mu_all = torch.cat((big_mu_all, big_mu.cpu()), dim=0)
            big_sigma_all = torch.cat((big_sigma_all, big_sigma.cpu()), dim=0)
            fr_all = torch.cat((fr_all, fr[0].cpu()), dim=0)
            fr_bar_all = torch.cat((fr_bar_all, fr[1].cpu()), dim=0)
            r_w_all = torch.cat((r_w_all, r_w.cpu()), dim=0)
            m_w_all = torch.cat((m_w_all, m_w.cpu()), dim=0)
            r_every_w_all = r_every_w.cpu()
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


            tot += len(big_mu)
        torch.cuda.empty_cache()
    fpr, tpr, _ = roc_curve(test.risk_mul_labels.reshape((-1)), outputs_mul_all.reshape((-1)).cpu().numpy())
    risk_mul_roc_auc = auc(fpr, tpr)
    risk_mul_roc_auc = risk_mul_roc_auc * 100

    # risk --> abs(mac_label - mac_prob)
    _machine_mul_pro = (1 - test.machine_mul_probs).reshape(-1, 1)
    fpr, tpr, _ = roc_curve(test.risk_mul_labels.reshape((-1)), _machine_mul_pro.reshape((-1)))
    baseline_mul_roc_auc = auc(fpr, tpr)
    baseline_mul_roc_auc = baseline_mul_roc_auc * 100

    fpr, tpr, _ = roc_curve(test.risk_labels, outputs_all.reshape((-1)).cpu().numpy())
    risk_roc_auc = auc(fpr, tpr)
    risk_roc_auc = risk_roc_auc * 100

    _machine_pro = 1 - test.machine_probs.reshape((-1, 1))
    fpr, tpr, _ = roc_curve(test.risk_labels, _machine_pro.reshape((-1)))
    baseline_roc_auc = auc(fpr, tpr)
    baseline_roc_auc = baseline_roc_auc * 100
    root = './results/CUB/'
    if (not os.path.exists(root)):
        os.makedirs(root)
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
    logging.info("risk mul_roc : %f, risk_mul_baseline : %f \n risk roc : %f,  baseline roc %f," % (risk_mul_roc_auc,
                                                                                                    baseline_mul_roc_auc,risk_roc_auc, baseline_roc_auc))
    print(torch.argmax(big_mu, 1).shape)

    acc = float(right) / float(tot) * 100.0
    fr_acc = float(right_fr) / float(tot) * 100.0
    fr_bar_acc = float(right_fr_bar) / float(tot) * 100.0
    entropy_acc = float(right_entropy) / float(tot) * 100.0
    print('big_mu right ={} tot = {}  acc = {:.4f}'.format(right, tot, acc))
    print('fr right ={} tot = {}  acc = {:.4f}'.format(right_fr, tot, fr_acc))
    print('fr_bar right ={} tot = {}  acc = {:.4f}'.format(right_fr_bar, tot, fr_bar_acc))
    # print('entropy right = {} tot = {} acc = {}'.format(right_entropy, tot, entropy_acc))
    # print(tot_2)

    if is_print:
        with open('roc.txt', 'a') as f:
            f.write('epoch {} baseline: {}, risk: {}, {}\n'.format(epoch, baseline_roc_auc, risk_roc_auc,
                                                          risk_roc_auc - baseline_roc_auc))
            f.write('epoch {} mul_baseline: {}, risk {}, {}\n'.format(epoch, baseline_mul_roc_auc, risk_mul_roc_auc, risk_mul_roc_auc - baseline_mul_roc_auc))
    # logging.info(torch.reshape(outputs, (-1, class_num)).cpu().numpy())
    # np.save('test.pro.npy', torch.reshape(outputs_mul_all, (-1, class_num)).cpu().numpy())
    return outputs_all.cpu().numpy()
    # logging.info(outputs.shape)
    # return outputs.cpu().numpy()
