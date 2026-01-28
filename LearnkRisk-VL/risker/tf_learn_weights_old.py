from __future__ import print_function, with_statement, division, absolute_import

import tensorflow as tf
# from tensorflow.distributions import Normal
import tensorflow_probability as tfp
from sklearn.metrics import roc_curve

from common import config
import numpy as np
from scipy import sparse as sp
from collections import Counter
import math
import logging
from tqdm import trange
import numpy as np

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Deploying CPU setting.
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.5

tfd = tfp.distributions

cfg = config.Configuration(config.global_data_selection, config.global_deep_learning_selection)

LEARN_VARIANCE = cfg.learn_variance
APPLY_WEIGHT_FUNC = cfg.apply_function_to_weight_classifier_output

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(module)s:%(levelname)s] - %(message)s")

class_num = cfg.get_class_num()


def my_truncated_normal_ppf(confidence, a, b, mean, stddev):
    '''
    tf_norm = tfd.Normal(mean, stddev)
    _nb = tf_norm.cdf(b)
    _na = tf_norm.cdf(a)
    _sb = tf_norm.survival_function(b)
    _sa = tf_norm.survival_function(a)

    return tf.where(a > 0,
                    -tf_norm.quantile(confidence * _sb + _sa * (1.0 - confidence)),
                    tf_norm.quantile(confidence * _nb + _na * (1.0 - confidence)))
    '''
    x = tf.zeros_like(mean)
    mean = tf.reshape(mean, [-1, 1])
    stddev = tf.reshape(stddev, [-1, 1])
    tf_norm = tfd.Normal(mean, stddev)
    _nb = tf_norm.cdf(b)
    _na = tf_norm.cdf(a)
    _sb = tf_norm.survival_function(b)
    _sa = tf_norm.survival_function(a)

    y = tf.where(a > 0,
                 -tf_norm.quantile(confidence * _sb + _sa * (1.0 - confidence)),
                 tf_norm.quantile(confidence * _nb + _na * (1.0 - confidence)))
    return tf.reshape(y, [-1, class_num])


def gaussian_function(a, b, c, x):
    _part = - tf.math.divide(tf.square(x - b), tf.multiply(tf.constant(2.0, dtype=tf.double), tf.square(c)))
    # _f = tf.multiply(a, tf.exp(_part))
    _f = - tf.exp(_part) + a + 1.0
    return _f


# def fit(machine_results, _mus_X, _sigmas_X, _y, _feature_activation_matrix, init_mu, init_variance=None):
def fit(machine_results, _mus_rule_X, _mus_machine_X,
        _sigmas_rule_X, _sigmas_machine_X,
        _feature_activation_rule_matrix, _feature_activation_machine_matrix, _machine_one, _y_activate,
        _y, _y_mul, init_rule_mu, init_machine_mu, rule_activation, init_variance=None):
    '''
    machine_results (n, 1)
    _mus_rule_X (n, 10, 30)
    _mus_machine_X (n, 10, 50)



    '''

    """
    All input parameters are numpy array type.
    :param machine_results: shape: (n, 1), the machine results.
    :param _mus_X: shape: (n, m), the means of m risk features. !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _sigmas_X: shape: (n, m), the variances of m risk features. !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _y: shape: (n, 1), the risk labels of n training data.
    :param _feature_activation_matrix: shape (n, m), indicate which features are used in each data.
                                        !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param init_mu: shape (m, 1)
    :param init_variance: shape (m, 1)
    :return:
    """

    tf.reset_default_graph()

    '''
    m = _mus_X.shape[1]
    print('-------------------------------------------------------')
    print(m)
    print('--------------------------------------------------------')
    '''

    if not APPLY_WEIGHT_FUNC:
        # # -- feature weights --
        # # continuous_m = config.get_interval_number_4_continuous_value()  # number of continuous features
        # continuous_m = 0
        # if continuous_m > 0:
        #     print("- One weight for multiple intervals.")
        #     one_conti_feature = 1
        # else:
        #     print("- Different weights for different intervals.")
        #     one_conti_feature = 0
        # discrete_m = m - continuous_m  # number of discrete features
        pass
    else:
        print("- Different weights for different probabilities (use Gaussian function).")
        continuous_m = cfg.interval_number_4_continuous_value
        discrete_m = _mus_rule_X.shape[2]
        print('---------------------discrete_m')
        print(discrete_m)
        one_conti_feature = 0

    # -- feature variances --
    # continuous_m_var = config.get_interval_number_4_continuous_value()
    continuous_m_var = 0
    if continuous_m_var > 0:
        print("- One variance for multiple intervals.")
        one_conti_feature_var = 1
    else:
        print("- Different variances for different intervals.")
        one_conti_feature_var = 0
    discrete_m_var = discrete_m + continuous_m

    data_len = len(_y)  # number of data
    np.random.seed(2019)
    tf.set_random_seed(2019)
    # The mean number of activated rule features.

    _feature_activation_matrix = np.concatenate(
        (np.array(_feature_activation_rule_matrix), np.array(_feature_activation_machine_matrix)), axis=2)
    rule_mean_number = np.mean(np.array(np.sum(_feature_activation_matrix, axis=2))) - 1.0 + 1e-10
    max_w = 1.0 / np.max(np.sum(_feature_activation_matrix, axis=2))
    class_count = Counter(_y.reshape(-1))
    print("- Max number of features: {}, 1/max={}.".format(1.0 / max_w, max_w))
    del _feature_activation_matrix
    # -- Set class weight w.r.t class size. --
    # risky_weight = 1.0 * class_count.get(0) / class_count.get(1)
    # -- or set to 1.0 --
    risky_weight = 1.0
    print("- Set risky label weight = {}. [{}]".format(risky_weight, class_count))
    learning_rate = cfg.learing_rate
    print("- Set learning rate = {}".format(learning_rate))

    with tf.name_scope('constants'):
        alpha = tf.constant(cfg.risk_confidence, dtype=tf.double)
        a = tf.constant(0.0, dtype=tf.double)
        b = tf.constant(1.0, dtype=tf.double)
        l1 = tf.constant(0.001, dtype=tf.double)
        l2 = tf.constant(0.001, dtype=tf.double)
        label_match_value = tf.constant(1.0, dtype=tf.double)
        label_unmatch_value = tf.constant(0.0, dtype=tf.double)
        # print("- Manually set parameters of Gaussian function.")
        # weight_func_a = tf.constant(0.5, dtype=tf.double)
        weight_func_b = tf.constant([0.5] * class_num, dtype=tf.double)
        # weight_func_c = tf.constant(1.0, dtype=tf.double)
        variance_initializer = None
        if init_variance is not None:
            variance_initializer = np.array(init_variance).reshape([-1, 1])[:discrete_m_var + one_conti_feature_var, 0]
            variance_initializer = tf.constant(variance_initializer,
                                               dtype=tf.double,
                                               shape=[discrete_m_var + one_conti_feature_var, 1])

    with tf.name_scope('inputs'):
        # Variables (Vectors)
        machine_label = tf.placeholder(tf.int8, name='ML')  # (n, 1)
        risk_y = tf.placeholder(tf.int64, name='y')  # (n, 1)
        risk_y_mul = tf.placeholder(tf.int64, [None, class_num], name='y_mul')  # (n, 10)
        mus_rule = tf.placeholder(tf.double, [None, class_num, discrete_m], name='mu_rule')  # (n, 10, 30)
        mus_machine = tf.placeholder(tf.double, [None, class_num, continuous_m], name='mu_mac')  # (n, 10, 50)
        sigmas_rule = tf.placeholder(tf.double, [None, class_num, discrete_m], name='sigma_rule')  # (n, 10, 30)
        sigmas_machine = tf.placeholder(tf.double, [None, class_num, continuous_m], name='sigma_mac')  # (n, 10, 50)
        feature_rule_matrix = tf.placeholder(tf.double, [None, class_num, discrete_m],
                                             name='featureRuleMatrix')  # (n, 10, 30)
        feature_machine_matrix = tf.placeholder(tf.double, [None, class_num, continuous_m],
                                                name='featureMacMatrix')  # (n, 10, 50)
        machine_one = tf.placeholder(tf.double, [None, class_num], name="machine_one")
        y_activate = tf.placeholder(tf.double, [None, class_num], name="y_activate")
        # parameters for learning to rank
        pairwise_risky_values = tf.placeholder(tf.double, name='pairwise_values')
        pairwise_risky_labels = tf.placeholder(tf.double, name='pairwise_labels')

        # alpha = tf.get_variable(name='alpha', shape=[], initializer=tf.random_uniform_initializer(0., 1.),
        #                         constraint=lambda t: tf.clip_by_value(t, 0., 1.), dtype=tf.double)
        # label_match_value = tf.get_variable(name='match_value',
        #                                     initializer=tf.constant(1.0, dtype=tf.double),
        #                                     constraint=lambda t: tf.clip_by_value(t, 0., 1.), dtype=tf.double)
        # label_unmatch_value = tf.get_variable(name='unmatch_value',
        #                                       initializer=tf.constant(0.0, dtype=tf.double),
        #                                       constraint=lambda t: tf.clip_by_value(t, 0., 1.), dtype=tf.double)

        # Discrete feature weights and one probability-based feature weight.

        discrete_w = tf.get_variable(name='discrete_w', shape=[class_num, discrete_m],
                                     dtype=tf.double,
                                     initializer=tf.random_uniform_initializer(0., max_w),
                                     regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                     constraint=lambda t: tf.abs(t)
                                     # constraint=lambda t: tf.clip_by_value(t, 1, 10)
                                     )  # (m ,1)

        # Note: for continuous features, different intervals have their own variances.
        if variance_initializer is None:
            # This variable is treated as relative standard deviation (RSD), rsd = sd / mu * 100%.
            # For simplicity, we do not change its name here.
            discrete_variances = tf.get_variable(name='discrete_variances',
                                                 shape=[class_num, discrete_m],
                                                 dtype=tf.double,
                                                 initializer=tf.random_uniform_initializer(0., 1.),
                                                 # constraint=lambda t: tf.abs(t),
                                                 constraint=lambda t: tf.clip_by_value(t, 0., 1.)
                                                 )
            continuous_variances = tf.get_variable(name='continuous_variances',
                                                   shape=[class_num, continuous_m],
                                                   dtype=tf.double,
                                                   initializer=tf.random_uniform_initializer(0., 1.),
                                                   # constraint=lambda t: tf.abs(t),
                                                   constraint=lambda t: tf.clip_by_value(t, 0., 1.)
                                                   )
        else:
            pass
            # discrete_variances = tf.get_variable(name='discrete_variances',
            #                                      dtype=tf.double,
            #                                      initializer=variance_initializer,
            #                                      # constraint=lambda t: tf.abs(t),
            #                                      constraint=lambda t: tf.clip_by_value(t, 0., 0.5)
            #                                      )

        learn2rank_sigma = tf.get_variable(name='learning_to_rank_sigma',
                                           initializer=tf.constant(1.0, dtype=tf.double),
                                           regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                           # constraint=lambda t: tf.abs(t)
                                           )

        weight_func_a = tf.get_variable(name='weight_function_w',
                                        initializer=tf.constant([1.0] * class_num, dtype=tf.double),
                                        regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                        constraint=lambda t: tf.abs(t))
        # weight_func_b = tf.get_variable(name='weight_function_mean',
        #                                 initializer=tf.constant(0.5, dtype=tf.double),
        #                                 constraint=lambda t: tf.abs(t))
        weight_func_c = tf.get_variable(name='weight_function_variance',
                                        initializer=tf.constant([0.5] * class_num, dtype=tf.double),
                                        regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                        constraint=lambda t: tf.abs(t))

    n = risk_y.get_shape()[0]
    print('---------------------')
    print(n)
    with tf.name_scope('prediction'):
        # Handling Feature Weights.
        if one_conti_feature == 1:
            # 1.1 Continuous feature weights.
            contin_w = tf.convert_to_tensor([discrete_w[discrete_m]] * (continuous_m - 1))
            # 1.2 Concat both weights.
            w = tf.concat([discrete_w, contin_w], axis=0)
        else:
            pass
            # w = discrete_w

        # Handling Feature Variances.
        if one_conti_feature_var == 1:
            # 2.1 Continuous feature variances.
            contin_variances = tf.convert_to_tensor([discrete_variances[discrete_m]] * (continuous_m_var - 1))
            # 2.2 Concat both weiths.
            variances = tf.concat([discrete_variances, contin_variances], axis=0)
        else:
            pass

        # In the newest solution, the above 'variances' is actually Relative Standard Deviation.
        # Here we transform it to real variances.
        '''
        new_init_mu = init_mu.copy()
        # If mu is 0.0, then the variance will be 0.0 in any case. So we set the 0.0 to 0.1 for non-zero variances.
        new_init_mu[np.where(new_init_mu == 0.0)] = 0.1
        standard_deviation = variances * new_init_mu
        #standard_deviation = variances / new_init_mu
        variances = tf.square(standard_deviation)
        '''
        '''
        new_init_rule_mu = init_rule_mu.copy()
        new_init_rule_mu[np.where(new_init_rule_mu == 0.0)] = 0.1
        standard_deviation_rule = discrete_variances * new_init_rule_mu
        discrete_variances = tf.square(standard_deviation_rule )

        new_init_machine_mu = init_machine_mu.copy()
        new_init_machine_mu[np.where(new_init_machine_mu == 0.0)] = 0.1
        standard_deviation_machine = continuous_variances * new_init_machine_mu
        continuous_variances = tf.square(standard_deviation_machine)
        '''
        if not APPLY_WEIGHT_FUNC:
            # #  Note: In practice, big_mu and big_sigma can be zero, and when calculates gradients,
            # #        the f(big_mu, big_sigma) can be zero and as the denominator at the same time.
            # #        So here we add a small number 1e-10.
            # big_mu = tf.matmul(mus, w) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            #
            # if not LEARN_VARIANCE:
            #     # -- ** 1. use pre-set variances. ** --
            #     print("- No learning variances.")
            #     big_sigma = tf.matmul(sigmas, tf.square(w)) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            # else:
            #     # -- ** 2. learn variances. ** --
            #     print("- Learning variances.")
            #     learn_sigmas_X = feature_matrix * tf.reshape(variances, [1, m])
            #     big_sigma = tf.matmul(learn_sigmas_X, tf.square(w)) + 1e-10
            #
            # # Normalize the weights of features in each pair.
            # weight_vector = tf.matmul(feature_matrix, w) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            # big_mu = big_mu / weight_vector
            # big_sigma = big_sigma / (tf.square(weight_vector))
            pass
        else:
            # Need to calculate weights for each probability.
            # rule_mus = tf.slice(mus, [0, 0], [-1, discrete_m])
            # sparse matrix, only one position has value.
            # machine_mus = tf.slice(mus, [0, discrete_m], [-1, continuous_m])
            # machine_mus_vector = tf.reshape(tf.reduce_sum(machine_mus, axis=1), [-1, 1])
            # rule_mus = mus_rule
            # machine_mus = mus_machine
            machine_mus_vector = tf.reshape(tf.reduce_sum(mus_machine, axis=2), [-1, class_num])

            machine_weights = gaussian_function(weight_func_a, weight_func_b, weight_func_c, machine_mus_vector)
            machine_weights = tf.reshape(machine_weights, [-1, class_num])

            _a = tf.reduce_sum(tf.multiply(mus_rule, discrete_w), axis=2)
            _b = tf.multiply(machine_mus_vector, machine_weights)

            # big_mu = tf.reduce_sum(tf.multiply(tf.multiply(mus_rule, discrete_w), rule_activation), axis=2) + tf.multiply(machine_mus_vector, machine_weights) + 1e-10

            # mus_rule = tf.multiply(mus_rule, feature_rule_matrix)
            big_mu = tf.reduce_sum(tf.multiply(mus_rule, discrete_w), axis=2) + tf.multiply(machine_mus_vector,
                                                                                            machine_weights) + 1e-10
            # big_mu = tf.reduce_sum(tf.multiply(mus_rule, discrete_w), axis=2) + 1e-10
            if not LEARN_VARIANCE:
                # # -- ** 1. use pre-set variances. ** --
                # print("- No learning variances.")
                # use_sigmas = sigmas
                pass
            else:
                # -- ** 2. learn variances. ** --
                print("- Learning variances.")
                rule_sigmas = tf.multiply(feature_rule_matrix, discrete_variances)
                machine_sigmas = tf.multiply(feature_machine_matrix, continuous_variances)
            # rule_sigmas = rule_sigmas
            # machine_sigmas = machine_sigmas
            machine_sigmas_vector = tf.reshape(tf.reduce_sum(machine_sigmas, axis=2), [-1, class_num])

            # big_sigma = tf.reduce_sum(tf.multiply(tf.multiply(rule_sigmas, discrete_w * discrete_w), rule_activation), axis=2) +  tf.multiply(machine_sigmas_vector,
            #                                                                        machine_weights * machine_weights) + 1e-10
            # _a = tf.multiply(rule_sigmas, discrete_w * discrete_w)

            _c = tf.reduce_sum(tf.multiply(rule_sigmas, discrete_w * discrete_w), axis=2)
            _d = tf.multiply(machine_sigmas_vector, machine_weights * machine_weights)

            big_sigma = tf.reduce_sum(tf.multiply(rule_sigmas, discrete_w * discrete_w), axis=2) + tf.multiply(
                machine_sigmas_vector,
                machine_weights * machine_weights) + 1e-10

            # big_sigma = tf.reduce_sum(tf.multiply(rule_sigmas, discrete_w * discrete_w), axis=2) + 1e-10

            # Normalize the weights of features in each pair.
            # rule_activate_matrix = feature_rule_matrix
            weight_vector = tf.reduce_sum(tf.multiply(feature_rule_matrix, discrete_w),
                                          axis=2) + machine_weights + 1e-10
            # weight_vector = tf.reduce_sum(tf.multiply(feature_rule_matrix, discrete_w), axis=2)  + 1e-10

            big_mu = big_mu / weight_vector  # n * 10
            big_sigma = big_sigma / (tf.square(weight_vector))  # n * 10

        # Truncated normal distribution.

        machine_label = tf.cast(machine_label, tf.double)
        machine_one = tf.cast(machine_one, tf.double)
        y_activate = tf.cast(y_activate, tf.double)
        # -- Option 1: Use match value: 1, unmatch value: 0. --
        # prob = Fr_alpha * (tf.ones_like(machine_label) - machine_label) + (
        #        tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_label

        print('-------------------------------------big_mu')
        prob = tf.zeros_like(machine_label)
        prob = tf.cast(prob, tf.double)
        # prob_ = tf.zeros_like(machine_label)
        prob_mul = tf.zeros_like(big_mu)
        prob = tf.zeros_like(machine_label)
        Fr_alpha = my_truncated_normal_ppf(alpha, a, b, big_mu, tf.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, big_mu, tf.sqrt(big_sigma))
        prob_mul = Fr_alpha * (tf.ones_like(machine_one) - machine_one) + (
                tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_one

        prob__ = prob_mul * y_activate
        prob = tf.reduce_sum(prob_mul * y_activate, axis=1)

        prob_mul = tf.reshape(prob_mul, [-1, 1])
        y_mul_ = tf.reshape(risk_y_mul, [-1, 1])

        '''

        for i in range(class_num):
            Fr_alpha = my_truncated_normal_ppf(alpha, a, b, tf.slice(big_mu,[0,i],[-1,1]), tf.sqrt(tf.slice(big_sigma,[0,i],[-1,1])))
            Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, tf.slice(big_mu,[0,i],[-1,1]), tf.sqrt(tf.slice(big_sigma,[0,i],[-1,1])))
            #rule_begin = rules_idx[i][0]
            #rule_end = rules_idx[i][1]
            #print(rule_begin, rule_end)
            #Fr_alpha = my_truncated_normal_ppf(alpha, a, b, tf.slice(big_mu, [rule_begin, 0], [rule_end - rule_begin - 1, 1]), tf.sqrt(tf.slice(big_sigma, [rule_begin, 0], [rule_end - rule_begin - 1, 1] ) ))
            #Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, tf.slice(big_mu, [rule_begin,0], [rule_end - rule_begin - 1,1]), tf.sqrt(tf.slice(big_sigma, [rule_begin, 0], [rule_end - rule_begin - 1, 1] ) ))
            #Fr_alpha = my_truncated_normal_ppf(alpha, a, b, big_mu, tf.sqrt(big_sigma))
            #Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, big_mu, tf.sqrt(big_sigma))
            change_machine_label =  tf.where(tf.equal(machine_label, i), tf.ones_like(machine_label), tf.zeros_like(machine_label))
            #change_machine_label = tf.where(tf.equal(machine_label, i), tf.zeros_like(machine_label), tf.ones_like(machine_label))
            change_machine_label = tf.cast(change_machine_label, tf.double)
            #prob += Fr_alpha * (tf.ones_like(change_machine_label) - change_machine_label) + (
            #    tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * change_machine_label

            prob += tf.where(tf.equal(machine_label, i),  (
                tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * change_machine_label, tf.zeros_like(change_machine_label) )
            #prob += Fr_alpha * (tf.ones_like(change_machine_label) - change_machine_label) + tf.abs(_y * change_machine_label -  (
            #    tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * change_machine_label)
            #prob /= class_num

            #prob_ = Fr_alpha * (tf.ones_like(change_machine_label) - change_machine_label) + (
            #    tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * change_machine_label

            #prob_mul = tf.concat([prob_mul, prob_], axis=1)
            #prob += tf.where(tf.equal(machine_label, i), (tf.slice(big_mu,[0,i],[-1,1]) - Fr_alpha_bar) *change_machine_label, tf.zeros_like(change_machine_label))
        '''
        # prob += 1e-10
        # prob_mul = tf.reshape(tf.slice(prob_mul, [0, 1], [-1, -1]), [-1, class_num])
        # risk_y_ = tf.reshape(risk_y_mul, [-1, 1])
        # prob_ = tf.reshape(prob_mul, [-1, 1])

        # -- Option 2: Use match value: learning, unmatch value: learning. --
        # prob = (Fr_alpha - tf.ones_like(Fr_alpha) * label_unmatch_value) * (
        #             tf.ones_like(machine_label) - machine_label) + (
        #                tf.ones_like(Fr_alpha_bar) * label_match_value - Fr_alpha_bar) * machine_label

        # -- Option 3: Use match value: mean, unmatch value: mean. --
        # prob = (Fr_alpha - big_mu) * (tf.ones_like(machine_label) - machine_label) + (
        #            big_mu - Fr_alpha_bar) * machine_label

        '''
        prob_2_label = tf.concat([prob, tf.cast(risk_y, tf.double)], axis=1)
        descent_order = tf.gather(prob_2_label, tf.nn.top_k(-prob_2_label[:, 0], k=tf.shape(risk_y)[0]).indices)
        order_risky_label = tf.slice(descent_order, [0, 1], [-1, 1])
        risky_index = tf.where(tf.equal(order_risky_label, 1))
        risky_row_index = tf.slice(risky_index, [0, 0], [-1, 1]) + 1

        acc = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.math.greater(prob, 0.5), tf.int64), risk_y), tf.int32))

        # Average Precision
        risky_count = tf.reshape(tf.where(tf.equal(risky_row_index, risky_row_index))[:, 0], [-1, 1]) + 1
        cut_off_k_precisions = tf.math.divide(tf.cast(risky_count, tf.double), tf.cast(risky_row_index, tf.double))
        avg_precision = tf.reduce_mean(cut_off_k_precisions)
        '''

    with tf.name_scope('loss'):
        # -- Option 1: Use risky values. --
        # cost = - tf.reduce_mean(tf.cast(risk_y, tf.double) * tf.log(tf.clip_by_value(prob, 1e-10, 1.0)) +
        #                         (1.0 - tf.cast(risk_y, tf.double)) * tf.log(tf.clip_by_value(1.0 - prob, 1e-10, 1.0)))
        # cost = - tf.reduce_sum(tf.cast(risk_y, tf.double) * tf.log(prob + 1e-10) * tf.cast(risky_weight, tf.double) +
        #                        (1.0 - tf.cast(risk_y, tf.double)) * tf.log(
        #     1.0 - prob + 1e-10)) + tf.losses.get_regularization_loss()

        # -- Option 2: Use rank positions. --
        # -- NOT WORK. "... ops that do not support gradients ..." --
        # cost = tf.reduce_sum(tf.cast(risky_row_index, tf.double))

        # -- Option 3: Use pairwise loss. --
        '''
        * cross entropy cost function.
        loss_function = - p'_ij * log(p_ij) - (1 - p'_ij) * log(1 - p_ij)
        where,
        p'_ij = 0.5 * (1 + higher_rank(xi, xj));
        p_ij = e^(o_ij) / (1 + e^(o_ij)), o_ij = f(xi) - f(xj);
        higher_rank(xi, xj) = xi_risky_label - xj_risky_label;
        Concise Form:
        loss_function = - p'_ij * o_ij + log(1 + e^(o_ij))
        Ref: Burges C, Shaked T, Renshaw E, et al. Learning to rank using gradient descent[C].
        Proceedings of the 22nd International Conference on Machine learning (ICML-05). 2005: 89-96.
        '''

        # cost = 0.0
        # current_data_size = 10
        # for i in range(0, current_data_size - 1):
        #     for j in range(i+1, current_data_size):
        #         p_target_ij = 0.5 * (1.0 + prob_2_label[i][1] - prob_2_label[j][1])
        #         o_ij = prob_2_label[i][0] - prob_2_label[j][0]
        #         cost += - p_target_ij * o_ij + tf.log(1.0 + tf.exp(o_ij))
        # cost += tf.losses.get_regularization_loss()

        def get_pairwise_combinations(input_data, out_result):
            start_index = tf.constant(0)

            def while_condition(_i, *args):
                return tf.less(_i, tf.shape(input_data)[0] - 1)

            def body(_i, data, result):
                # do something here which you want to do in your loop
                # increment i
                result = tf.concat(
                    [result, tf.reshape(tf.stack(tf.meshgrid(data[_i], data[_i + 1:]), axis=-1), [-1, 2])], axis=0)
                return [tf.add(_i, 1), data, result]

            # do the loop:
            r = tf.while_loop(while_condition, body, [start_index, input_data, out_result])[2][1:]
            return r

        # pairwise_probs = get_pairwise_combinations(prob, pairwise_risky_values)
        # pairwise_labels = get_pairwise_combinations(tf.cast(risk_y, tf.double), pairwise_risky_labels)
        pairwise_probs = get_pairwise_combinations(prob_mul, pairwise_risky_values)
        pairwise_labels = get_pairwise_combinations(tf.cast(y_mul_, tf.double), pairwise_risky_labels)
        p_target_ij = 0.5 * (1.0 + pairwise_labels[:, 0] - pairwise_labels[:, 1])
        o_ij = pairwise_probs[:, 0] - pairwise_probs[:, 1]
        # -- 1. apply parabola to remove pairs that have same labels.
        # cost = tf.reduce_mean((- p_target_ij * o_ij + tf.log(1.0 + tf.exp(o_ij))) * tf.square(
        #    p_target_ij - 0.5) * 4) + tf.losses.get_regularization_loss()
        # -- 2. Remove pairs that have same labels directly.

        diff_label_indices = tf.where(tf.not_equal(p_target_ij, 0.5))  # indices of pairs have different labels.
        new_p_target_ij = tf.gather(p_target_ij, diff_label_indices)

        new_o_ij = tf.gather(o_ij, diff_label_indices) * learn2rank_sigma
        cost = tf.reduce_sum(
            - new_p_target_ij * new_o_ij + tf.log(1.0 + tf.exp(new_o_ij))) + tf.losses.get_regularization_loss()

    with tf.name_scope('optimization'):
        global_step = tf.Variable(0, name="tr_global_step", trainable=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost,
                                                                                            global_step=global_step)

    with tf.Session(config=tfconfig) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        if not os.path.exists(cfg.model_save_path):
            os.mkdir(cfg.model_save_path)
        # mainloop
        ep = cfg.risk_training_epochs
        shuffle_data = True
        # bs = np.maximum(int(data_len * 0.05), 2)
        bs = 8
        batch_num = data_len // bs + (1 if data_len % bs else 0)
        print("\n- Set number of training epochs={}.".format(ep))
        print("- The train data batch size={}, batch number={}.".format(bs, batch_num))
        print("- Shuffle data each epoch: {}".format(shuffle_data))
        if batch_num > 1:
            print("- The average precision is approximated! Cause there exists multiple batches and "
                  "the rank positions of risky data are partially evaluated.")
        first_last_loss = []
        for epoch in trange(ep, desc="Train Risk Model"):
            if shuffle_data:
                # reshuffle training data
                index = np.random.permutation(np.arange(data_len))
            else:
                index = np.arange(data_len)  # non shuffle version
            accuracy = 0
            average_precision = []
            loss = 0.
            a_all = np.array([[0] * class_num])
            b_all = np.array([[0] * class_num])
            c_all = np.array([[0] * class_num])
            d_all = np.array([[0] * class_num])
            all_big_mu = np.array([[0] * class_num])
            all_big_sigma = np.array(([[0] * class_num]))
            # mini_batch training
            for i in range(batch_num):
                machine_results_batch = machine_results[index][bs * i:bs * i + bs]
                y_batch = _y[index][bs * i:bs * i + bs]
                y_mul_batch = _y_mul[index][bs * i:bs * i + bs]
                mus_rule_batch = _mus_rule_X[index][bs * i:bs * i + bs]
                mus_machine_batch = _mus_machine_X[index][bs * i:bs * i + bs]
                sigmas_rule_batch = _sigmas_rule_X[index][bs * i:bs * i + bs]
                sigmas_machine_batch = _sigmas_machine_X[index][bs * i:bs * i + bs]
                activation_rule_batch = _feature_activation_rule_matrix[index][bs * i:bs * i + bs]
                activation_machine_batch = _feature_activation_machine_matrix[index][bs * i:bs * i + bs]
                machine_one_batch = _machine_one[index][bs * i:bs * i + bs]
                y_activate_batch = _y_activate[index][bs * i: bs * i + bs]

                '''
                    [optimizer, global_step, prob, discrete_w, cost,
                    acc, alpha, discrete_variances, continuous_variances, label_match_value, 
                    label_unmatch_value, weight_func_a, weight_func_b, weight_func_c, avg_precision,
                    learn2rank_sigma, prob_mul, machine_label, change_machine_label, descent_order,
                    prob_2_label, pairwise_probs, pairwise_labels, pairwise_risky_values, mus_rule, 
                    mus_machine, machine_mus_vector, machine_weights, Fr_alpha,
                    Fr_alpha_bar, big_mu, big_sigma,],
                '''

                return_values = sess.run(

                    [optimizer, global_step, prob, discrete_w, cost,
                     alpha, alpha, discrete_variances, continuous_variances, label_match_value,
                     label_unmatch_value, weight_func_a, weight_func_b, weight_func_c, weight_func_c,
                     learn2rank_sigma, prob_mul, Fr_alpha, prob__, machine_weights,
                     _a, _b, weight_vector, _c, _d,
                     big_mu, big_sigma, ],

                    feed_dict={machine_label: machine_results_batch,
                               risk_y: y_batch,
                               risk_y_mul: y_mul_batch,
                               mus_rule: mus_rule_batch,
                               mus_machine: mus_machine_batch,
                               sigmas_rule: sigmas_rule_batch,
                               sigmas_machine: sigmas_machine_batch,
                               feature_rule_matrix: activation_rule_batch,
                               feature_machine_matrix: activation_machine_batch,
                               machine_one: machine_one_batch,
                               y_activate: y_activate_batch,
                               pairwise_risky_values: [[0., 0.]],  # This fist element will be removed.
                               pairwise_risky_labels: [[0., 0.]]})
                _ = return_values[0]
                step = return_values[1]
                prob_ = return_values[2]
                rule_learn_w = return_values[3]
                cost_ = return_values[4]
                # acc_ = return_values[5]
                alpha_ = return_values[6]
                discrete_variances_ = return_values[7]
                continuous_variances_ = return_values[8]
                label_match_value_ = return_values[9]
                label_unmatch_value_ = return_values[10]
                _func_a = return_values[11]
                _func_b = return_values[12]
                _func_c = return_values[13]
                _avg_precision = return_values[14]
                _l2rank_sigma = return_values[15]
                _prob_mul = return_values[16]
                __prob_mul = return_values[18]
                _machine_weight = return_values[19]
                __a = return_values[20]
                __b = return_values[21]
                _weight_vector = return_values[22]
                __c = return_values[23]
                __d = return_values[24]
                __big_mu = return_values[25]
                __big_sigma = return_values[26]

                a_all = np.concatenate([a_all, __a], axis=0)
                b_all = np.concatenate([b_all, __b], axis=0)
                c_all = np.concatenate([c_all, __c], axis=0)
                d_all = np.concatenate([d_all, __d], axis=0)
                all_big_mu = np.concatenate([all_big_mu, __big_mu], axis=0)
                all_big_sigma = np.concatenate([all_big_sigma, __big_sigma], axis=0)

                np.save('a.npy', a_all)
                np.save('b.npy', b_all)
                np.save('c.npy', c_all)
                np.save('d.npy', d_all)
                np.save('big_mu', all_big_mu)
                np.save('big_sigma', all_big_sigma)

                # accuracy += acc_
                loss += cost_
                # logging.info("epoch={}/{},loss={}".format(epoch+1, i, cost_))
                # if not math.isnan(_avg_precision):
                #    average_precision.append(_avg_precision)
                # Estimated sum of rule feature weights in each pair.
                # rw_each = np.sum(w_.reshape(-1)) * 10 / discrete_m

                # print("epoch={}, loss={},  ".format(epoch + 1,
                #                                     loss,
                #                                     ))
                # print("func_a={}, func_b={}, func_c={}, sigma={},".format(_func_a,
                #                                                           _func_b,
                #                                                           _func_c,
                #                                                           _l2rank_sigma,
                #                                                           ))

                saver.save(sess, os.path.join(cfg.model_save_path, 'model.ckpt'), global_step=epoch)
            logging.info("epoch={}, loss={:.5f},func_a={}, func_c={}, "
                         "sigma={}".format(epoch + 1,
                                           loss,
                                           _func_a,
                                           _func_c,
                                           _l2rank_sigma,
                                           ))
            if epoch == 0 or epoch >= (ep - 5):
                first_last_loss.append(loss)
        # print('------------------------discrete_variances_')
        # print(discrete_variances_)
        # print('-----------------------continuous_variances_')
        # print(continuous_variances_)
        # print('-----------------------weight')
        # print(rule_learn_w)

        print("\n-- The Evolution Loss of Risk Model: {} --> ... --> {}.".format(first_last_loss[0],
                                                                                 first_last_loss[1:]))
        return rule_learn_w, alpha_, discrete_variances_, continuous_variances_, label_match_value_, label_unmatch_value_, [
            _func_a, _func_b, _func_c]


def predict(machine_results, _mus_rule_X, _mus_machine_X, _sigmas_rule_X, _sigmas_machine_X,
            _feature_activation_rule_matrix, _feature_activation_machine_matrix, _machine_one, _y_activate,
            _rule_learn_w, _machine_learn_w, _alpha,
            _rule_variance, _machine_variance, _match_value, _unmatch_value, func_parameters, rule_activation,
            apply_learn_v=True):
    """
    All input parameters are numpy array type.
    :param machine_results: shape: (n, 1), the machine results.
    :param _mus_X: shape: (n, m), the means of m risk features. !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _sigmas_X: shape: (n, m), the variances of m risk features. !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _feature_activation_matrix: shape (n, m), indicate which features are used in each data.
                                        !!! Scipy Sparse Matrix (lil_matrix) !!!
    :param _w: shape: (m, 1), the learned feature weights.
    :param _alpha: the confidence for risk analysis.
    :param _variances: shape: (m, 1), the learned variances.
    :param _match_value:
    :param _unmatch_value:
    :param func_parameters: the parameters for gaussian function
    :param apply_learn_v: If False, use _sigma_X instead of _variances.
    :return:
    """
    tf.reset_default_graph()
    # m = _mus_X.shape[1]  # number of features
    continuous_m = cfg.interval_number_4_continuous_value  # *  class_num
    discrete_m = _mus_rule_X.shape[2]

    with tf.name_scope('constants'):
        # alpha = tf.constant(config.get_risk_confidence(), dtype=tf.double)
        a = tf.constant(0.0, dtype=tf.double)
        b = tf.constant(1.0, dtype=tf.double)

    with tf.name_scope('inputs'):
        # Variables (Vectors)
        machine_label = tf.placeholder(tf.int8, name='ML')  # (n, 1)

        mus_rule = tf.placeholder(tf.double, name='mu_rule')  # (n, 10, 30)
        mus_machine = tf.placeholder(tf.double, name='mu_mac')  # (n, 10, 50)
        sigmas_rule = tf.placeholder(tf.double, name='sigma_rule')  # (n, 10, 30)
        sigmas_machine = tf.placeholder(tf.double, name='sigma_mac')  # (n, 10, 50)
        feature_rule_matrix = tf.placeholder(tf.double, name='featureRuleMatrix')  # (n, 10, 30)
        feature_machine_matrix = tf.placeholder(tf.double, name='featureMacMatrix')  # (n, 10, 50)
        machine_one = tf.placeholder(tf.double, [None, class_num], name="machine_one")
        y_activate = tf.placeholder(tf.double, [None, class_num], name="y_activate")

        # mus = tf.placeholder(tf.double, name='mu')  # (n, m)
        # sigmas = tf.placeholder(tf.double, name='sigma')  # (n, m)
        rule_learn_w = tf.placeholder(tf.double, name='r_w')  # (m ,1)
        machine_learn_w = tf.placeholder(tf.double, name='m_w')
        alpha = tf.placeholder(tf.double, name='alpha')
        # feature_matrix = tf.placeholder(tf.double, name='featureMatrix')  # (n, m)
        rule_variance = tf.placeholder(tf.double, name='rule_variance')
        machine_variance = tf.placeholder(tf.double, name='machine_variance')
        match_value = tf.placeholder(tf.double, name='match_value')
        unmatch_value = tf.placeholder(tf.double, name='unmatch_value')
        weight_func_a = tf.placeholder(tf.double, name='weight_function_w')
        weight_func_b = tf.placeholder(tf.double, name='weight_function_mean')
        weight_func_c = tf.placeholder(tf.double, name='weight_function_variance')

    with tf.name_scope('prediction'):
        if not APPLY_WEIGHT_FUNC or not apply_learn_v:
            pass
            '''
            big_mu = tf.matmul(mus, w) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            # -- 1. use pre-set variances. --
            if not apply_learn_v or not LEARN_VARIANCE:
                print("- Apply pre-set variances.")
                big_sigma = tf.matmul(sigmas, tf.square(w)) + 1e-10  # (n, m) * (m, 1) -> (n, 1)
            else:
                # -- 2. learn variances. --
                print("- Apply learned variances.")
                learn_sigmas_X = feature_matrix * tf.reshape(variances, [1, m])
                big_sigma = tf.matmul(learn_sigmas_X, tf.square(w)) + 1e-10

            # Normalize the weights of features in each pair.
            # feature_matrix = tf.cast(tf.math.not_equal(mus, .0), tf.double)
            weight_vector = tf.matmul(feature_matrix, w)  # (n, m) * (m, 1) -> (n, 1)
            big_mu = big_mu / weight_vector
            big_sigma = big_sigma / (tf.square(weight_vector))
            '''
        else:
            # Need to calculate weights for each probability.
            discrete_w = rule_learn_w
            # sparse matrix, only one position has value.
            machine_mus = mus_machine
            machine_mus_vector = tf.reshape(tf.reduce_sum(machine_mus, axis=2), [-1, class_num])
            machine_weights = gaussian_function(weight_func_a, weight_func_b, weight_func_c, machine_mus_vector)
            machine_weights = tf.reshape(machine_weights, [-1, class_num])

            _a = tf.reduce_sum(tf.multiply(mus_rule, discrete_w), axis=2)
            _b = tf.multiply(machine_mus_vector, machine_weights)

            # mus_rule = tf.multiply(mus_rule, feature_rule_matrix)
            big_mu = tf.reduce_sum(tf.multiply(mus_rule, discrete_w), axis=2) + tf.multiply(machine_mus_vector,
                                                                                            machine_weights) + 1e-10
            # big_mu = tf.reduce_sum(tf.multiply(rule_mus, discrete_w), axis=2) + 1e-10

            if not LEARN_VARIANCE:
                # # -- ** 1. use pre-set variances. ** --
                # print("- Apply pre-set variances.")
                # use_sigmas = sigmas
                pass
            else:
                # -- ** 2. learn variances. ** --
                print("- Apply learned variances.")

                rule_sigmas = tf.multiply(feature_rule_matrix, rule_variance)
                machine_sigmas = tf.multiply(feature_machine_matrix, machine_variance)
            # rule_sigmas = rule_sigmas
            # machine_sigmas = machine_sigmas
            machine_sigmas_vector = tf.reshape(tf.reduce_sum(machine_sigmas, axis=2), [-1, class_num])

            _c = tf.reduce_sum(tf.multiply(rule_sigmas, discrete_w * discrete_w), axis=2)
            _d = tf.multiply(machine_sigmas_vector, machine_weights * machine_weights)

            big_sigma = tf.reduce_sum(tf.multiply(rule_sigmas, discrete_w * discrete_w), axis=2) + tf.multiply(
                machine_sigmas_vector,
                machine_weights * machine_weights) + 1e-10
            # big_sigma = tf.reduce_sum(tf.multiply(rule_sigmas, discrete_w * discrete_w), axis=2) + 1e-10

            # Normalize the weights of features in each pair.
            weight_vector = tf.reduce_sum(tf.multiply(feature_rule_matrix, discrete_w),
                                          axis=2) + machine_weights + 1e-10
            # weight_vector = tf.reduce_sum(tf.multiply(feature_rule_matrix, discrete_w), axis=2) + 1e-10
            weight_rule = discrete_w
            weight_rule_class = tf.reduce_sum(tf.multiply(feature_rule_matrix, discrete_w), axis=2)
            weight_mac = machine_weights

            big_mu = big_mu / weight_vector  # n * 10
            big_sigma = big_sigma / (tf.square(weight_vector))  # n * 10

        # Truncated normal distribution.

        machine_label = tf.cast(machine_label, tf.double)
        machine_one = tf.cast(machine_one, tf.double)
        y_activate = tf.cast(y_activate, tf.double)
        y_activate = tf.reshape(y_activate, [-1, class_num])
        # -- Option 1: Use match value: 1, unmatch value: 0. --
        # prob = Fr_alpha * (tf.ones_like(machine_label) - machine_label) + (
        #        tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_label

        print('-------------------------------------big_mu')
        prob = tf.zeros_like(machine_label)
        prob = tf.cast(prob, tf.double)
        # prob_ = tf.zeros_like(machine_label)
        prob_mul = tf.zeros_like(big_mu)
        prob = tf.zeros_like(machine_label)
        Fr_alpha = my_truncated_normal_ppf(alpha, a, b, big_mu, tf.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, big_mu, tf.sqrt(big_sigma))
        prob_mul = Fr_alpha * (tf.ones_like(machine_one) - machine_one) + (
                tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_one
        logging.info(prob_mul)
        prob = tf.reduce_sum(prob_mul * machine_one, axis=1)

        '''
        machine_label = tf.cast(machine_label, tf.double)
        prob = tf.zeros_like(machine_label)
        prob = tf.cast(prob, tf.double)
        prob_mul = tf.zeros_like(machine_label)
        for i in range(class_num):
            Fr_alpha = my_truncated_normal_ppf(alpha, a, b, tf.slice(big_mu,[0,i],[-1,1]), tf.sqrt(tf.slice(big_sigma,[0,i],[-1,1])))
            Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, tf.slice(big_mu,[0,i],[-1,1]), tf.sqrt(tf.slice(big_sigma,[0,i],[-1,1])))
            #rule_begin = rules_idx[i][0]
            #rule_end = rules_idx[i][1]
            #print(rule_begin, rule_end)
            #Fr_alpha = my_truncated_normal_ppf(alpha, a, b, tf.slice(big_mu, [rule_begin, 0], [rule_end - rule_begin - 1, 1]), tf.sqrt(tf.slice(big_sigma, [rule_begin, 0], [rule_end - rule_begin - 1, 1] ) ))
            #Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, tf.slice(big_mu, [rule_begin,0], [rule_end - rule_begin - 1,1]), tf.sqrt(tf.slice(big_sigma, [rule_begin, 0], [rule_end - rule_begin - 1, 1] ) ))
            #Fr_alpha = my_truncated_normal_ppf(alpha, a, b, big_mu, tf.sqrt(big_sigma))
            #Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, big_mu, tf.sqrt(big_sigma))
            change_machine_label =  tf.where(tf.equal(machine_label, i), tf.ones_like(machine_label), tf.zeros_like(machine_label))
            #change_machine_label = tf.where(tf.equal(machine_label, i), tf.zeros_like(machine_label), tf.ones_like(machine_label))
            change_machine_label = tf.cast(change_machine_label, tf.double)
            #prob += Fr_alpha * (tf.ones_like(change_machine_label) - change_machine_label) + (
            #    tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * change_machine_label

            x = Fr_alpha * (tf.ones_like(change_machine_label) - change_machine_label) + (tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * change_machine_label
            prob_mul = tf.concat([prob_mul, x], 1)

            #Fr_alpha * (tf.ones_like(change_machine_label) - change_machine_label) + (
            prob += tf.where(tf.equal(machine_label, i),  (
                tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * change_machine_label, tf.zeros_like(change_machine_label) )

            #prob += tf.where(tf.equal(machine_label, i), (tf.slice(big_mu,[0,i],[-1,1]) - Fr_alpha_bar) *change_machine_label, tf.zeros_like(change_machine_label))
        '''

        # -- Option 2: Use match value: learning, unmatch value: learning. --
        # prob = (Fr_alpha - tf.ones_like(Fr_alpha) * unmatch_value) * (tf.ones_like(machine_label) - machine_label) + (
        #         tf.ones_like(Fr_alpha_bar) * match_value - Fr_alpha_bar) * machine_label

        # -- Option 3: Use match value: mean, unmatch value: mean. --
        # prob = (Fr_alpha - big_mu) * (tf.ones_like(machine_label) - machine_label) + (
        #         big_mu - Fr_alpha_bar) * machine_label

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # To handle ResourceExhaustedError (see above for traceback):
        # OOM when allocating tensor with shape[24850,3836]..., use batch prediction.
        data_len = _mus_rule_X.shape[0]
        # bs = np.maximum(int(data_len * 0.05), 2)
        bs = 32
        batch_num = data_len // bs + (1 if data_len % bs else 0)

        print("The test data batch size={}, batch number={}".format(bs, batch_num))

        prob1 = tf.constant([[0.]], dtype=tf.double)
        big_mu1 = tf.constant([[0.]], dtype=tf.double)
        big_sigma1 = tf.constant([[0.]], dtype=tf.double)
        pro_all = np.array([0.])
        pro_mul_all = np.array([[0.] * (class_num)])

        # mini_batch
        for i in range(batch_num):
            machine_results_batch = machine_results[bs * i:bs * i + bs]
            mus_rule_batch = _mus_rule_X[bs * i:bs * i + bs]
            mus_machine_batch = _mus_machine_X[bs * i:bs * i + bs]
            sigmas_rule_batch = _sigmas_rule_X[bs * i:bs * i + bs]
            sigmas_machine_batch = _sigmas_machine_X[bs * i:bs * i + bs]
            activate_features_rule_batch = _feature_activation_rule_matrix[bs * i:bs * i + bs]
            activate_features_machine_batch = _feature_activation_machine_matrix[bs * i:bs * i + bs]
            machine_one_batch = _machine_one[bs * i: bs * i + bs]
            y_activate_batch = _y_activate[bs * i: bs * i + bs]
            print(machine_results_batch.shape)
            print(mus_rule_batch.shape)
            print(mus_machine_batch.shape)
            print(sigmas_rule_batch.shape)
            print(sigmas_machine_batch.shape)
            print(activate_features_rule_batch.shape)
            print(activate_features_machine_batch.shape)
            print(_rule_variance.shape)
            print(_machine_variance.shape)
            print(rule_learn_w)

            prob_b, big_mu_b, big_sigma_b, prob_mul_b, a_test, b_test, c_test, d_test, wv_test, _weight_rule, \
            _weight_rule_class, _weight_mac, fr, fr_bar, r_sigma, m_sigma, r_mus, m_mus, mac_prob = sess.run(
                [prob, big_mu, big_sigma, prob_mul, _a, _b, _c, _d, weight_vector, weight_rule, weight_rule_class,
                 weight_mac, Fr_alpha, Fr_alpha_bar, rule_sigmas, machine_sigmas, mus_rule, machine_mus, machine_mus_vector],
                feed_dict={machine_label: machine_results_batch,
                           mus_rule: mus_rule_batch,
                           mus_machine: mus_machine_batch,
                           sigmas_rule: sigmas_rule_batch,
                           sigmas_machine: sigmas_machine_batch,
                           feature_rule_matrix: activate_features_rule_batch,
                           feature_machine_matrix: activate_features_machine_batch,
                           rule_learn_w: _rule_learn_w,
                           machine_learn_w: _machine_learn_w,
                           machine_one: machine_one_batch,
                           y_activate: y_activate_batch,
                           alpha: _alpha,
                           rule_variance: _rule_variance,
                           machine_variance: _machine_variance,
                           match_value: _match_value,
                           unmatch_value: _unmatch_value,
                           weight_func_a: func_parameters[0],
                           weight_func_b: func_parameters[1],
                           weight_func_c: func_parameters[2]})
            # pro_all.

            np.save('test_rule_mu.npy', a_test)
            np.save('test_mac_mu.npy', b_test)
            np.save('test_rule_sigma.npy', c_test)
            np.save('test_mac_sigma.npy', d_test)
            np.save('test_big_mu.npy', big_mu_b)
            np.save('test_big_sigma.npy', big_sigma_b)
            np.save('rule_w.npy',_weight_rule)
            np.save('rule_class.npy', _weight_rule_class)
            np.save('mac_w.npy', _weight_mac)
            np.save('fr.npy', fr)
            np.save('fr_bar.npy', fr_bar)
            np.save('prob_mul.npy', prob_mul_b)
            np.save('prob.npy', prob_b)
            np.save('rule_sigma.npy', r_sigma)
            np.save('mac_sigma.npy', m_sigma)
            np.save('rule_mus.npy', r_mus)
            np.save('mac_mus.npy', m_mus)
            np.save('mac_prob.npy', mac_prob)

            # print('-----------------------------------x')
            # print(x)
            pro_all = np.concatenate([pro_all, prob_b])
            pro_mul_all = np.concatenate([pro_mul_all, prob_mul_b])
            # prob1 = tf.concat([prob1, prob_b], 0)
            # big_mu1 = tf.concat([big_mu1, big_mu_b], 0)
            # big_sigma1 = tf.concat([big_sigma1, big_sigma_b], 0)
            # prob1, big_mu1, big_sigma1 = sess.run([prob1, big_mu1, big_sigma1])
            # prob1 = sess.run([prob1])
        # Remove the meaningless first one.
        # prob1 = prob1[1:]
        # print('--------------------------------prob1')
        pro_all = pro_all[1:]  # prob_b#
        pro_mul_all = pro_mul_all[1:, :]  # prob_mul_b#

        big_mu1 = None
        big_sigma1 = None
        # return prob_b, big_mu1, big_sigma1
        return pro_all, big_mu1, big_sigma1


def testf():
    m_results = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0]).reshape([-1, 1])
    train_X = np.array([[1, 1, 1, 1]] * 10)
    train_y = np.array([0, 1, 0, 1, 1, 1, 0, 0, 1, 1]).reshape([-1, 1])
    mus = np.array([0.2, 0.8, 0.4, 0.9])
    sigmas = (np.array([0.1, 0.15, 0.12, 0.05]) ** 2)
    # sigmas = np.array([0.1, 0.1, 0.1, 0.1])
    mus_X = train_X * mus
    sigmas_X = train_X * sigmas
    weights, alpha, variances, match_value, unmatch_value, func_params = fit(m_results, sp.lil_matrix(mus_X),
                                                                             sp.lil_matrix(sigmas_X), train_y,
                                                                             sp.lil_matrix(train_X),
                                                                             mus.reshape([-1, 1]),
                                                                             None)
    print(weights.shape)
    print("Discrete feature weights:", weights)
    print("Function parameters:", func_params)
    print("Match value: {}, Unmatch value: {}".format(match_value, unmatch_value))

    print("Initial variances: {}".format(sigmas))
    print("Learned variances: {}".format(variances))

    # weights, alpha, variances, match_value, unmatch_value, func_params = fit(m_results, sp.lil_matrix(mus_X),
    #                                                                          sp.lil_matrix(sigmas_X), train_y,
    #                                                                          sp.lil_matrix(train_X))

    test_X = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]])
    test_m_results = np.array([0, 0, 1, 1, 0]).reshape([-1, 1])
    test_mu_X = test_X * mus
    test_sigmas_X = test_X * sigmas
    probs, pair_mus, pair_sigmas = predict(test_m_results, sp.lil_matrix(test_mu_X), sp.lil_matrix(test_sigmas_X),
                                           weights, alpha, sp.lil_matrix(test_X), variances, match_value, unmatch_value,
                                           func_params)
    print(probs)
    print(probs.reshape(-1))
    print(pair_mus.reshape(-1))
    print(pair_sigmas.reshape(-1))


if __name__ == '__main__':
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    testf()
