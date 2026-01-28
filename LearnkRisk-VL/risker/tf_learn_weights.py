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
def fit(_machine_results, _rule_mu, _machine_mu, _feature_activation_rule_matrix, _feature_activation_machine_matrix,
        _machine_one, _risk_labels, _risk_mul_labels, init_variance=None):
    tf.reset_default_graph()

    print("- Different weights for different probabilities (use Gaussian function).")
    continuous_m = cfg.interval_number_4_continuous_value
    discrete_m = _rule_mu.shape[2]

    data_len = len(_risk_labels)  # number of data
    np.random.seed(2020)
    tf.set_random_seed(2020)

    # The mean number of activated rule features.
    _feature_activation_matrix = np.concatenate(
        (np.array(_feature_activation_rule_matrix), np.array(_feature_activation_machine_matrix)), axis=2)
    max_w = 1.0 / np.max(np.sum(_feature_activation_matrix, axis=2))
    class_count = Counter(_risk_labels.reshape(-1))
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
        weight_func_b = tf.constant([0.5], dtype=tf.double)
        # weight_func_c = tf.constant(1.0, dtype=tf.double)

    with tf.name_scope('inputs'):
        # Variables (Vectors)
        machine_label = tf.placeholder(tf.int8, name='ML')  # (n, 1)
        risk = tf.placeholder(tf.int64, name='risk')  # (n, 1)
        risk_mul = tf.placeholder(tf.int64, [None, class_num], name='risk_mu')  # (n, 10)
        rule_mu = tf.placeholder(tf.double, [None, class_num, discrete_m], name='rule_mu')  # (n, 10, 30)
        machine_mu = tf.placeholder(tf.double, [None, class_num, continuous_m], name='mac_mu')  # (n, 10, 50)
        feature_rule_matrix = tf.placeholder(tf.double, [None, class_num, discrete_m],
                                             name='featureRuleMatrix')  # (n, 10, 30)
        feature_machine_matrix = tf.placeholder(tf.double, [None, class_num, continuous_m],
                                                name='featureMacMatrix')  # (n, 10, 50)
        machine_one = tf.placeholder(tf.double, [None, class_num], name="machine_one")
        # parameters for learning to rank
        pairwise_risky_values = tf.placeholder(tf.double, name='pairwise_values')
        pairwise_risky_labels = tf.placeholder(tf.double, name='pairwise_labels')

        rule_w = tf.get_variable(name='rule_w', shape=[class_num, discrete_m],
                                 dtype=tf.double,
                                 initializer=tf.random_uniform_initializer(0., max_w),
                                 regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                 constraint=lambda t: tf.abs(t)
                                 # constraint=lambda t: tf.clip_by_value(t, 1, 10)
                                 )  # (m ,1)

        rule_variances = tf.get_variable(name='rule_variances',
                                         shape=[class_num, discrete_m],
                                         dtype=tf.double,
                                         initializer=tf.random_uniform_initializer(0., 1.),
                                         # constraint=lambda t: tf.abs(t),
                                         constraint=lambda t: tf.clip_by_value(t, 0., 1.)
                                         )
        machine_variances = tf.get_variable(name='machine_variances',
                                            shape=[continuous_m],
                                            dtype=tf.double,
                                            initializer=tf.random_uniform_initializer(0., 1.),
                                            # constraint=lambda t: tf.abs(t),
                                            constraint=lambda t: tf.clip_by_value(t, 0., 1.)
                                            )

        learn2rank_sigma = tf.get_variable(name='learning_to_rank_sigma',
                                           initializer=tf.constant(1.0, dtype=tf.double),
                                           regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                           # constraint=lambda t: tf.abs(t)
                                           )

        weight_func_a = tf.get_variable(name='weight_function_w',
                                        initializer=tf.constant([1.0], dtype=tf.double),
                                        regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                        constraint=lambda t: tf.abs(t))
        # weight_func_b = tf.get_variable(name='weight_function_mean',
        #                                 initializer=tf.constant(0.5, dtype=tf.double),
        #                                 constraint=lambda t: tf.abs(t))
        weight_func_c = tf.get_variable(name='weight_function_variance',
                                        initializer=tf.constant([0.5], dtype=tf.double),
                                        regularizer=tf.contrib.layers.l1_l2_regularizer(l1, l2),
                                        constraint=lambda t: tf.abs(t))

    with tf.name_scope('prediction'):
        machine_mus_vector = tf.reshape(tf.reduce_sum(machine_mu, axis=2), [-1, class_num])
        machine_weights = gaussian_function(weight_func_a, weight_func_b, weight_func_c,
                                            tf.reshape(machine_mus_vector, [-1, 1]))
        machine_weights = tf.reshape(machine_weights, [-1, class_num])
        print(machine_weights)

        big_mu = tf.reduce_sum(tf.multiply(rule_mu, rule_w), axis=2) + tf.multiply(
            machine_mus_vector, machine_weights) + 1e-10

        print("- Learning variances.")
        # In the newest solution, the above 'variances' is actually Relative Standard Deviation.
        # Here we transform it to real variances.


        rule_sigmas = tf.multiply(feature_rule_matrix, rule_variances)
        machine_sigmas = tf.multiply(feature_machine_matrix, machine_variances)
        machine_sigmas_vector = tf.reshape(tf.reduce_sum(machine_sigmas, axis=2), [-1, class_num])

        big_sigma = tf.reduce_sum(tf.multiply(rule_sigmas, rule_w * rule_w), axis=2) + tf.multiply(
            machine_sigmas_vector,
            machine_weights * machine_weights) + 1e-10

        # Normalize the weights of features in each pair.
        # rule_activate_matrix = feature_rule_matrix
        weight_vector = tf.reduce_sum(tf.multiply(feature_rule_matrix, rule_w),
                                      axis=2) + machine_weights + 1e-10

        big_mu = big_mu / weight_vector  # n * 10
        big_sigma = big_sigma / (tf.square(weight_vector))  # n * 10

        # Truncated normal distribution.

        machine_label = tf.cast(machine_label, tf.double)
        machine_one = tf.cast(machine_one, tf.double)
        # -- Option 1: Use match value: 1, unmatch value: 0. --
        # prob = Fr_alpha * (tf.ones_like(machine_label) - machine_label) + (
        #        tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_label

        Fr_alpha = my_truncated_normal_ppf(alpha, a, b, big_mu, tf.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, big_mu, tf.sqrt(big_sigma))

        prob_mul = Fr_alpha * (tf.ones_like(machine_one) - machine_one) + (
                tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_one
        prob = tf.reduce_sum(prob_mul * machine_one, axis=1)

        _prob_mul = tf.reshape(prob_mul, [-1, 1])
        _risk_mul = tf.reshape(risk_mul, [-1, 1])

        # -- Option 2: Use match value: learning, unmatch value: learning. --
        # prob = (Fr_alpha - tf.ones_like(Fr_alpha) * label_unmatch_value) * (
        #             tf.ones_like(machine_label) - machine_label) + (
        #                tf.ones_like(Fr_alpha_bar) * label_match_value - Fr_alpha_bar) * machine_label

        # -- Option 3: Use match value: mean, unmatch value: mean. --
        # prob = (Fr_alpha - big_mu) * (tf.ones_like(machine_label) - machine_label) + (
        #            big_mu - Fr_alpha_bar) * machine_label

    with tf.name_scope('loss'):
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
        pairwise_probs = get_pairwise_combinations(_prob_mul, pairwise_risky_values)
        pairwise_labels = get_pairwise_combinations(tf.cast(_risk_mul, tf.double), pairwise_risky_labels)
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
        bs = 4
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
            all_big_mu = np.array([[0] * class_num])
            all_big_sigma = np.array(([[0] * class_num]))
            # mini_batch training
            for i in range(batch_num):
                machine_results_batch = _machine_results[index][bs * i:bs * i + bs]
                risk_batch = _risk_labels[index][bs * i:bs * i + bs]
                risk_mul_batch = _risk_mul_labels[index][bs * i:bs * i + bs]
                rule_mu_batch = _rule_mu[index][bs * i:bs * i + bs]
                machine_mu_batch = _machine_mu[index][bs * i:bs * i + bs]
                activation_rule_batch = _feature_activation_rule_matrix[index][bs * i:bs * i + bs]
                activation_machine_batch = _feature_activation_machine_matrix[index][bs * i:bs * i + bs]
                machine_one_batch = _machine_one[index][bs * i:bs * i + bs]

                return_values = sess.run(

                    [optimizer, global_step, prob, rule_w, cost,
                     alpha, rule_variances, machine_variances, label_match_value, label_unmatch_value,
                     weight_func_a, weight_func_b, weight_func_c, learn2rank_sigma, prob_mul,
                     Fr_alpha, Fr_alpha_bar, machine_weights, big_mu, big_sigma, pairwise_probs],

                    feed_dict={machine_label: machine_results_batch,
                               risk: risk_batch,
                               risk_mul: risk_mul_batch,
                               rule_mu: rule_mu_batch,
                               machine_mu: machine_mu_batch,
                               feature_rule_matrix: activation_rule_batch,
                               feature_machine_matrix: activation_machine_batch,
                               machine_one: machine_one_batch,
                               pairwise_risky_values: [[0., 0.]],  # This fist element will be removed.
                               pairwise_risky_labels: [[0., 0.]]})
                _ = return_values[0]
                step_ = return_values[1]
                prob_ = return_values[2]
                rule_learn_w = return_values[3]
                cost_ = return_values[4]
                alpha_ = return_values[5]
                rule_learn_variances = return_values[6]
                machine_learn_variances = return_values[7]
                label_match_value_ = return_values[8]
                label_unmatch_value_ = return_values[9]
                _func_a = return_values[10]
                _func_b = return_values[11]
                _func_c = return_values[12]
                _l2rank_sigma = return_values[13]
                _prob_mul = return_values[14]
                _big_mu = return_values[18]
                _big_sigma = return_values[19]
                logging.info('-----------------------')
                logging.info(return_values[20])
                logging.info(return_values[20].shape)

                all_big_mu = np.concatenate([all_big_mu, _big_mu], axis=0)
                all_big_sigma = np.concatenate([all_big_sigma, _big_sigma], axis=0)

                # np.save('big_mu', all_big_mu)
                # np.save('big_sigma', all_big_sigma)

                # accuracy += acc_
                loss += cost_
                logging.info("epoch={}/{},loss={}".format(epoch+1, i, cost_))
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

        print("\n-- The Evolution Loss of Risk Model: {} --> ... --> {}.".format(first_last_loss[0],
                                                                                 first_last_loss[1:]))
        return rule_learn_w, alpha_, rule_learn_variances, machine_learn_variances, label_match_value_, \
               label_unmatch_value_, [_func_a, _func_b, _func_c]


def predict(_machine_results, _rule_mu, _machine_mu,
            _feature_activation_rule_matrix, _feature_activation_machine_matrix, _machine_one,
            _rule_learn_w, _alpha, _rule_variance, _machine_variance,
            _match_value, _unmatch_value, func_parameters,
            apply_learn_v=True):

    tf.reset_default_graph()
    continuous_m = cfg.interval_number_4_continuous_value  # *  class_num
    discrete_m = _rule_mu.shape[2]

    with tf.name_scope('constants'):
        # alpha = tf.constant(config.get_risk_confidence(), dtype=tf.double)
        a = tf.constant(0.0, dtype=tf.double)
        b = tf.constant(1.0, dtype=tf.double)

    with tf.name_scope('inputs'):
        # Variables (Vectors)
        machine_label = tf.placeholder(tf.int8, name='ML')  # (n, 1)

        rule_mu = tf.placeholder(tf.double, name='rule_mu')  # (n, 10, 30)
        machine_mu = tf.placeholder(tf.double, name='mac_mu')  # (n, 10, 50)
        feature_rule_matrix = tf.placeholder(tf.double, name='featureRuleMatrix')  # (n, 10, 30)
        feature_machine_matrix = tf.placeholder(tf.double, name='featureMacMatrix')  # (n, 10, 50)
        machine_one = tf.placeholder(tf.double, [None, class_num], name="machine_one")

        # mus = tf.placeholder(tf.double, name='mu')  # (n, m)
        # sigmas = tf.placeholder(tf.double, name='sigma')  # (n, m)
        rule_learn_w = tf.placeholder(tf.double, name='r_w')  # (m ,1)
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
        # Need to calculate weights for each probability.
        rule_w = rule_learn_w
        # sparse matrix, only one position has value.
        machine_mus_vector = tf.reshape(tf.reduce_sum(machine_mu, axis=2), [-1, class_num])
        machine_weights = gaussian_function(weight_func_a, weight_func_b, weight_func_c, machine_mus_vector)
        machine_weights = tf.reshape(machine_weights, [-1, class_num])

        _a = tf.reduce_sum(tf.multiply(rule_mu, rule_w), axis=2)
        _b = tf.multiply(machine_mus_vector, machine_weights)

        big_mu = tf.reduce_sum(tf.multiply(rule_mu, rule_w), axis=2) + tf.multiply(machine_mus_vector,
                                                                                            machine_weights) + 1e-10

        rule_sigmas = tf.multiply(feature_rule_matrix, rule_variance)
        machine_sigmas = tf.multiply(feature_machine_matrix, machine_variance)
        machine_sigmas_vector = tf.reshape(tf.reduce_sum(machine_sigmas, axis=2), [-1, class_num])

        _c = tf.reduce_sum(tf.multiply(rule_sigmas, rule_w * rule_w), axis=2)
        _d = tf.multiply(machine_sigmas_vector, machine_weights * machine_weights)
        big_sigma = tf.reduce_sum(tf.multiply(rule_sigmas, rule_w * rule_w), axis=2) + tf.multiply(
                machine_sigmas_vector,
                machine_weights * machine_weights) + 1e-10

        # Normalize the weights of features in each pair.
        _e = tf.reduce_sum(tf.multiply(feature_rule_matrix, rule_w), axis=2)
        _f = machine_weights
        weight_vector = tf.reduce_sum(tf.multiply(feature_rule_matrix, rule_w),
                                          axis=2) + machine_weights + 1e-10

        big_mu = big_mu / weight_vector  # n * 10
        big_sigma = big_sigma / (tf.square(weight_vector))  # n * 10

        # Truncated normal distribution.

        machine_label = tf.cast(machine_label, tf.double)
        machine_one = tf.cast(machine_one, tf.double)
        # -- Option 1: Use match value: 1, unmatch value: 0. --
        # prob = Fr_alpha * (tf.ones_like(machine_label) - machine_label) + (
        #        tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_label

        Fr_alpha = my_truncated_normal_ppf(alpha, a, b, big_mu, tf.sqrt(big_sigma))
        Fr_alpha_bar = my_truncated_normal_ppf(1 - alpha, a, b, big_mu, tf.sqrt(big_sigma))

        prob_mul = Fr_alpha * (tf.ones_like(machine_one) - machine_one) + (
                tf.ones_like(Fr_alpha_bar) - Fr_alpha_bar) * machine_one
        prob = tf.reduce_sum(prob_mul * machine_one, axis=1)

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
        data_len = _rule_mu.shape[0]
        # bs = np.maximum(int(data_len * 0.05), 2)
        bs = 32
        batch_num = data_len // bs + (1 if data_len % bs else 0)

        print("The test data batch size={}, batch number={}".format(bs, batch_num))

        pro_all = np.array([0.])
        pro_mul_all = np.array([[0.] * class_num])

        # mini_batch
        for i in range(batch_num):
            machine_results_batch = _machine_results[bs * i:bs * i + bs]
            rule_mu_batch = _rule_mu[bs * i:bs * i + bs]
            machine_mu_batch = _machine_mu[bs * i:bs * i + bs]
            activate_features_rule_batch = _feature_activation_rule_matrix[bs * i:bs * i + bs]
            activate_features_machine_batch = _feature_activation_machine_matrix[bs * i:bs * i + bs]
            machine_one_batch = _machine_one[bs * i: bs * i + bs]

            return_values = sess.run(
                [prob, prob_mul, rule_mu, rule_sigmas, rule_w, machine_mu,
                 machine_sigmas, machine_weights, big_mu, big_sigma, Fr_alpha,
                 Fr_alpha_bar, _a, _b, _c, _d,
                 _e, _f],
                feed_dict={machine_label: machine_results_batch,
                           rule_mu: rule_mu_batch,
                           machine_mu: machine_mu_batch,
                           feature_rule_matrix: activate_features_rule_batch,
                           feature_machine_matrix: activate_features_machine_batch,
                           rule_learn_w: _rule_learn_w,
                           machine_one: machine_one_batch,
                           alpha: _alpha,
                           rule_variance: _rule_variance,
                           machine_variance: _machine_variance,
                           match_value: _match_value,
                           unmatch_value: _unmatch_value,
                           weight_func_a: func_parameters[0],
                           weight_func_b: func_parameters[1],
                           weight_func_c: func_parameters[2]})
            # pro_all.
            np.save('pro_mul', return_values[1])
            np.save('rule_mu', return_values[2])
            np.save('rule_sigma', return_values[3])
            np.save('rule_w', return_values[4])
            np.save('mac_mu', return_values[5])
            np.save('mac_sigma', return_values[6])
            np.save('mac_w', return_values[7])
            np.save('big_mu', return_values[8])
            np.save('big_sigma', return_values[9])
            np.save('fr', return_values[10])
            np.save('fr_bar', return_values[11])
            np.save('rule_all_mu', return_values[12])
            np.save('mac_all_mu', return_values[13])
            np.save('rule_all_sigma', return_values[14])
            np.save('mac_all_sigma', return_values[15])
            np.save('rule_all_w', return_values[16])
            np.save('mac_all_w', return_values[17])

            pro_all = np.concatenate([pro_all, return_values[0]])
            pro_mul_all = np.concatenate([pro_mul_all, return_values[1]])

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
