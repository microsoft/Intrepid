import torch
import math
import os
import numpy as np
import scipy.misc

from utils.cuda import cuda_var


def _cross_entropy_binary(p, q, eps=0.00001):
    """ Compute cross entropy of two probability distributions p and q of size 2 """
    return - p[0] * math.log(q[0] + eps) - p[1] * math.log(q[1] + eps)


def l1_penalty(vector, sigmas):
    """ Compute L1 penalty on Vector using hyperparameter sigma.
    :param vector of size batch-size x dimension
    :param sigmas of size batch-size """
    return (torch.abs(vector).sum(1) / sigmas).mean() - torch.log(2 * sigmas).mean()


def save_correlation_figure_(num_homing_policies, model, test_batches, exp_name):

    correlation_stats = {}

    for batch in test_batches:

        prev_observations = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_curr_obs())).view(1, -1)
                                                for point in batch], dim=0)).float()
        actions = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_action())).view(1, -1)
                                      for point in batch], dim=0)).long()
        observations = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_next_obs())).view(1, -1)
                                           for point in batch], dim=0)).float()

        # Compute loss
        _, info_dict = model.gen_prob(prev_observations, actions, observations)  # batch x 2
        assigned_states = info_dict["assigned_states"]

        for i, point in enumerate(batch):
            assigned_state = int(assigned_states[i])
            if point.get_next_state() in correlation_stats:
                correlation_stats[point.get_next_state()][assigned_state] += 1.0
            else:
                vec = np.zeros(num_homing_policies, dtype=np.float32)
                vec[assigned_state] = 1.0
                correlation_stats[point.get_next_state()] = vec

    num_states = 0
    image = []
    for key in sorted(correlation_stats):
        vec = correlation_stats[key]
        vec = vec / max(1.0, vec.sum())
        image.append(vec)
        num_states += 1
    image = np.vstack(image)
    image = scipy.misc.imresize(image, (num_states * 100, num_homing_policies * 100))

    filelist = os.listdir('./%s' % exp_name)
    num_images = len(filelist)
    scipy.misc.imsave("./%s/image_%d.png" % (exp_name, num_images + 1), image)


def log_model_performance(num_homing_policies, model, test_batches, best_test_loss, logger):

    predictions_weight = {}
    predictions_values_gold = {}
    predictions_values_inferred = {}
    num_counts = 0

    mode_correlation_stats, prob_correlation_stats = {}, {}

    states_mapping = {}

    for batch in test_batches:

        prev_observations = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_curr_obs())).view(1, -1)
                                                for point in batch], dim=0)).float()
        actions = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_action())).view(1, -1)
                                      for point in batch], dim=0)).long()
        observations = cuda_var(torch.cat([torch.from_numpy(np.array(point.get_next_obs())).view(1, -1)
                                           for point in batch], dim=0)).float()

        # Compute loss
        batch_probs, info_dict = model.gen_prob(prev_observations=prev_observations,
                                                actions=actions,
                                                observations=observations,
                                                discretized=True)  # batch x 2
        inferred_labels = torch.distributions.Bernoulli(batch_probs[:, 1]).sample()

        if "assigned_states" not in info_dict or "prob" not in info_dict:
            logger.log("Info dict doesn't contain assigned_states or prob")
            return

        assigned_states = info_dict["assigned_states"]
        assigned_prob = info_dict["prob"]
        num_abstract_states = assigned_prob.size(1)

        for i, point in enumerate(batch):

            assigned_state = int(assigned_states[i])
            next_state = point.get_next_state()

            if next_state not in states_mapping:
                states_mapping[next_state] = set()
            states_mapping[next_state].add(assigned_state)

            if next_state in mode_correlation_stats:
                mode_correlation_stats[next_state][assigned_state] += 1.0
                for j in range(0, num_abstract_states):
                    prob_correlation_stats[next_state][j] += float(assigned_prob[i][j])
            else:
                mode_vec = np.zeros(num_homing_policies, dtype=np.float32)
                mode_vec[assigned_state] = 1.0
                mode_correlation_stats[next_state] = mode_vec

                prob_correlation_stats[next_state] = np.zeros(num_homing_policies, dtype=np.float32)
                for j in range(0, num_abstract_states):
                    prob_correlation_stats[next_state][j] += float(assigned_prob[i][j])

            num_counts = num_counts + 1
            key = "%r -> %r -> %r" % (point.get_curr_state(), point.get_action(), point.get_next_state())
            if key in predictions_weight:
                predictions_weight[key] = predictions_weight[key] + 1
            else:
                predictions_weight[key] = 1

            if key in predictions_values_gold:
                probs = predictions_values_gold[key]
                probs[point.is_valid()] += 1
                predictions_values_gold[key] = probs
            else:
                probs = [0, 0]
                probs[point.is_valid()] += 1
                predictions_values_gold[key] = probs

            if key in predictions_values_inferred:
                probs = predictions_values_inferred[key]
                inf_ix = int(inferred_labels[i])
                probs[inf_ix] += 1
                predictions_values_inferred[key] = probs
            else:
                probs = [0, 0]
                inf_ix = int(inferred_labels[i])
                probs[inf_ix] += 1
                predictions_values_inferred[key] = probs

    optimal_loss, predicted_loss = 0.0, 0.0

    for state, vec in sorted(mode_correlation_stats.items()):
        norm_vec = (vec/max(1.0, vec.sum())).tolist()
        norm_vec = [(i, round(v, 2)) for i, v in enumerate(norm_vec) if v > 0.0]
        logger.log("Real State: %r mode correlations %r" % (state, norm_vec))

    for state, vec in sorted(prob_correlation_stats.items()):
        norm_vec = (vec / max(1.0, vec.sum())).tolist()
        norm_vec = [(i, round(v, 2)) for i, v in enumerate(norm_vec) if v > 0.0]
        logger.log("Real State: %r prob correlations %r" % (state, norm_vec))

    ordered_states = list(mode_correlation_stats.keys())
    num_states = len(ordered_states)
    for i in range(0, num_homing_policies):
        vec = np.zeros(num_states, dtype=np.float32)
        for j, state in enumerate(ordered_states):
            vec[j] += mode_correlation_stats[state][i]  # number of times i matched to state
        vec = vec/max(1.0, vec.sum())
        log_str = ""
        for j, state in enumerate(ordered_states):
            if vec[j] > 0:
                log_str += "%r: %r,  " % (state, round(vec[j], 2))
        logger.log("Abstract State: %r correlations %r" % (i, log_str))

    for key in sorted(predictions_weight, key=lambda x: x[::-1]):   # Sort based on ordering of reverse of key strings
        prior = predictions_weight[key] / float(max(1, num_counts))

        gold = predictions_values_gold[key]
        gold_z = float(max(1, gold[0] + gold[1]))
        gold = [round(gold[0] / gold_z, 2), round(gold[1] / gold_z, 2)]
        entropy_loss_gold = _cross_entropy_binary(gold, gold)
        optimal_loss_ = prior * entropy_loss_gold
        optimal_loss += optimal_loss_

        inferred = predictions_values_inferred[key]
        inferred_z = float(max(1, inferred[0] + inferred[1]))
        inferred = [round(inferred[0] / inferred_z, 2), round(inferred[1] / inferred_z, 2)]
        entropy_loss_predicted = _cross_entropy_binary(gold, inferred)
        predicted_loss_ = prior * entropy_loss_predicted
        predicted_loss += predicted_loss_

        logger.log("%r: Prior: %r, Gold: %r, Inferred: %r, Delta: %r" %
                   (key, prior, gold, inferred, round(abs(predicted_loss_ - optimal_loss_), 2)))

    logger.log("Optimal Loss %r vs Model Loss %r (Model empirical loss %r)" %
               (round(optimal_loss, 2), round(predicted_loss, 2), round(best_test_loss, 2)))


def log_dataset_stats(dataset, logger):

    dataset_size = len(dataset)

    # Find number of positive and negative samples
    pos, neg = 0.0, 0.0

    # Find counts for number of states reached
    state_counts = dict()

    for dp in dataset:
        if dp.get_next_state() in state_counts:
            state_counts[dp.get_next_state()] += 1.0
        else:
            state_counts[dp.get_next_state()] = 1.0

        if dp.is_valid() == 1.0:
            pos += 1.0
        else:
            neg += 1.0

    pos = (pos * 100.0) / float(max(1, dataset_size))
    neg = (neg * 100.0) / float(max(1, dataset_size))

    for state in state_counts:
        state_counts[state] = round((state_counts[state] * 100.0)/float(max(1, dataset_size)), 2)

    logger.log("Dataset size %r {pos: %r, neg: %r}" % (dataset_size, round(pos, 2), round(neg, 2)))
    logger.log("State Visitation Stats %s" % state_counts)
