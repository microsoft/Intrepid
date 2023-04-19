import random


""" Basic functionality for sampling from discrete probability distributions """


def sample_action_from_prob(prob):
    """ Pick an action sampled from the probability distribution """

    num_actions = len(prob)
    if num_actions == 0:
        raise AssertionError("There must be atleast one action.")

    v = random.random()

    for i in range(0, num_actions):
        v = v - prob[i]
        if v <= 0:
            return i

    return num_actions - 1


def sample_uniform_from_prob(num_actions):
    return random.randint(0, num_actions - 1)


def get_argmax_action(model_out_val):
    """ Returns argmax_a model_out_val(a) with random tie breaking. """

    num_actions = len(model_out_val)

    if num_actions == 0:
        raise AssertionError("There must be atleast one action.")

    ix_max = [0]
    for i in range(1, num_actions):
        if model_out_val[i] > model_out_val[ix_max[0]]:
            ix_max[:] = [i]
        elif model_out_val[i] == model_out_val[ix_max[0]]:
            ix_max.append(i)

    return ix_max[random.randint(0, len(ix_max) - 1)]
