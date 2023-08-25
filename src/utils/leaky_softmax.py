import torch


def leaky_softmax(matrix):
    """Given a matrix of size batch x num_factors we output another matrix using leaky softmax based on
    Dynamic Routing Between Capsules, Sabour et al., 2017
    """

    vector_norms = torch.norm(matrix, dim=1).view(-1)  # Batch
    sq_vector_norms = vector_norms * vector_norms  # Batch
    ratio = sq_vector_norms / (1.0 + sq_vector_norms)  # Batch

    unit_vector = matrix / vector_norms.view(-1, 1)  # Batch x num_factors
    output = unit_vector * ratio.view(-1, 1)

    return output
