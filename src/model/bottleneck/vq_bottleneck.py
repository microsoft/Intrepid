
class VQBottleneckWrapper:

    def __init__(self):
        pass

    @staticmethod
    def get_bottleneck(model_name, constants, heads=None, codebook_size=None):

        if model_name == "vq":

            from vector_quantize_pytorch import VectorQuantize

            return VectorQuantize(

                dim=constants["vq_dim"],
                codebook_size=constants["vq_codebook_size"] if codebook_size is None else codebook_size,

                # the exponential moving average decay, lower means the dictionary will change faster
                decay=constants["vq_decay"],

                # 1.,  # the weight on the commitment loss
                commitment_weight=constants["vq_commitment_weight"],

                # in paper, they recommended a value of 10
                orthogonal_reg_weight=constants["vq_orthogonal_reg_weight"],

                # this would randomly sample from the codebook for the orthogonal regularization loss,
                # for limiting memory usage
                orthogonal_reg_max_codes=constants["vq_orthogonal_reg_max_codes"],

                # set this to True if you have a very large codebook, and would only like to enforce the
                # loss on the activated codes per batch
                orthogonal_reg_active_codes_only=constants["vq_orthogonal_reg_active_codes_only"],

                # number of heads to vector quantize, codebook shared across all heads
                heads=constants["vq_heads"] if heads is None else heads,

                # whether to have a separate codebook per head. False would mean 1 shared codebook
                separate_codebook_per_head=constants["vq_separate_codebook_per_head"],

                codebook_dim=constants["vq_codebook_dim"],
                sample_codebook_temp=constants["vq_sample_codebook_temp"],
                kmeans_init=constants["vq_kmeans_init"],  # set to True

                # number of kmeans iterations to calculate the centroids for the codebook on init
                kmeans_iters=constants["vq_kmeans_iters"]
            )

        else:
            raise AssertionError("Unhandled model name %r" % model_name)

    @staticmethod
    def vq_helper(vq_model, encoding):
        encoding = encoding.unsqueeze(0)
        encoding, indices, vq_loss = vq_model(encoding)   # https://github.com/lucidrains/vector-quantize-pytorch
        vq_loss = vq_loss.sum()
        encoding = encoding.squeeze(0)
        return encoding, indices, vq_loss
