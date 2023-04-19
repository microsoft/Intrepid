from learning.learning_utils.clustering_algorithm import *
from learning.learning_utils.generic_learner import GenericLearner


class GenericTrainClassifier(GenericLearner):
    """ Class for training a classifier. Fairly generic with minimal assumption """

    def __init__(self, exp_setup):

        GenericLearner.__init__(self, exp_setup)

    @staticmethod
    def calc_prob(model, batch):

        obs = cuda_var(torch.cat([torch.from_numpy(np.array(pt[0])).view(1, -1) for pt in batch], dim=0)).float()

        prob, info_dict = model.gen_prob(obs)           # Batch x Num Classes

        return prob, info_dict

    def calc_loss(self, model, batch, test=False):

        obs = cuda_var(torch.cat([torch.from_numpy(np.array(pt[0])).view(1, -1) for pt in batch], dim=0)).float()
        y = cuda_var(torch.LongTensor([pt[1] for pt in batch]).view(-1))

        log_prob, info_dict = model.gen_log_prob(obs)           # Batch x Num Classes

        selected_log_prob = log_prob.gather(1, y.view(-1, 1))   # Batch
        loss = - selected_log_prob.mean()

        return loss, info_dict

    def get_class_mean_prob(self, model, dataset):
        """
        :param model:   A classification model f for mapping input space X to distribution over K classes. Given input
                        x in X, the model f(j | x) denotes the probability of class j.
        :param dataset: A list of tuples where first dimension of tuple is input x in X, and second is class label,
                        rest dimensions are ignored and can be used for adding meta-information.
        :return:        A pytorch cpu matrix of size dataset_size x N where (i, j)^{th} value denote
                        1/N f(j | x_i) where N is the size of dataset and x_i is the i^{th} input in the dataset.
        """

        dataset_size = len(dataset)
        batches = [dataset[i:i + self.batch_size] for i in range(0, dataset_size, self.batch_size)]
        all_prob = []

        for batch in batches:

            prob, info_dict = self.calc_prob(model, batch)
            prob = prob.detach().data.cpu()        # batch x num_class

            all_prob.append(prob)

        all_prob = torch.cat(all_prob, dim=0)      # Dataset x num_class

        return all_prob / float(all_prob.size(0))
