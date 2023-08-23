import torch
import torch.nn as nn
import torch.nn.functional as F



class IndependenceTestModel(nn.Module):
    def __init__(self, config, model_input_dim, hidden_dim):
        super(IndependenceTestModel, self).__init__()

        self.config = config

        if config["feature_type"] == "feature":
            # Model head
            self.classifier = nn.Sequential(
                nn.Linear(model_input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 2),
            )

        else:
            raise AssertionError("Unhandled feature type")

        if torch.cuda.is_available():
            self.cuda()

    def gen_logits_(self, model_input, type="logsoftmax"):
        """
        :param model_input:    Pytorch float tensor of size batch x dim
        :return:
        """

        if self.config["feature_type"] == "image":
            raise AssertionError()

        logits = self.classifier(model_input)

        if type == "logsoftmax":
            result = F.log_softmax(logits, dim=1)
        elif type == "softmax":
            result = F.softmax(logits, dim=1)
        else:
            raise AssertionError("Unhandled type ", type)

        return result

    def gen_log_prob(self, model_input):
        return self.gen_logits_(model_input, type="logsoftmax")

    def gen_prob(self, model_input):
        return self.gen_logits_(model_input, type="softmax")
