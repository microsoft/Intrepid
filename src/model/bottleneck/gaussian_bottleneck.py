import torch
import torch.nn as nn


class GaussianBottleneck(nn.Module):

    def __init__(self, hidden_dim):
        super(GaussianBottleneck, self).__init__()

        self.hidden_dim = hidden_dim
        self.pre_enc = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        self.post_enc = nn.Linear(self.hidden_dim, self.hidden_dim)

        # TODO add command line argument
        self.kl_weight = 0.0001

        self.mu_prior = nn.Parameter(torch.zeros(self.hidden_dim))
        self.sigma_prior = nn.Parameter(torch.ones(self.hidden_dim))

        if torch.cuda.is_available():
            self.cuda()

    def gb_helper(self, h):

        h = self.pre_enc(h)
        mu = h[:, :self.hidden_dim]
        std = torch.exp(h[:, self.hidden_dim:]) + 1e-6
        q_z = torch.distributions.Normal(loc=mu, scale=std)

        if self.training:
            # print('std: {}, mu: {}, mu_prior: {}'.format(std.mean(), torch.abs(mu).mean(),
            #                                              torch.abs(self.mu_prior).mean()))
            # h = mu + torch.randn_like(std) * std
            h = q_z.rsample()
            # klb_loss = (mu**2 + std**2 - 2*torch.log(std)).sum(dim=1).mean() * self.kl_weight
            p_z = torch.distributions.Normal(loc=self.mu_prior, scale=self.sigma_prior)
            klb_loss = self.kl_weight * torch.distributions.kl_divergence(q_z, p_z).sum(dim=1).mean()
        else:
            h = mu
            klb_loss = 0.0

        # h = self.post_enc(h)
        return h, klb_loss
