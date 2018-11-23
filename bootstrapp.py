""" Implements Bootstrapped Ensembles.
"""

from copy import deepcopy

import torch


class BootstrappedEstimator:
    """ Implements a bootstraped ensemble of estimators.
    """

    def __init__(self, proto_model, B=20, vote=True):
        self.__ensemble = [deepcopy(proto_model) for _ in range(B)]
        self.__bno = B
        self.__vote = vote

        for model in self.__ensemble:
            model.weight.data.normal_(0, 0.1)
            # model.reset_parameters()

    def forward(self, x):
        """ Forward pass through the ensemble. Returns the mean value.
            Supports batch operations.
        """
        with torch.no_grad():
            ys = [model(x).unsqueeze(0) for model in self.__ensemble]
            ys = torch.cat(ys, 0)
        if self.__vote:
            return self.__agreed_q_vals(ys)
        return ys.mean(0)

    def get_uncertainty(self, states):
        """ Forward through each model and add a dimension representing the
            model in the ensemble. Then return the uncertainty of the ensemble.

            Supports batch operations.
        """
        with torch.no_grad():
            ys = [model(states).unsqueeze(0) for model in self.__ensemble]
            ys = torch.cat(ys, 0)
        return ys.std(0)

    def __agreed_q_vals(self, ys):
        bno, state_no = self.__bno, ys.shape[1]
        max_vals, max_idxs = ys.max(2)
        min_vals, _ = ys.min(2)

        # count the votes
        vote_cnt = max_idxs.sum(0).float()

        # the agreed wining action for each state
        winning_acts = vote_cnt > torch.zeros_like(vote_cnt).fill_(bno / 2)

        # mask according to the agreed wining action
        mask = torch.where(winning_acts.byte(), max_idxs, 1 - max_idxs).byte()

        qvals = torch.zeros(state_no, 2)
        for i, argmax in enumerate(winning_acts):
            max_val = (max_vals[:, i].masked_select(mask[:, i])).mean()
            min_val = (min_vals[:, i].masked_select(mask[:, i])).mean()

            qvals[i][argmax.item()] = max_val
            qvals[i][1 - argmax.item()] = min_val
        return qvals

    def __call__(self, transition):
        return self.forward(transition)

    def __iter__(self):
        return iter(self.__ensemble)

    def __len__(self):
        return len(self.__ensemble)

    def __repr__(self):
        return f"BootstrappedEstimator(N={len(self)})"
