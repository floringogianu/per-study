""" Compares different experience replay sampling methods. """
from functools import partial
from collections import OrderedDict

import torch
from torch.nn import functional as F
import pandas as pd

import gym_fast_envs  # pylint: disable=unused-import
from liftoff.config import read_config, config_to_string

from utils import create_paths, configure_experiment, get_ground_truth


def learn(estimator, transition, loss_fn, optimizer, gamma):
    """ Computes the TD Error and optimises the model.

    Args:
        model (torch.nn.Module): An ensemble model.
        loss_fn (function): The loss functions.
        optimizer (torch.optim.Optim): An optimizer.
        transition (list): A s, a, r_, s_, d_, meta tuple.
        gamma (float): The discount factor.

    Returns:
        torch.tensor: The TD Error.
    """
    state, action, reward, state_, mask, _ = transition

    with torch.no_grad():
        q_targets = estimator(state_)

    qsa = estimator(state).gather(1, action)

    qsa_target = torch.zeros_like(qsa)
    qsa_target[mask] = q_targets.max(1, keepdim=True)[0][mask]

    qsa_target = (qsa_target * gamma) + reward
    loss = loss_fn(qsa, qsa_target)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return loss


def learn_ensemble(ensemble, transition, learner, mem):
    """A wrapper over the `learn` function for optimizing an ensemble of
    estimators (`model`).
    """

    if mem.shuffle:
        transition[5]["boot_mask"] = mask = mem.sample_mask()
    else:
        mask = transition[5]["boot_mask"]

    mids = [mid for mid, m in enumerate(mask) if m != 0]
    for mid in mids:
        # optim.step() is called on all the parameters of the
        # ensemble when we learn a single component of the
        # ensemble. therefore we delete the gradients of all the
        # other estimators to avoid unnecessary operations in
        # optim.step().
        for i, estimator in enumerate(ensemble):
            if i != mid:
                estimator.weight.grad = None

        ensemble_ = partial(ensemble, mid=mid)
        learner(ensemble_, transition)

    return ensemble.var(transition[0], transition[1][0].item())


def train(mem, model, learner, tester, update_cb=None):
    """ The training sequence.
    """
    has_converged = False
    step_cnt = 0

    while not has_converged:
        batch = mem.sample()

        # we are not doing mini-batch learning
        for transition in batch:

            priority = learner(model, transition)

            # update priorities
            if update_cb:
                update_cb(mem, transition, priority)

            step_cnt += 1

            # check for convergence
            has_converged = tester(step_cnt)

            if has_converged:
                break


class ConvergenceTester:
    """ Checks for convergence on one or two metrics.
    """
    def __init__(self, model, mdp_size, metrics, threshold=0.001):
        self.__model = model
        self.__threshold = threshold
        gamma = 1 - 1 / mdp_size
        test_states, optim_q_vals = get_ground_truth(mdp_size, gamma)
        self.__test_states, self.__optim_q_vals = test_states, optim_q_vals
        self.__converged = OrderedDict(
            {metric: {"converged": False, "step_cnt": 0} for metric in metrics}
        )

    def get_results(self):
        print(self.__converged)
        return [metric["step_cnt"] for metric in self.__converged.values()]

    def __call__(self, step_cnt):
        return self.__has_converged(step_cnt)

    def __has_converged(self, step_cnt):
        losses = []
        with torch.no_grad():
            y = self.__model(self.__test_states)
            if isinstance(y, tuple):
                for yi in y:
                    losses.append(F.mse_loss(yi, self.__optim_q_vals).item())
            else:
                losses.append(F.mse_loss(y, self.__optim_q_vals).item())

        for i, (k, metric) in enumerate(self.__converged.items()):
            if not metric["converged"]:
                self.__converged[k]["converged"] = losses[i] < self.__threshold
                self.__converged[k]["step_cnt"] = step_cnt

        return all([m["converged"] for m in self.__converged.values()])


def run(opt):
    """ Experiment trial. """
    lr = 0.25
    learners = [learn, learn_ensemble]

    mem, cb, model, learner, tag = configure_experiment(opt, lr, learners)
    n, mem_size = opt.mdp_size, len(mem)

    # configure tester
    metrics = ["optim_steps"]
    try:
        if opt.experience_replay.bayesian:
            metrics = ["optim_steps", "vote_optim_steps"]
    except AttributeError:
        pass
    tester = ConvergenceTester(model, n, metrics)

    # run training
    train(mem, model, learner, tester, update_cb=cb)

    # do reporting
    optim_steps = tester.get_results()
    columns = ["N", "mem_size", *metrics, "trial", "sampling_type"]
    data = [[n, mem_size, *optim_steps, opt.run_id, tag]]
    result = pd.DataFrame(data, columns=columns)

    # save panda
    result.to_msgpack(f"./{opt.out_dir}/results.msgpack")

    # log results
    print(config_to_string(opt), flush=True)
    print(f"N={n}, trial={opt.run_id} results -------", flush=True)
    print(result, flush=True)


def main():
    """ Entry point. """
    opt = read_config()
    opt = create_paths(opt)
    run(opt)


if __name__ == "__main__":
    main()
