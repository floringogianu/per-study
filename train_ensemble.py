""" Look at the uncertainty estimates.
"""
import os
from datetime import datetime
from functools import partial

import torch
from torch import nn
from torch.optim import SGD
from torch.nn import functional as F
import pandas as pd
import gym
import fire
from termcolor import colored as clr

from utils import get_all_transitions, get_ground_truth
from experience_replay import NaiveExperienceReplay, _collate
from bootstrapp import BootstrappedEstimator
from comparison import learn


def test(model, test_states, ground_truth_values):
    """ Checks for convergence.
    """
    with torch.no_grad():
        q = model(test_states)
        mse_loss = F.mse_loss(q, ground_truth_values).item()
    return mse_loss


def log(step, logger, model, test_states, trial):
    """ Log some stats and save them in a panda file. """
    std = model.get_uncertainty(test_states)
    for obs in range(std.shape[0]):
        for act in range(std.shape[1]):
            logger.update(step, std[obs, act].item(), f"s{obs}a{act}", trial)
    logger.save()


def train(mem, model, optimizer, gamma, test_cb, log_cb, dist, verbose=True):
    """ The training sequence.
    """

    has_converged = False
    step_cnt = 0

    while not has_converged:
        batch = mem.sample()

        # we are not doing mini-batch learning
        for transition in batch:

            # compute the td error and optimize the model
            mask = dist.sample((len(transition[5]["boot_mask"]),))
            for mdl, opt, cond in zip(model, optimizer, mask):
                if cond:
                    learn(mdl, F.mse_loss, opt, transition, gamma)

            # test for convergence
            test_loss = test_cb()
            has_converged = test_loss < 0.001
            step_cnt += 1

            # log the model uncertainty
            log_cb(step_cnt)

            # do some logging
            if step_cnt % 100 == 0 and verbose:
                print(f"{step_cnt:3d}  test_loss={test_loss:2.4f}.")

            if has_converged:
                break

    if verbose:
        print(f"Found ground truth in {step_cnt:6d} steps.")

    return step_cnt


class Logger:
    """ Dumb logger that saves the results in a panda.
    """

    def __init__(self, columns, path):
        self.__cols = columns
        self.__df = pd.DataFrame(columns=columns)
        self.__path = path

    def update(self, step, uncertainty, qsa, trial):
        """ Add rows in each of the columns received as arguments. """
        data = [[step, uncertainty, qsa, trial]]
        df = pd.DataFrame(data, columns=self.__cols)
        self.__df = self.__df.append(df, ignore_index=True)

    def save(self):
        """ Saves the dataframe. """
        self.__df.to_msgpack(f"{self.__path}.msgpack")


def configure_experiment(n, boot_p, boot_no, vote, lr):
    """ Constructs estimator, optimizer and memory. """

    # sample the env
    env = gym.make(f"BlindCliffWalk-N{n}-v0")
    transitions = get_all_transitions(env, n)
    mem = NaiveExperienceReplay(
        capacity=len(transitions),
        batch_size=1,
        collate=_collate,
        full_transition=True,
    )

    bern = torch.distributions.Bernoulli(boot_p)
    for transition in transitions:
        mem.push((*transition, {"boot_mask": bern.sample((boot_no,))}))

    # configure estimator and optimizer
    estimator = nn.Linear(n + 1, 2, bias=False)  # already added in the state
    estimator.weight.data.normal_(0, 0.1)
    optimizer = SGD(estimator.parameters(), lr=lr)
    optimizer.zero_grad()

    estimator = BootstrappedEstimator(estimator, B=boot_no, vote=vote)
    optimizer = [SGD(model.parameters(), lr=lr) for model in estimator]
    for optim in optimizer:
        optim.zero_grad()

    return mem, estimator, optimizer, bern


def run_experiment(trial, n=2, boot_p=0.5, boot_no=10, vote=True, path=""):
    """ Configures an experiment and calls `train`. """
    lr = 0.25
    gamma = 1 - 1 / n

    # get test and train data
    test_states, ground_truth_values = get_ground_truth(n, gamma)
    mem, estimator, optimizer, mask_dist = configure_experiment(
        n, boot_p, boot_no, vote, lr
    )

    # configure the test callback
    test_cb = partial(
        test,
        model=estimator,
        test_states=test_states,
        ground_truth_values=ground_truth_values,
    )

    # do reporting
    log_cb = partial(
        log,
        logger=Logger(["step", "uncertainty", "qsa", "trial"], path),
        model=estimator,
        test_states=test_states,
        trial=trial,
    )

    # train
    msg = f"Training {estimator} on {mem} with p={boot_p} and vote={vote}."
    print(clr(msg, "green"))

    train(mem, estimator, optimizer, gamma, test_cb, log_cb, mask_dist)


def main(n=2, boot_p=0.5, boot_no=10, vote=True, trials=1):
    """ Entry point.
    """
    # configure experiment outputs
    boot_p_str = str(boot_p).replace(".", "")
    exp_name = f"mdp{n}_n{boot_no}_p{boot_p_str}_vote{int(vote)}"
    exp_date = "{:%Y%b%d_%H%M%S}".format(datetime.now())
    print(exp_name)
    exp_dir = f"./results/confidence_dynamics/{exp_date}_{exp_name}/"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    for trial in range(trials):
        exp_path = f"{exp_dir}trial_{trial}"
        run_experiment(trial, n, boot_p, boot_no, vote, exp_path)


if __name__ == "__main__":
    fire.Fire(main)
