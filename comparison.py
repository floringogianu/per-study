""" Compares different experience replay sampling methods. """
import torch
from torch import nn
from torch.optim import SGD
from torch.nn import functional as F
import numpy as np
import pandas as pd
import gym

import gym_fast_envs
from liftoff.config import read_config, config_to_string

from utils import get_all_transitions, get_ground_truth, create_paths
from experience_replay import get_experience_replay
from bootstrapp import BootstrappedEstimator


LOSS_F = {"mse": F.mse_loss, "huber": F.smooth_l1_loss}


def learn(estimator, loss_fn, optimizer, transition, gamma):
    """ Compute the td error and optimize the model.
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
    estimator.zero_grad()
    return loss


def train(n, mem, model, loss_fn, optimizer, update_cb=None, verbose=True):
    """ The training sequence.
    """

    gamma = 1 - 1 / n
    test_states, ground_truth_values = get_ground_truth(n, gamma)

    has_converged = False
    step_cnt = 0

    while not has_converged:
        batch = mem.sample()

        # we are not doing mini-batch learning
        for transition in batch:

            # compute the td error and optimize the model
            loss = learn(model, loss_fn, optimizer, transition, gamma)

            # update priorities
            if update_cb:
                update_cb(mem, transition, loss)

            # check for convergence
            with torch.no_grad():
                q = model(test_states)
                mse_loss = F.mse_loss(q, ground_truth_values).item()

            has_converged = mse_loss < 0.001
            step_cnt += 1

            # do some logging
            if step_cnt % 100_000 == 0 and verbose:
                print(f"{step_cnt:3d}  mse_loss={mse_loss:2.4f}.")

            if has_converged:
                break

    if verbose:
        print(f"Found ground truth in {step_cnt:6d} steps.")

    return step_cnt


def configure_experiment(opt, lr=0.25):
    """ Sets up the objects required for running the experiment.
    """
    n = opt.mdp_size
    sampling_type = opt.experience_replay.sampling

    # sample the env
    env = gym.make(f"BlindCliffWalk-N{n}-v0")
    transitions = get_all_transitions(env, n)

    # construct and populate Experience Replay
    mem, cb = get_experience_replay(
        len(transitions), **opt.experience_replay.__dict__
    )
    for transition in transitions:
        mem.push(transition)

    # loss function
    if "loss" in opt.__dict__:
        loss_fn = LOSS_F[opt.loss]
    else:
        loss_fn = LOSS_F["huber"]

    # configure estimator and optimizer
    estimator = nn.Linear(n + 1, 2, bias=False)  # already added in the state
    optimizer = SGD(estimator.parameters(), lr=lr)
    optimizer.zero_grad()

    # get sampling type tag, we use it for reporting
    tag = get_sampling_variant(**opt.experience_replay.__dict__)

    # add loss name in tag if it exists
    tag = f"{tag}_{opt.loss}" if "loss" in opt.__dict__ else tag

    print(f">>  Experience Replay: {mem}")

    return mem, cb, estimator, loss_fn, optimizer, tag


def get_sampling_variant(sampling="uniform", **kwargs):
    """ Creates a tag of the form sampling + hyperparams if hyperparams exist
    """
    hp_names = ("alpha", "beta", "batch_size")
    hyperparams = {h: kwargs[h] for h in hp_names if h in kwargs}

    if not kwargs:
        return sampling
    for k, v in hyperparams.items():
        sampling += f"_{k}:{v}"
    return sampling


def run(opt):
    """ Experiment trial. """
    mem, cb, estimator, loss_fn, optimizer, tag = configure_experiment(opt)
    n, mem_size = opt.mdp_size, len(mem)

    # initialize weights
    estimator.weight.data.normal_(0, 0.1)

    # run training
    step_cnt = train(
        n, mem, estimator, loss_fn, optimizer, update_cb=cb, verbose=True
    )

    # do reporting
    columns = ["N", "mem_size", "optim_steps", "trial", "sampling_type"]
    data = [[n, mem_size, step_cnt, opt.run_id, tag]]
    result = pd.DataFrame(data, columns=columns)

    # save panda
    result.to_msgpack(f"./{opt.out_dir}/results.msgpack")

    # log results
    print(f"N={n}, trial={opt.run_id} results -------", flush=True)
    print(result, flush=True)


def main():
    """ Entry point. """
    opt = read_config()
    opt = create_paths(opt)
    run(opt)


if __name__ == "__main__":
    main()
