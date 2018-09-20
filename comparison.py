""" Compares different experience replay sampling methods. """
import torch
from torch import nn
from torch.optim import SGD
from torch.nn import functional as F
import numpy as np
import pandas as pd
import gym

import gym_fast_envs
from wintermute.policy_improvement import get_td_error
from liftoff.config import read_config, config_to_string

from utils import get_all_transitions, get_ground_truth, create_paths
from experience_replay import get_experience_replay, torch2numpy


def train(n, mem, estimator, optimizer, verbose=True, update_priorities=None):
    """ The training sequence. """
    gamma = 1-1/n

    test_states, ground_truth_values = get_ground_truth(n, gamma)

    has_converged = False
    step_cnt = 0

    while not has_converged:
        batch = mem.sample()
        try:
            states, actions, rewards, states_, mask = batch
        except ValueError:
            _, states, actions, rewards, states_, mask = batch

        with torch.no_grad():
            q_targets = estimator(states_)

        q_values = estimator(states)
        qsa = q_values.gather(1, actions)

        qsa_target = torch.zeros_like(qsa)
        qsa_target[mask] = q_targets.max(1, keepdim=True)[0][mask]

        losses = get_td_error(qsa, qsa_target, rewards, gamma, reduction='none')
        loss = losses.mean()
        loss.backward()

        optimizer.step()
        estimator.zero_grad()

        # update priorities
        if update_priorities:
            transitions = torch2numpy(batch)
            update_priorities(mem, transitions, losses.detach().numpy())

        # check for convergence
        with torch.no_grad():
            q = estimator(test_states)
            mse_loss = F.mse_loss(q, ground_truth_values).item()

        has_converged = mse_loss < 0.001
        step_cnt += states.shape[0]

        # do some logging
        if step_cnt % 100000 == 0 and verbose:
            print(f'{step_cnt:3d}  mse_loss={mse_loss:2.4f}.')

    if verbose:
        print(f'Found ground truth in {step_cnt:6d} steps.')

    return step_cnt


def configure_experiment(opt, lr=0.25):
    """ Sets up the objects required for running the experiment.
    """
    n = opt.mdp_size
    sampling_type = opt.experience_replay.sampling

    # sample the env
    env = gym.make(f'BlindCliffWalk-N{n}-v0')
    transitions = get_all_transitions(env, n)

    # construct and populate Experience Replay
    mem, cb = get_experience_replay(len(transitions),
                                    **opt.experience_replay.__dict__)
    for transition in transitions:
        mem.push(transition)

    # configure estimator and optimizer
    estimator = nn.Linear(n+1, 2, bias=True)
    optimizer = SGD(estimator.parameters(), lr=lr)
    optimizer.zero_grad()

    # get sampling type tag, we use it for reporting
    tag = get_sampling_variant(**opt.experience_replay.__dict__)

    print(f'>>  Experience Replay: {mem}')

    return mem, cb, estimator, optimizer, tag


def get_sampling_variant(sampling='uniform', **kwargs):
    """ Creates a tag of the form sampling + hyperparams if hyperparams exist
    """
    hp_names = ('alpha', 'beta')
    hyperparams = {h: kwargs[h] for h in hp_names if h in kwargs}

    if not kwargs:
        return sampling
    for k, v in hyperparams.items():
        sampling += f'_{k}:{v}'
    return sampling


def run(opt):
    """ Experiment trial. """
    n = opt.mdp_size

    mem, cb, estimator, optimizer, tag = configure_experiment(opt)

    # initialize weights
    estimator.weight.data.normal_(0, 0.1)
    estimator.bias.data.normal_(0, 1)

    # run training
    step_cnt = train(n, mem, estimator, optimizer, update_priorities=cb,
                     verbose=True)

    # do reporting
    columns = ['N', 'mem_size', 'optim_steps', 'trial', 'sampling_type']
    data = [[n, len(mem), step_cnt, opt.run_id, tag]]
    result = pd.DataFrame(data, columns=columns)

    # save panda
    result.to_msgpack(f'./{opt.out_dir}/results.msgpack')

    # log results
    print(f'N={n}, trial={opt.run_id} results -------', flush=True)
    print(result, flush=True)


def main():
    """ Entry point. """
    opt = read_config()
    opt = create_paths(opt)
    run(opt)


if __name__ == "__main__":
    main()
