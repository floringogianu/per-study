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
from experience_replay_strategies import get_experience_replay, torch2numpy


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


def configure_experiment(n, strategy='uniform', lr=0.25):
    """ Sets up the objects required for running the experiment. """
    env = gym.make(f'BlindCliffWalk-N{n}-v0')
    transitions = get_all_transitions(env, n)

    mem = get_experience_replay(len(transitions), strategy=strategy)

    for transition in transitions:
        mem.push(transition)

    estimator = nn.Linear(n+1, 2, bias=True)
    optimizer = SGD(estimator.parameters(), lr=lr)
    optimizer.zero_grad()

    return mem, estimator, optimizer


def greedy_update(mem, transitions, losses):
    """ Callback for reinserting transitions in the experience replay with the
    new priorities.
    """
    td_errors = np.abs(losses).squeeze()
    td_errors = [td_errors] if td_errors.shape == () else td_errors

    for td_err, transition in zip(td_errors, transitions):
        mem.push_updated(td_err, transition)


def rank_update(mem, transitions, losses):
    """ Callback for updating priorities in the experience replay.
    """
    td_errors = np.abs(losses).squeeze()
    td_errors = [td_errors] if td_errors.shape == () else td_errors

    for td_err, transition in zip(td_errors, transitions):
        # for rank based updates the sampled transition contains the idx in the
        # replay buffer from where it was sampled
        mem.update(transition[0], td_err)


def run(args):
    """ Experiment trial. """
    n = args.mdp_size
    strategy = args.strategy

    mem, estimator, optimizer = configure_experiment(n, strategy)
    # initialize weights
    estimator.weight.data.normal_(0, 0.1)
    estimator.bias.data.normal_(0, 1)

    # run training
    cb = greedy_update if strategy in ('greedy-pq', 'greedy-hpq') else None
    cb = rank_update if strategy == 'rank' else cb
    step_cnt = train(n, mem, estimator, optimizer, update_priorities=cb,
                     verbose=True)

    # do reporting
    columns = ['N', 'mem_size', 'optim_steps', 'trial', 'sampling_type']
    data = [[n, len(mem), step_cnt, args.run_id, strategy]]
    result = pd.DataFrame(data, columns=columns)

    # log results
    print(f'N={n}, trial={args.run_id} results -------', flush=True)
    print(result, flush=True)

    # save panda
    result.to_msgpack(f'./{args.out_dir}/results.msgpack')


def main():
    """ Entry point. """
    args = read_config()
    args = create_paths(args)
    run(args)


if __name__ == "__main__":
    main()
