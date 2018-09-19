""" Experience Replay implementations. """
import heapq

import torch
import numpy as np

from wintermute.data_structures import NaiveExperienceReplay
from data_structures import PriorityQueue


def get_experience_replay(capacity, batch_size=1, strategy='uniform'):
    """ Factory for various Experience Replay implementations. """
    if strategy == 'uniform':
        mem = NaiveExperienceReplay(capacity, batch_size=batch_size,
                                    collate=_collate, full_transition=True)
    elif strategy == 'greedy-pq':
        mem = GreedyPQSampling(capacity, batch_size=batch_size,
                               collate=_collate)
    elif strategy == 'greedy-hpq':
        mem = GreedyHeapqSampling(capacity, batch_size=batch_size,
                                  collate=_collate)
    return mem


def _collate(samples):
    batch = list(zip(*samples))
    states = torch.cat([torch.from_numpy(s) for s in batch[0]], 0)
    actions = torch.LongTensor(batch[1]).unsqueeze(1)
    rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
    next_states = torch.cat([torch.from_numpy(s) for s in batch[3]], 0)
    mask = 1 - torch.ByteTensor(batch[4]).unsqueeze(1)
    return [states.float(), actions, rewards, next_states.float(), mask]


class GreedyHeapqSampling:
    """ Implements the greedy TD-error sampling technique in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) used in the
        BlindCliffWalk experiments.
    """
    def __init__(self, capacity, batch_size=1, collate=None):
        self.__memory = []
        self.__batch_size = batch_size
        self.__capacity = capacity

        self.__collate = collate or _collate
        self.__position = 0


    def push(self, transition, priority=None):
        """ Adds a transition in the priority queue.

            The first element in the item is used in the comparison operations
            in the heapq. Having no initial td errors we populate them with a
            very small value so that we make sure they get sampled. We will
            gradually update the priorities.

            Second element is used by `heapq` to break eventual ties in
            priority and for the first optimization sweep will act as the
            actual prioriy.
        """
        priority = (self.__position + 1000) or priority
        heapq.heappush(self.__memory, (-priority, transition))
        self.__position = (self.__position + 1) % self.__capacity


    def sample(self):
        """ Returns the transitions with the highest priority. """
        samples = [heapq.heappop(self.__memory)[1] for _ in
                   range(self.__batch_size)]
        return self.__collate(samples)


    def push_updated(self, td_err, transition):
        """ Adds a transition prioritized by the TD-Error. """
        td_err += np.random.rand() * 1e-7
        heapq.heappush(self.__memory, (-td_err, transition))


    def __len__(self):
        return len(self.__memory)



class GreedyPQSampling:
    """ Implements the greedy TD-error sampling technique in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) used in the
        BlindCliffWalk experiments.
    """
    def __init__(self, capacity, batch_size=1, collate=None):
        self.__pq = PriorityQueue()
        self.__batch_size = batch_size
        self.__capacity = capacity

        self.__collate = collate or _collate
        self.__position = 0


    def push(self, transition, priority=None):
        priority = (self.__position + 1000) or priority
        self.__pq.push((-priority, transition))
        self.__position = (self.__position + 1) % self.__capacity


    def sample(self):
        samples = [self.__pq.pop()[1] for _ in range(self.__batch_size)]
        return self.__collate(samples)


    def push_updated(self, td_err, transition):
        self.__pq.push((-td_err, transition))


    def __len__(self):
        return len(self.__pq)


def torch2numpy(states, actions, rewards, states_, mask):
    """ Convert a torch batch to a list of game transitions.

        Used for reinserting in the experience replay while updating the
        priorities.
    """
    batch_size = states.shape[0]

    transitions = [[states[i].unsqueeze(0).numpy(),
                    actions[i].item(),
                    rewards[i].item(),
                    states_[i].unsqueeze(0).numpy(),
                    1 - mask[i].item()] for i in range(batch_size)]
    return transitions
