""" Experience Replay implementations.

    1. GreedyHeapqSampler implements greedy sampling from the buffer using
    python's standard library `heapq`.

    2. GreedyPQSampler also implements greedy sampling but using the
    PriorityQueue in `data_structures.py`. The two implementations perform the
    same

    3. RankSampler implements the rank-based prioritization using the
    PriorityQueue in `data_structures.py`.
"""
import heapq

import torch
import numpy as np

from wintermute.data_structures import NaiveExperienceReplay
from data_structures import PriorityQueue


def get_experience_replay(capacity, batch_size=1, strategy='uniform'):
    """ Factory for various Experience Replay implementations. """

    # TODO: fix the factory, all experience replay implementations should
    # have a common constructor.

    if strategy == 'uniform':
        return NaiveExperienceReplay(capacity, batch_size=batch_size,
                                     collate=_collate, full_transition=True)
    return BUFFERS[strategy](capacity, batch_size=batch_size)


def _collate(samples):
    batch = list(zip(*samples))
    states = torch.cat([torch.from_numpy(s) for s in batch[0]], 0)
    actions = torch.LongTensor(batch[1]).unsqueeze(1)
    rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
    next_states = torch.cat([torch.from_numpy(s) for s in batch[3]], 0)
    mask = 1 - torch.ByteTensor(batch[4]).unsqueeze(1)
    return [states.float(), actions, rewards, next_states.float(), mask]


def _collate_with_index(samples):
    idxs, samples = list(zip(*samples))
    batch = list(zip(*samples))

    idxs = torch.LongTensor(idxs).unsqueeze(1)
    states = torch.cat([torch.from_numpy(s) for s in batch[0]], 0)
    actions = torch.LongTensor(batch[1]).unsqueeze(1)
    rewards = torch.FloatTensor(batch[2]).unsqueeze(1)
    next_states = torch.cat([torch.from_numpy(s) for s in batch[3]], 0)
    mask = 1 - torch.ByteTensor(batch[4]).unsqueeze(1)
    return [idxs, states.float(), actions, rewards, next_states.float(), mask]


class GreedyHeapqSampler:
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



class GreedyPQSampler:
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



class RankSampler:
    """ Implements the rank-based sampling technique in [Prioritized
        Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) used in the
        BlindCliffWalk experiments.
    """
    def __init__(self, capacity, batch_size=1, collate=None, k=None):
        self.__pq = PriorityQueue()
        self.__capacity = capacity
        self.__batch_size = batch_size

        self.__collate = collate or _collate_with_index
        self.__position = 0

        self.__alpha = 0.7
        self.__k = k if k else batch_size
        self.__partitions = []
        self.__segments = []
        self.__segment_probs = []


    def push(self, transition, priority=None):
        priority = (self.__position + 1000) or priority
        self.__pq.push((-priority, transition))
        self.__position = (self.__position + 1) % self.__capacity

        if self.__capacity == len(self):
            self.__compute_segments()
            for s, p in zip(self.__segments, self.__segment_probs):
                print(f'segment{s} with probability {p:6.3f}.')


    def sample(self):
        """ TODO: """
        segment_idxs = np.random.choice(len(self.__segments),
                                        size=self.__batch_size,
                                        p=self.__segment_probs)
        segments = [self.__segments[sid] for sid in segment_idxs]
        idxs = [np.random.randint(*segment) for segment in segments]

        # warning, atypical use of a priority queue
        samples = [(i, self.__pq._PriorityQueue__heap[i][1]) for i in idxs]

        return self.__collate(samples)


    def update(self, idx, priority):
        self.__pq.update(idx, -priority)


    def sort(self):
        for _ in range(len(self)):
            self.__pq.push(self.__pq.pop())


    def __compute_segments(self):
        N = len(self)
        self.__partitions = []
        self.__segments = []

        segment_sz = int(np.round(N / self.__k))
        for i in range(self.__k):
            a = i * segment_sz
            b = (i+1) * segment_sz if i != (self.__k-1) else N

            partition = [(1 / (idx+1)) ** self.__alpha for idx in range(a, b)]

            self.__partitions.append(np.sum(partition))
            self.__segments.append((a, b))

        self.__segment_probs = [p / sum(self.__partitions) for p
                                in self.__partitions]


    def __len__(self):
        return len(self.__pq)



def torch2numpy(batch):
    """ Convert a torch batch to a list of game transitions.

        A batch can contain sampled indices besides the transitions.
    """
    idxs = None
    try:
        states, actions, rewards, states_, mask = batch
    except ValueError:
        idxs, states, actions, rewards, states_, mask = batch

    batch_size = states.shape[0]

    if idxs is not None:
        return [[int(idxs[i].item()),
                 states[i].unsqueeze(0).numpy(),
                 actions[i].item(),
                 rewards[i].item(),
                 states_[i].unsqueeze(0).numpy(),
                 1 - mask[i].item()] for i in range(batch_size)]
    return [[states[i].unsqueeze(0).numpy(),
             actions[i].item(),
             rewards[i].item(),
             states_[i].unsqueeze(0).numpy(),
             1 - mask[i].item()] for i in range(batch_size)]


BUFFERS = {'uniform': NaiveExperienceReplay,
           'greedy-pq': GreedyPQSampler,
           'greedy-hpq': GreedyHeapqSampler,
           'rank': RankSampler}


def main():
    import gym
    import gym_fast_envs
    from utils.sampling import get_all_transitions

    N = 8
    transitions = get_all_transitions(gym.make(f'BlindCliffWalk-N{N}-v0'), n=N)
    mem = RankSampler(len(transitions), batch_size=32)

    for transition in transitions:
        mem.push(transition)

    for i in range(5):
        idxs = mem.sample()[0]
        print(idxs.float().mean())
        print([t.item() for t in list(idxs)])



if __name__ == '__main__':
    main()
