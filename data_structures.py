""" Data structures used in experience replay implementations. """
import collections
import typing as T


class PriorityQueue:
    """ A priority queue which can update the priority the items it holds.
    """
    def __init__(self, data: T.Optional[list] = None):
        self.__heap = collections.deque()
        if data:
            for item in data:
                self.push(item)


    def push(self, item: tuple):
        """ Inserts the (priority, content) element and percolates it up in the
        binary heap.

        args:
            item (tuple(priority, content)): The item to be stored in the heap.
            It comprises of a priority term that is used for ordering the item
            in the data structure and the actual content of the item.
        """
        self.__heap.append(item)
        self.__sift_up(self.__len__()-1)


    def pop(self) -> tuple:
        """ Pops the item wih the largest priority and repairs the heap.
        """
        item = self.__heap.popleft()

        if len(self) > 1:
            self.__heap.appendleft(self.__heap.pop())
            self.__sift_down(0)

        return item


    def update(self, idx: int, new_priority: T.Union[int, float]):
        """ Updates the priority of an item.
        """
        old_priority, item = self.__heap[idx]
        self.__heap[idx] = (new_priority, item)

        if new_priority < old_priority:
            self.__sift_up(idx)
        else:
            self.__sift_down(idx)


    def __sift_up(self, i: int):
        """ Percolates up the item if the first element in the item is smaller
        than the first element in the parent.
        """
        while i > 0:
            parent = (i-1) // 2
            if self.__heap[i][0] < self.__heap[parent][0]:
                tmp = self.__heap[parent]
                self.__heap[parent] = self.__heap[i]
                self.__heap[i] = tmp
            i = parent


    def __sift_down(self, i: int):
        """ Percolates down the item if the first element in the item is larger
        than the first element in the child.
        """
        while (2*i+1) <= self.__len__()-1:

            child_idx = self.__get_smallest_child(i)

            if self.__heap[i][0] > self.__heap[child_idx][0]:
                tmp = self.__heap[i]
                self.__heap[i] = self.__heap[child_idx]
                self.__heap[child_idx] = tmp
            i = child_idx


    def __get_smallest_child(self, i):
        left, right = 2*i+1, 2*i+2
        if right > len(self)-1:
            return left
        if self.__heap[left][0] < self.__heap[right][0]:
            return left
        return right


    def __len__(self):
        return len(self.__heap)


    def __str__(self):
        out = ('-' * 24 + '\n')
        out += f'  PriorityQueue(N={len(self)})\n'
        out += ('-' * 24 + '\n')
        if self.__len__() < 30:
            for i, (prt, item) in enumerate(self.__heap):
                out += f'{i:3d}. {prt:5.2f}  -->  {item}\n'
        else:
            head = [self.__heap[i] for i in range(0, 5)]
            tail = [self.__heap[i] for i in range(len(self)-5, len(self))]
            for i, (prt, item) in enumerate(head):
                out += f'{i:3d}. {prt:8.2f}  -->  {item}\n'
            out += f'[ ... {len(self) - 10} more elements ]\n'
            for i, (prt, item) in enumerate(tail):
                out += f'{len(self)-5+i:3d}. {prt:8.2f}  -->  {item}\n'
        return out


def main():
    """ Entry point. """
    import numpy as np
    import random
    np.random.seed(10)

    data = [(i, f'item_{i}') for i in range(100)]
    random.shuffle(data)

    pqueue = PriorityQueue(data)

    for _ in range(35):
        idx = random.randint(0, 100-1)
        pqueue.update(idx, random.randint(0, 100))

    print(pqueue)
    res = [pqueue.pop()[0] for _ in range(len(data))]
    assert sorted(res) == res, "Blana"


if __name__ == "__main__":
    main()
