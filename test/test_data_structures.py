""" Test for the data_structures module. """
import random
import unittest
import heapq
from bayesian_per.data_structures import PriorityQueue


class TestPriorityQueue(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPriorityQueue, self).__init__(*args, **kwargs)
        self.max_n = 100

    def test_pop_of_reversed_insert(self):
        """ pop N=[1, 100] reversed inserts."""
        N = [i for i in range(1, self.max_n)]
        dataset = [[(i, None) for i in range(n)] for n in N]

        for data in dataset:
            with self.subTest(msg=f'Failed for N={len(data)}', data=data):
                pq = PriorityQueue(list(reversed(data)))
                res = [pq.pop() for _ in range(len(data))]
                self.assertEqual(data, res)


    def test_pop_of_ordered_insert(self):
        """ pop N=[1, 100] orderly inserts."""
        N = [i for i in range(1, self.max_n)]
        dataset = [[(i, None) for i in range(n)] for n in N]

        for data in dataset:
            with self.subTest(msg=f'failed for N={len(data)}', data=data):
                pq = PriorityQueue(data)
                res = [pq.pop() for _ in range(len(data))]
                self.assertEqual(data, res)


    def test_pop_of_random_insert_without_replacement(self):
        """ pop N=[1, 100] random non-repeating inserts."""
        N = [i for i in range(1, self.max_n)]
        dataset = [zip(random.sample(range(n), k=n), [None for _ in range(n)])
                   for n in N]

        for data in dataset:
            data = list(data)
            with self.subTest(msg=f'failed for N={len(data)}', data=data):
                pq = PriorityQueue(data)
                res = [pq.pop() for _ in range(len(data))]
                self.assertEqual(sorted(data), res)


    def test_pop_of_random_insert_with_replacement(self):
        """ pop N=[1, 100] random repeating inserts."""
        N = [i for i in range(1, self.max_n)]
        dataset = [zip(random.choices(range(n), k=n), [None for _ in range(n)])
                   for n in N]

        for data in dataset:
            data = list(data)
            with self.subTest(msg=f'failed for N={len(data)}', data=data):
                pq = PriorityQueue(data)
                res = [pq.pop() for _ in range(len(data))]
                self.assertEqual(sorted(data), res)


    def test_heap_property(self):
        """ heapq == PriorityQueue for N=[1, 100] reversed items."""
        N = [i for i in range(1, self.max_n)]
        dataset = [[(i, None) for i in range(n)] for n in N]

        for data in dataset:
            with self.subTest(msg=f'Failed for N={len(data)}', data=data):
                hpq = []
                pq = PriorityQueue()
                for item in reversed(data):
                    heapq.heappush(hpq, item)
                    pq.push(item)
                self.assertEqual(hpq, list(pq._PriorityQueue__heap))


    def test_pop_after_one_update(self):
        """ pop 30 orderly inserts after one update """
        data = [(i, f'item_{i}') for i in range(30)]

        pq = PriorityQueue(data)

        pq.update(21, 3.5)
        data[21] = (3.5, data[21][1])

        res = [pq.pop() for _ in range(len(data))]
        self.assertEqual(sorted(data), res)


    def test_pop_after_updates(self):
        """ pop 100 orderly inserts after 35 updates """
        data = [(i, f'item_{i}') for i in range(100)]
        random.shuffle(data)

        pq = PriorityQueue(data)

        for _ in range(35):
            idx = random.randint(0, 100-1)
            pq.update(idx, random.randint(0, 100))

        res = [pq.pop()[0] for _ in range(len(data))]
        self.assertEqual(sorted(res), res)
