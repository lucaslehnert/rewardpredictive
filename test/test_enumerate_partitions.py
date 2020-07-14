#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
from unittest import TestCase


class TestEnumeratePartitions(TestCase):
    def test_enumerate_n_partitions_0(self):
        import rewardpredictive as rp
        import numpy as np

        part_list = rp.enumerate_partitions.enumerate_n_partitions(3, 1)
        self.assertEqual(len(part_list), 1)
        self.assertTrue(np.all(np.array(part_list, dtype=np.int) == np.array([[0, 0, 0]], dtype=np.int)))

    def test_enumerate_n_partitions_1(self):
        import rewardpredictive as rp
        import numpy as np

        part_list = rp.enumerate_partitions.enumerate_n_partitions(3, 2)
        self.assertEqual(len(part_list), 3)
        part_list_correct = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0]
        ], dtype=np.int)
        self.assertTrue(np.all(np.array(part_list, dtype=np.int) == part_list_correct))

    def test_enumerate_n_partitions_2(self):
        import rewardpredictive as rp
        import numpy as np

        part_list = rp.enumerate_partitions.enumerate_n_partitions(2, 2)
        part_list_correct = np.array([
            [0, 1]
        ], dtype=np.int)
        self.assertTrue(np.all(np.array(part_list, dtype=np.int) == part_list_correct))

    def test_enumerate_n_partitions_3(self):
        import rewardpredictive as rp
        import numpy as np

        part_list = rp.enumerate_partitions.enumerate_n_partitions(4, 2)
        part_list_correct = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0]
        ], dtype=np.int)
        self.assertTrue(np.all(np.array(part_list, dtype=np.int) == part_list_correct))

    def test_enumerate_n_partitions_4(self):
        import rewardpredictive as rp
        import numpy as np
        from itertools import product

        part_list = rp.enumerate_partitions.enumerate_n_partitions(6, 3)
        """
        Test if part list is a proper partition index matrix.
        """
        for i in range(3):
            cnt_set = set(np.sum(part_list == i, axis=-1))
            self.assertEqual(len(cnt_set - {1, 2, 3, 4}), 0)
        part_cnt = np.stack([np.sum(part_list == i, axis=-1) for i in range(3)])
        self.assertTrue(np.all(np.sum(part_cnt, axis=0) == 6))
        """
        The number of partitions is equal to Sterling number of the second kind. For partitioning six elements into 
        three partitions this number is 90.
        """
        self.assertEqual(np.shape(part_list)[0], 90)
        self.assertEqual(np.shape(part_list)[1], 6)
        """
        Test if each row in part_cnt is unique.
        """
        for i, j in product(range(90), range(90)):
            if i < j:
                self.assertFalse(np.all(part_list[i] == part_list[j]))

    def test_enumerate_all_partitions(self):
        import rewardpredictive as rp
        import numpy as np

        part_list = rp.enumerate_partitions.enumerate_all_partitions(3)
        part_list_correct = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 2]
        ], dtype=np.int)
        self.assertTrue(np.all(np.array(part_list, dtype=np.int) == part_list_correct))

        # print(np.shape(rp.enumerate_all_partitions(4)))
