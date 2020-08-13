import unittest
# from pygco import cut_simple, cut_from_graph
from pygco import cut_from_graph
import numpy as np


class TestPyGco(unittest.TestCase):
    """Test the main pygco methods."""

    def setUp(self):
        """Set the random seed for reproducability."""
        np.random.seed(1234)

    def binary_data(self):
        # generate trivial data
        x = np.ones((10, 10))
        x[:, 5:] = -1

        x_noisy = x + np.random.normal(0, 0.8, size=x.shape)

        # create unaries
        unaries = x_noisy
        # Split into two channels, positive and negative
        unaries = np.dstack([unaries, -unaries])
        # as we convert to int, we need to multipy to get sensible values
        unaries = (10 * unaries.copy("C")).astype(np.int32)

        expected = np.zeros(x.shape, dtype=np.int32)
        # The left side has a high cost for class 0 and the right side
        # has a high cost for class 1
        expected[:, :5] = 1

        # create potts pairwise
        pairwise = -10 * np.eye(2, dtype=np.int32)

        # construct edges from a grid graph
        inds = np.arange(x.size).reshape(x.shape)
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        edges = np.vstack([horz, vert]).astype(np.int32)

        return unaries, pairwise, edges, expected


    def binary_data_float(self):
        # generate trivial data
        x = np.ones((10, 10))
        x[:, 5:] = -1

        x_noisy = x + np.random.normal(0, 0.8, size=x.shape)

        # create unaries
        unaries = x_noisy
        # Split into two channels, positive and negative
        unaries = np.dstack([unaries, -unaries])

        expected = np.zeros(x.shape, dtype=np.int32)
        # The left side has a high cost for class 0 and the right side
        # has a high cost for class 1
        expected[:, :5] = 1

        # create potts pairwise
        pairwise = -10 * np.eye(2, dtype=np.int32)

        # construct edges from a grid graph
        inds = np.arange(x.size).reshape(x.shape)
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        edges = np.vstack([horz, vert]).astype(np.int32)

        return unaries, pairwise, edges, expected


    # def test_cut_simple(self):
    #     """Test the cut_simple method."""
    #     unaries, pairwise, edges, expected = self.binary_data()

    #     result = cut_simple(unaries, pairwise)

    #     self.assertTrue(np.array_equal(result, expected))

    # def test_cut_from_grpah(self):
    #     """Test the cut_from_graph method."""
    #     unaries, pairwise, edges, expected = self.binary_data()

    #     result = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
    #     result = result.reshape(unaries.shape[:2])

    #     self.assertTrue(np.array_equal(result, expected))

    # def test_label_costs_simple(self):
    #     """Test the label_costs argument with cut_simple."""
    #     unaries, pairwise, edges, expected = self.binary_data()
    #     # Give a slight preference to class 0
    #     unaries[:, :, 1] += 1

    #     result = cut_simple(unaries, pairwise, label_cost=1)
    #     self.assertTrue(np.array_equal(result, expected))

    #     # Try again with a very high label cost to collapse to a single label
    #     result = cut_simple(unaries, pairwise, label_cost=1000)
    #     self.assertTrue(np.array_equal(result, np.zeros_like(result)))

    # def test_label_costs_graph(self):
    #     """Test the label_costs argument with cut_from_graph."""
    #     unaries, pairwise, edges, expected = self.binary_data()
    #     # Give a slight preference to class 0
    #     unaries[:, :, 1] += 1

    #     result = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise, label_cost=1)
    #     result = result.reshape(unaries.shape[:2])
    #     self.assertTrue(np.array_equal(result, expected))

    #     # Try again with a very high label cost to collapse to a single label
    #     result = cut_from_graph(edges, unaries.reshape(-1, 2), pairwise, label_cost=1000)
    #     result = result.reshape(unaries.shape[:2])
    #     self.assertTrue(np.array_equal(result, np.zeros_like(result)))

    def test_label_costs_graph(self):
        """Test the label_costs argument with cut_from_graph."""
        unaries, pairwise, edges, expected = self.binary_data()
        # unaries, pairwise, edges, expected = self.binary_data_float()
        # Give a slight preference to class 0
        unaries[:, :, 1] += 1
        
        label_cost = np.array([1,1])
        # label_cost = np.random.rand(2)

        weight=np.random.rand(edges.shape[0]).astype(np.float32)
        
        unaries=unaries.astype(np.float32)
        pairwise=pairwise.astype(np.float32)
        label_cost=label_cost.astype(np.float32)
        print(unaries, unaries.dtype)
        print(pairwise, pairwise.dtype)
        print(edges.shape)
        print(expected, expected.dtype)
        print(label_cost, label_cost.dtype)
   
        result = cut_from_graph(edges, weight, unaries.reshape(-1, 2), pairwise, label_cost)
        result = result.reshape(unaries.shape[:2])
        print(result)
        self.assertTrue(np.array_equal(result, expected))

if __name__ == '__main__':
    unittest.main()
