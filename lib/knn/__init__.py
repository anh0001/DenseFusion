# lib/knn/__init__.py

import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Function
from lib.knn import knn_pytorch as knn_pytorch  # Ensure this path is correct

class KNearestNeighbor(Function):
    """Compute k nearest neighbors for each query point."""

    @staticmethod
    def forward(ctx, ref, query, k):
        """
        Forward pass to compute k-nearest neighbors.

        Args:
            ref (torch.Tensor): Reference points of shape (batch_size, D, N).
            query (torch.Tensor): Query points of shape (batch_size, D, M).
            k (int): Number of nearest neighbors to find.

        Returns:
            torch.Tensor: Indices of k-nearest neighbors of shape (batch_size, k, M).
        """
        # Ensure inputs are on CUDA and of type float
        ref = ref.float().cuda()
        query = query.float().cuda()

        # Initialize tensor to hold indices
        inds = torch.empty(query.shape[0], k, query.shape[2], dtype=torch.long, device=query.device)

        # Call the KNN CUDA implementation
        knn_pytorch.knn(ref, query, inds)

        return inds

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass (not needed for KNN as it's non-differentiable).

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            Tuple[None, None, None]: Gradients with respect to inputs (None for all).
        """
        # KNN is non-differentiable; no gradients are passed backward
        return None, None, None

class TestKNearestNeighbor(unittest.TestCase):
    """Unit tests for KNearestNeighbor."""

    def test_forward(self):
        """
        Test the forward pass of KNearestNeighbor.
        """
        k = 2  # Number of nearest neighbors
        while True:
            D, N, M = 128, 100, 1000  # Dimensions and number of points
            ref = torch.rand(2, D, N, device='cuda')
            query = torch.rand(2, D, M, device='cuda')

            # Apply the KNN function
            inds = KNearestNeighbor.apply(ref, query, k)

            # Optional: Inspect tensors in memory (can be removed if not needed)
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    size = functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0
                    print(size, type(obj), obj.size())

            print(inds)
            break  # Remove or adjust as needed to prevent infinite loop

if __name__ == '__main__':
    unittest.main()