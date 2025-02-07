import numpy as np
import pandas as pd

def direction(p_prev, p_last, p_current):
    # p_ are all points, with shape (n_points, 1, 2)
    vector_last = p_last - p_prev
    vector_current = p_current - p_last
    # print(vector_current)
    # print(vector_current[0, :, :])
    # print(vector_current[0, :, :].transpose())

    current_direction = [] # a list of directions for each point labeled

    for i in range(vector_current.shape[0]):
        dot_product = np.dot(vector_last[i, :, :], vector_current[i, :, :].transpose())
        # dot product determines the direction of change
        if dot_product >= 0: # when direction stays the same (perform addition)
            new_direction = 1
            current_direction.append(new_direction)
        elif dot_product < 0:
            new_direction = -1
            current_direction.append(new_direction)
        return current_direction # multiple current direction with the distance of recent movement, add to the cumulative distance

def euclidean(p_last, p_current):
    try:
        euclidean_distance = []
        for pt_idx in range(len(p_last)):
            euclidean_distance.append(np.linalg.norm(p_current[pt_idx, :, :] - p_last[pt_idx, :, :]))
            # print(euclidean_distance)
    except Exception as e:
        print(f"Something wrong with the values of p_last and p_current {e}")
    return euclidean_distance # returning a list of distances traveled by each point

class RollingBuffer:
    def __init__(self, size, dtype=np.float64):
        """
        Args:
        size (int): The maximum size of the buffer.
        dtype: The data type of the elements in the buffer. Default is np.float64.
        """
        self.buffer = np.zeros(size, dtype=dtype)
        self.size = size
        self.current_size = 0  # Keeps track of the current number of elements in the buffer

    def add(self, element):
        if self.current_size < self.size:
            # If the buffer is not full, add the element to the end and increment the size
            self.buffer[self.current_size] = element
            self.current_size += 1
        else:
            # If the buffer is full, shift the array left and add the new element to the last index
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = element

    def get(self):
        return self.buffer[:self.current_size]