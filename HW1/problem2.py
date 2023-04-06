import numpy as np
import matplotlib.pyplot as plt
import math as m
from bisect import bisect_left # TODO: Check if I can use bisect_left

class problem2:

    def compute_heavy_hitters():
        total_length = sum((1000*(i+1)-1000*i+1) for i in range(1, 10))  # rule 1
        total_length += sum(i**2 for i in range(1, 61))  # rule 2

        # identify heavy hitters
        # heavy_hitters = []
        # for i in range(1, 10001):
        #     counter = i * (1000 * (i // 1000 + 1) - 1000 * (i // 1000)) if i <= 9000 else i**2
        #     if counter >= 0.01 * total_length:
        #         heavy_hitters.append(i)
        heavy_hitters = []
        for i in range(1, 10001):
            count = i*(1000*(i//1000+1)-1000*(i//1000)) if i <= 9000 else i**2  # compute the count of i in the stream
            if count >= 0.01*total_length:
                heavy_hitters.append(i)
        
        print(len(heavy_hitters))

    if __name__ == '__main__':
        compute_heavy_hitters()