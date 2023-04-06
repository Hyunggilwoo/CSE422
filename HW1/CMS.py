import numpy as np
import matplotlib.pyplot as plt
import math as m
import hashlib # import MD5

class CMS:
    """
    Representation of Count-Min Sketch.
    hash-table is present and number of buckets are needed to represent it.
    As an implementer, we modulate the episilon for the percentage that the computation is correct, and
    also the percentage of when the computation is also correct.

    """
    # TODO: hard coding l and b value for the count-min sketch. 
    def __init__(self, s: int):
        """
        Takes s as an input.
        """
        self.s = s
        # should 2-D array's row = 5, not hash function?
        self.l = [hashlib.md5] * 5 
        self.b = 256

        # Does the array index have the same diemsnion as the dimension accessed by INC?
        self.CMS = np.zeros((self.l, self.b), dtype = int)
    
    def inc(self, x:int, s:int):
        """
        Takes s as input.
        Where does s fit in incrementing the stuff?"""
        for i in range(self.b):
            # TODO: How should h(x)
            h_i = self.l[i](x) % s # hi(x) := MD5(x * i) mod b 
            self.CMS[i][h_i] += 1 # 


    def count(self, x:int, s:int) -> int:
        """
        Return the estimated count of an item in CMS
        """
        min_count = float('inf')

        for i in range(self.b):
            h_i = self.l[i](x) % s
            min_count = min(min_count, self.array[i][h_i])

        return min_count
    
        """
        The five hash functions h1,h2,h3,h4,h5:U→{0,1,…,255}
        are computed as follows:

        For x∈U, define hi(x) to be the i_th byte of the MD5 hash of str(x)∘str(s), where str(⋅)
        is the string representation of an integer and ∘ corresponds to concatenation of strings.
        """

        # computing the five hash functions

    def count_total_length():
        total_length = 0
        for i in range(1, 10):
            sum(i*(1000*(i+1) - 1000*i))
            
    def heavy_hitter(array):
        """
        Implementing the heavy hitter algorithm that was computed earlier as 
        a baseline.
        """
        counter = 0
        current = None
        length = len(array)
        for i in range(length):
            if counter == 0:
                current = array[i]
                counter += 1
            elif array[i] == current:
                counter += 1
            else:
                counter += -1
        return current
    
            