import numpy as np
import matplotlib.pyplot as plt
import math as m
from bisect import bisect_left # TODO: Check if I can use bisect_left

class problem1:

    def strategy_1(N : int):
        """
        Select one of the M bins uniformly at random
        input: bins # equals ball #
        output: most populated bin

        """
        # instantiate np bin of 
        bins = np.zeros(N, dtype=int)
        for _ in range(N):
            n = np.random.randint(low=0, high=N, size=None) # random number generator
            bins[n] += 1    
        return np.argmax(bins) + 1
        # return max bin

    def strategy_2(N):
        """
        select two of the N bins B_1 and B_2 at random
        Place the ball in whichever B_1 or B_2 that has fewest ball
        if B_1 equals B_2, then B_1 wins
        """
        bins = np.zeros(N, dtype=int)
        for _ in range(N):
            b_1 = np.random.randint(low=0, high=N, size=None)
            b_2 = np.random.randint(low= 0, high=N, size=None)
            # b_1 and b_2 are not the same number
            if bins[b_1] <= bins[b_2]:
                bins[b_1] += 1
            else:
                bins[b_2] += 1
        return np.argmax(bins) + 1 
    
    def strategy_3(N):
        """
        select 3 bins uniformly at random. Same strategy as strategy_2, 
        but break ties among the 3 bins
        if all three bins are tied, the first bins increments first

        """
        bins = np.zeros(N, dtype=int)
        for _ in range(N):
            b_1, b_2, b_3 = np.random.randint(N, size=3)
            if bins[b_1] == bins[b_2] == bins[b_3]:
                bins[b_1] += 1
            else:
                minimal_bin = np.argmin([bins[b_1], bins[b_2], bins[b_3]]) # passing one array
                if minimal_bin == 0:
                    bins[b_1] += 1
                elif minimal_bin == 1:
                    bins[b_2] += 1
                else:
                    bins[b_3] += 1
        return np.argmax(bins) + 1

    def strategy_4 (N):
        """
        select B_1 uniformly at random from the first (floor) N/2 bins
        and B_2 at the last N/2 (ceiling). 
        """
        bins = np.zeros(N, dtype=int)
        for _ in range(N):
            b_1 = np.random.randint(m.floor(N // 2))
            b_2 = np.random.randint(m.ceil(N // 2), N)
            if bins[b_1] <= bins[b_2]:
                bins[b_1] += 1
            else:
                bins[b_2] += 1
        return np.argmax(bins) + 1
    
    def simulate_with_cons_hashing(strategy, N):
        """
        Implemented an adaptation to the following.
        """
        bins = np.zeros(N, dtype=int)
        servers = sorted(np.random.sample(range(2**32), N))
        for i in range(N):
            x = np.random.randint(0, 2**32 - 1)
            bin_index = bisect_left(servers, x) % N
            bins[bin_index] += 1
        max_balls = strategy(bins)
        return max_balls
        """
        Select a bin at random (uniformly random)
        The rule: ball in interval preceding s_i falls in bin B_i
        """
        # How do I make a server to contain upto a preceding number of intervals in the bin? 
        # instantiate a bin
        # balls in a preceding s_i interval falls in bin B_i.
        # mod operation across all N 


        """
        Consistent hashing <- according to Akamai paper, is implemented by using:
        random cache trees & consistent hashing
        Plaxton/Rajaraman algo: swamping??
        """

    def plot_histogram(bins, filename=None):
        """
        This function wraps a number of hairy matplotlib calls to smooth the plotting 
        part of this assignment.

        Inputs:
        - bins:     numpy array of shape max_bin_population X num_strategies numpy array. For this 
                    assignment this must be 200000 X 4. 
                    WATCH YOUR INDEXING! The element bins[i,j] represents the number of times the most 
                    populated bin has i+1 balls for strategy j+1. 
        
        - filename: Optional argument, if set 'filename'.png will be saved to the current 
                    directory. THIS WILL OVERWRITE 'filename'.png
        """
        assert bins.shape == (200000, 4), "Input bins must be a numpy array of shape (max_bin_population, num_strategies)"
        assert np.array_equal(np.sum(bins, axis=0), (np.array([50, 50, 50, 50]))), "There must be 50 runs for each strategy"

        thresh = max(np.nonzero(bins)[0]) + 3
        n_bins = thresh
        bins = bins[:thresh, :]
        print("\nPLOTTING: Removed empty tail. Only the first non-zero bins will be plotted\n")

        ind = np.arange(n_bins)
        width = 1.0 / 6.0

        fig, ax = plt.subplots()
        rects_strat_1 = ax.bar(ind + width, bins[:, 0], width, color='yellow')
        rects_strat_2 = ax.bar(ind + width * 2, bins[:, 1], width, color='orange')
        rects_strat_3 = ax.bar(ind + width * 3, bins[:, 2], width, color='red')
        rects_strat_4 = ax.bar(ind + width * 4, bins[:, 3], width, color='k')

        ax.set_ylabel('Number Occurrences in 50 Runs')
        ax.set_xlabel('Number of Balls In The Most Populated Bin')
        ax.set_title('Histogram: Load on Most Populated Bin For Each Strategy')

        ax.set_xticks(ind)
        ax.set_xticks(ind + width * 3, minor=True)
        ax.set_xticklabels([str(i + 1) for i in range(0, n_bins)], minor=True)
        ax.tick_params(axis=u'x', which=u'minor', length=0)

        ax.legend((rects_strat_1[0], rects_strat_2[0], rects_strat_3[0], rects_strat_4[0]),
                ('Strategy 1', 'Strategy 2', 'Strategy 3', 'Strategy 4'))
        plt.setp(ax.get_xmajorticklabels(), visible=False)

        if filename is not None: plt.savefig(filename + '.png', bbox_inches='tight')

        plt.show()

"""
This is where the main functions runs to compute at least 1a) of the question

"""
if __name__ == '__main__':
    N = 200000
    num_simulation = 50
    bins = np.zeros((N, 4), dtype=int)

    strategies = [problem1.strategy_1, problem1.strategy_2,
                                      problem1.strategy_3, problem1.strategy_4]
    # for loop produces the number of strategies that can be completed using some stuff.
    for i in range(num_simulation):
        results = np.array([strategy(N) for strategy in strategies]) - 1
        # iterate for loop to check the stuff
        bins[results, np.arange(4)] += 1 
            # run each strategy
            # plot histogram
    problem1.plot_histogram(bins)