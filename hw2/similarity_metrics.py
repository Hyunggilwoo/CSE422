import yaml
import math as m
import matplotlib.pyplot as plt
import numpy as np
import warnings



def import_data_set(file_name='newsgroup_data.yaml')-> dict:
    """
    Opens the .yaml file into a nested map.
    
    """
    dictionary_data = dict()
    # scan the file
    with open(file_name) as nd:
        dictionary_data = yaml.safe_load(nd)    
    print(dictionary_data['alt.atheism'].keys())
    # print(len(dictionary_data.get(alt.atheism).keys()))
    return dictionary_data
    

def jaccard_similarity(x, y):
    """
    pre: x and y must be same data set with the same length
    Implements the Jacard similarity
    sum of min(x_i, y_i) / sum of max(x_i, y_i)
    """
    # if both sets are empty, return 0 instead of undefined
    if not x and not y:
        return 0
    
    #TODO: if map length = 0, the return 0
#    min  = 0
    # max = 0

    top = sum(min(x.get(word, 0), y.get(word, 0)) for word in set(x) | set(y))
    bottom = sum(max(x.get(word, 0), y.get(word, 0)) for word in set(x) | set(y))

    # If there is no union, then just return 0    
    # if max == 0: return 0
    
    return top / bottom # first testing 
    
def compute_jaccard_similarity(data):
    similarities = {}
    newsgroups = list(data.keys())

    for i in range (len(newsgroups)):
        for j in range(i + 1, len(newsgroups)):

            newsgroup_pair = (newsgroups[i], newsgroups[j])
            similarities[newsgroup_pair]= {}

            article_1 = data[newsgroups[i]]
            article_2 = data[newsgroups[j]]


            for article_1_id, article_1_content in article_1.items(): # compute article
                for article_2_id, article_2_content in article_2.items():
                    article_pair = (article_1_id, article_2_id)
                    similarities[newsgroup_pair][article_pair] = jaccard_similarity(article_1_content, article_2_content)

    return similarities
        # compute article_1 from A and article_2 from B
            
def makeHeatMap(data, name, color, outputFileName):
    """
    How to implement heatmap's color:
    """
    # To catch "falling back" to Agg warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = plt.subplots()
    # create table with color bar legend
        heatmap = ax.pcolor(data, cmap=color)
        cbar = plt.colorbar(heatmap)

        ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

        # want a more natural, table like display
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(range(1, 21))
        ax.set_yticklabels(name)

        plt.tight_layout()
        plt.savefig(outputFileName, format = "png")
        plt.close()



#    def print_heat_map():

if __name__ == '__main__':
    
    data = import_data_set()
    jaccard_data_set = compute_jaccard_similarity(data)
    makeHeatMap(jaccard_data_set, "Yo_mama", "plt.cm.Blues", "Jaccard_similarity")

