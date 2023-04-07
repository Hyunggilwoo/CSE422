import yaml
import math as m

class similarity_metrics:
    """
    This is an implementation of a KDTree.
    
    The tree can be queried for the 'r' closest neighbors of any given point (optionally returning only those within some maximum distance of the point).
    It can also be queries, with a substantial gain in efficiency, for the r approximate closest neighbors.
    
    """    

    def import_data_set()-> dict:
        """
        Opens the .yaml file into a nested map.
        
        """
        dictionary_data = dict()
        # scan the file
        with open('newsgroup_data.yaml') as nd:
            dictionary_data = yaml.safe_load(nd)    
        # print(dictionary_data['alt.atheism'].keys())
        # print(len(dictionary_data.get(alt.atheism).keys()))
        return dictionary_data
    def jacard_similarity(x, y):
        """
        pre: x and y must be same data set with the same length
        Implements the Jacard similarity
        sum of min(x_i, y_i) / sum of max(x_i, y_i)
        """
        #TODO: if map length = 0, the return 0
        min  = 0
        max = 0
        # TODO: Find Jacard similarity across each article of the same type

        for (k, v), (k2, v2) in zip (x.items(), y.items()):
            min += m.min(v, v2)
            max += m.max(v, v2)

#    def print_heat_map():

    if __name__ == '__main__':
        import_data_set()