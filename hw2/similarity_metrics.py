from scipy.spatial import KDTree
import yaml

class similarity_metrics:
    """
    This is an implementation of a KDTree.
    
    The tree can be queried for the 'r' closest neighbors of any given point (optionally returning only those within some maximum distance of the point).
    It can also be queries, with a substantial gain in efficiency, for the r approximate closest neighbors.
    
    """    

    def add_in_tree():
        """
        add data set, 
        newsgroup_data.yaml is used to parse the data into the map of map
        
        """
        # scan the file
        with open('newsgroup_data.yaml') as nd:
            dictionary_data = yaml.safe_load(nd)    
        print(dictionary_data['alt.atheism'].keys())

    if __name__ == '__main__':
        add_in_tree()