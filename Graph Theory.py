# spec.py
"""Volume II Lab 7: Breadth-First Search (Kevin Bacon)
Allyson Tom
"""

"""
Graph theory has many practical applications. A graph may represent a complex system or network, and analyzing the graph often
reveals critical information about the network. In this lab, we learn to store graphs as adjacency dictionaries, implement a 
breadth-first search to identify the shortest path between two nodes, then use the NetworkX package to solve the so-called
"Kevin Bacon problem".
"""

from collections import deque
import networkx as nx


class Graph(object):
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a list of the
    corresponding node's neighbors.

    Attributes:
        dictionary: the adjacency list of the graph.
    """

    def __init__(self, adjacency):
        """Store the adjacency dictionary as a class attribute."""
        self.dictionary = adjacency

    
    def __str__(self):
        """String representation: a sorted view of the adjacency dictionary.
        
        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> print(Graph(test))
            A: B
            B: A; C
            C: B
        """
        keys = sorted(self.dictionary.keys())
        my_list = []
        for i in keys:
            values = sorted(self.dictionary[i])
            my_list.append(str(i) + ": " + "; ".join(values))
        return "\n".join(my_list)

        

    
    def traverse(self, start):
        """Begin at 'start' and perform a breadth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation).

        Raises:
            ValueError: if 'start' is not in the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> Graph(test).traverse('B')
            ['B', 'A', 'C']
        """
        if start not in self.dictionary.keys():
            raise ValueError('Starting node is not in graph.')

        marked = set()
        visited = list()
        visit_queue = deque()

        visit_queue.append(start)
        marked.add(start)
        current = visit_queue.popleft()

        while current is not None:
            visited.append(current)
            for neighbor in self.dictionary[current]:
                if neighbor not in marked:
                    visit_queue.append(neighbor)
                    marked.add(neighbor)
            if len(visit_queue) == 0:
                current = None
            else:
                current = visit_queue.popleft()
        return visited

    
    def DFS(self, start):
        """Begin at 'start' and perform a depth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited. If 'start' is not in the
        adjacency dictionary, raise a ValueError.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation)
        """
        if start not in self.dictionary.keys():
            raise ValueError('Starting node is not in graph.')

        marked = set()
        visited = list()
        visit_queue = deque()

        visit_queue.append(start)
        marked.add(start)
        current = visit_queue.pop()

        while current is not None:
            visited.append(current)
            for neighbor in self.dictionary[current]:
                print neighbor
                if neighbor not in marked:
                    visit_queue.append(neighbor)
                    marked.add(neighbor)
            if len(visit_queue) == 0:
                current = None
            else:
                current = visit_queue.pop()
        return visited 

    
    def shortest_path(self, start, target):
        """Begin at the node containing 'start' and perform a breadth-first
        search until the node containing 'target' is found. Return a list
        containg the shortest path from 'start' to 'target'. If either of
        the inputs are not in the adjacency graph, raise a ValueError.

        Inputs:
            start: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from start to target,
                including the endpoints.

        Example:
            >>> test = {'A':['B', 'F'], 'B':['A', 'C'], 'C':['B', 'D'],
            ...         'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'],
            ...         'G':['A', 'F']}
            >>> Graph(test).shortest_path('A', 'G')
            ['A', 'F', 'G']
        """
        path = dict()
        if start not in self.dictionary.keys() or target not in self.dictionary.keys():
            raise ValueError('Starting or target node is not in graph.')

        marked = set()
        visited = list()
        visit_queue = deque()

        visit_queue.append(start)
        marked.add(start)
        path[start] = None
        current = visit_queue.popleft()

        while current is not target:
            visited.append(current)
            for neighbor in self.dictionary[current]:
                if neighbor not in marked:
                    visit_queue.append(neighbor)
                    path[neighbor] = current
                    marked.add(neighbor)
            if len(visit_queue) == 0:
                current = None
            else:
                current = visit_queue.popleft()   
                
        shortest_path = []
        i = target
        while i is not None:
            shortest_path.append(i)
            i = path[i]
        return shortest_path[::-1]
        
        




def convert_to_networkx(dictionary):
    """Convert 'dictionary' to a networkX object and return it."""
    nx_graph = nx.Graph()
    for i in dictionary.keys():
        for j in dictionary[i]:
            nx_graph.add_edge(i,j)
    return nx_graph


# Helper function
def parse(filename="movieData.txt"):
    """Generate an adjacency dictionary where each key is
    a movie and each value is a list of actors in the movie.
    """

    # open the file, read it in, and split the text by '\n'
    with open(filename, 'r') as movieFile:
        moviesList = movieFile.read().split('\n')
    graph = dict()

    # for each movie in the file,
    for movie in moviesList:
        # get movie name and list of actors
        names = movie.split('/')
        title = names[0]
        graph[title] = []
        # add the actors to the dictionary
        for actor in names[1:]:
            graph[title].append(actor)
    
    return graph



class BaconSolver(object):
    """Class for solving the Kevin Bacon problem."""

    
    def __init__(self, filename="movieData.txt"):
        """Initialize the networkX graph and with data from the specified
        file. Store the graph as a class attribute. Also store the collection
        of actors in the file as an attribute.
        """
        self.dictionary = parse(filename)
        self.nx_graph = convert_to_networkx(self.dictionary)
        actor_set = set()
        for i in self.dictionary.keys():
            for j in self.dictionary[i]:
                if j not in actor_set:
                    actor_set.add(j)
        self.actors = actor_set

    
    def path_to_bacon(self, start, target="Bacon, Kevin"):
        """Find the shortest path from 'start' to 'target'."""
        if start not in self.actors or target not in self.actors:
            raise ValueError('Start or target actor not in data set.')
        return nx.shortest_path(self.nx_graph, start, target)

    
    def bacon_number(self, start, target="Bacon, Kevin"):
        """Return the Bacon number of 'start'."""
        return (len(self.path_to_bacon(start, target))-1)/2

    
    def average_bacon(self, target="Bacon, Kevin"):
        """Calculate the average Bacon number in the data set.
        Note that actors are not guaranteed to be connected to the target.

        Inputs:
            target (str): the node to search the graph for
        """
        
        my_sum = 0
        not_connected = 0
        for actor in self.actors:
            try:
                my_sum += self.bacon_number(actor,target)
            except nx.NetworkXNoPath:
                not_connected += 1
        return (my_sum+0.)/(len(self.actors)-not_connected),not_connected
        

# =========================== END OF FILE =============================== #


