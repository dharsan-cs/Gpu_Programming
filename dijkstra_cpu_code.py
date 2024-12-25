
class Node : 
    def  __init__( self ,id ): 
        self.id = id 
        self.dist = float('inf')
        self.par = None  
        self.visited : bool = False

def cpu_dijkstra( distance_matrix ,str_node ):
    assert len(distance_matrix) == len(distance_matrix[0]) ,"distance matrix is not square"
    assert str_node > 0  and str_node <  len(distance_matrix ) ,"str_node is invalid"
    
    ##creating a list of node elements  
    num_node = len(distance_matrix)
    node_list = [ Node(i) for i in range( num_node ) ]
    
    ##setting dist ,found value for start node 
    node_list[str_node].dist = 0 
    for _ in range( num_node - 1 ):
        ## choosing a non-visted node with minimum dist value  
        min_dist = float('inf')
        choosen_node = None 
        for node in node_list: 
            if not node.visited and node.dist < min_dist :
                min_dist = node.dist
                choosen_node = node 
        
        if choosen_node is None : 
            print( "Logical error ")
            break 
        
        choosen_node.visited = True  
        
        ## relaxing the remaining node based on the choosen node  
        for node in node_list : 
            if not node.visited and distance_matrix[node.id][choosen_node.id] > 0 :
                new_dist = choosen_node.dist + distance_matrix[node.id][choosen_node.id]
                if node.dist > new_dist  : 
                    node.dist = new_dist 
                    node.par = choosen_node.id  
    
    return node_list
     








    
    