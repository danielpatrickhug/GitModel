import networkx as nx
from networkx.algorithms import community

def community_detection(A):
    # Convert the adjacency matrix to a NetworkX graph
    G = nx.from_numpy_array(A)

    # Perform Girvan-Newman community detection
    communities_generator = community.girvan_newman(G)

    # Get the first level of communities
    communities = next(communities_generator)

    # Sort communities by size (number of nodes)
    sorted_communities = sorted(communities, key=lambda x: len(x), reverse=True)

    return sorted_communities

def print_communities(data, A):
    # Print sorted communities
    sorted_communities = community_detection(data, A)
    for i, community in enumerate(sorted_communities):
        print(f"Community {i + 1}:")
        print(f"Nodes: {list(community)}")
        component_files_structure = []
        component_text_structure = []
        component_history_structure = []
        for row_key in list(community):
            file_component = data.loc[row_key, "file_name"]
            text_component = data.loc[row_key, "assistant_reply"]
            history_component = data.loc[row_key, "conversation_history"]
            component_files_structure.append(file_component)
            component_text_structure.append(text_component)
            component_history_structure.append(history_component)
        print(component_files_structure)
        for text, history in zip(component_text_structure, component_history_structure):
            # print(history[-2]['content'])
            print()
            print(text)
            print()
        print()


def dfs(v, visited, adj_matrix, component):
    '''Depth-first search algorithm.'''
    visited[v] = True
    component.append(v)
    for i, val in enumerate(adj_matrix[v]):
        if val > 0 and not visited[i]:
            dfs(i, visited, adj_matrix, component)


def connected_components(adj_matrix):
    '''Find connected components in a graph represented by an adjacency matrix.'''
    visited = [False for _ in range(adj_matrix.shape[0])]
    components = []

    for v in range(adj_matrix.shape[0]):
        if not visited[v]:
            component = []
            dfs(v, visited, adj_matrix, component)
            components.append(component)

    return components