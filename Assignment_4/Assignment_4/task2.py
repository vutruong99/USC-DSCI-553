from pyspark import SparkContext
import os
import sys
import time
import math
from itertools import combinations, permutations
import csv
from collections import deque

# Function to format output.
def list_to_output(l):
    sorted_l = sorted(l)
    res = []
    for item in sorted_l:
        res.append("'" + str(item) + "'")
    return res

# Function to calculate modularity of a community.
def calculate_modularity(community):
    if len(community) == 1:
        degree = degree_dict[community[0]]
        return 2 * (0 - (degree * degree / (2 * m)))

    community_pairs = permutations(list(community), 2)

    modularity = 0

    for pair in community_pairs:
        first_vertex = pair[0]
        second_vertex = pair[1]

        degree_of_first_vertex = degree_dict[first_vertex]
        degree_of_second_vertex = degree_dict[second_vertex]

        if first_vertex in adjacency_list[second_vertex] or second_vertex in adjacency_list[first_vertex]:
            Aij = 1
        else:
            Aij = 0

        modularity += Aij - (degree_of_first_vertex * degree_of_second_vertex / (2 * m))

    return modularity

# Function to calculate betweenness of a graph.
def calculate_betweenness(adjacency_list, edge_list):
    betweenness_dict = dict()

    for edge in edge_list:
        user1 = edge[0]
        user2 = edge[1]

        sorted_vertices = sorted([user1, user2])
        betweenness_dict[(sorted_vertices[0], sorted_vertices[1])] = 0

    vertex_list = adjacency_list.keys()

    for vertex in vertex_list:
        shortest_distance = dict()
        number_of_shortest_paths = dict()
        parents = dict()
        credits = dict()
        vertices_at_level = dict()

        for v in vertex_list:
            shortest_distance[v] = -1
            number_of_shortest_paths[v] = 1
            parents[v] = []
            credits[v] = 1

        q = deque()
        q.append(vertex)

        shortest_distance[vertex] = 0

        visited_vertices = set()
        visited_vertices.add(vertex)

        while q:
            current = q.popleft()
            for adjacent_vertex in adjacency_list[current]:
                if adjacent_vertex not in visited_vertices:
                    visited_vertices.add(adjacent_vertex)
                    q.append(adjacent_vertex)

                if shortest_distance[adjacent_vertex] == -1:
                    shortest_distance[adjacent_vertex] = shortest_distance[current] + 1
                    parents[adjacent_vertex].append(current)

                    if number_of_shortest_paths[adjacent_vertex] < number_of_shortest_paths[current]:
                        number_of_shortest_paths[adjacent_vertex] = number_of_shortest_paths[current]

                    if shortest_distance[adjacent_vertex] not in vertices_at_level.keys():
                        vertices_at_level[shortest_distance[adjacent_vertex]] = []
                        vertices_at_level[shortest_distance[adjacent_vertex]].append(adjacent_vertex)
                    else:
                        vertices_at_level[shortest_distance[adjacent_vertex]].append(adjacent_vertex)
                else:
                    if shortest_distance[adjacent_vertex] == shortest_distance[current] + 1:
                        number_of_shortest_paths[adjacent_vertex] += number_of_shortest_paths[current]
                        parents[adjacent_vertex].append(current)

        levels = sorted(vertices_at_level.keys(), reverse=True)

        for level in levels:
            for vertex_at_level in vertices_at_level[level]:
                for parent in parents[vertex_at_level]:
                    edge_betweenness = credits[vertex_at_level] * number_of_shortest_paths[parent] / \
                                       number_of_shortest_paths[vertex_at_level]
                    try:
                        betweenness_dict[(vertex_at_level, parent)] += edge_betweenness
                    except:
                        betweenness_dict[(parent, vertex_at_level)] += edge_betweenness

                    credits[parent] += edge_betweenness

    for key, value in betweenness_dict.items():
        betweenness_dict[key] = value / 2

    sorted_betweenness_list = sorted(betweenness_dict.items(), key=lambda x: (-x[1], x[0][0]))
    largest_betweennesses = [sorted_betweenness_list[0][1]]
    cut_edges = [sorted_betweenness_list[0][0]]

    for item in sorted_betweenness_list[1:]:
        edge = item[0]
        betweenness = item[1]

        if betweenness == largest_betweennesses[-1]:
            largest_betweennesses.append(betweenness)
            cut_edges.append(edge)
        else:
            break

    return sorted_betweenness_list, cut_edges

# Function to format output.
def edge_to_output(edge, betweenness):
    result = "('" + str(edge[0]) + "', '" + str(edge[1]) + "'), " + str(round(betweenness, 5))
    return result

# Function to detect an edge between two users.
def has_edge(business_list1, business_list_2):
    if len(list(set(business_list1) & set(business_list_2))) >= threshold:
        return 1
    else:
        return 0

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

if __name__ == "__main__":
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    output_file_path_2 = sys.argv[4]

    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    
    # Read data.
    ub_sample_data_rdd = sc.textFile(input_file_path)
    headers = ub_sample_data_rdd.first()
    ub_sample_data_rdd = ub_sample_data_rdd.filter(lambda x: x != headers)
    ub_sample_data_rdd = ub_sample_data_rdd.map(lambda values: values.split(","))
    threshold = filter_threshold

    # Get a dictionary with users as keys and the businesses that they rated as values.
    # {user1: [business1, business2], user2: [business1, business3]}
    user_business_dictionary = ub_sample_data_rdd.groupByKey().mapValues(list).collectAsMap()

    # Get list of users.
    user_list = user_business_dictionary.keys()

    # Get user pairs.
    user_pairs = combinations(user_list, 2)

    vertex_set = set()
    edge_list = []

    for user_pair in user_pairs:
        user1 = user_pair[0]
        user2 = user_pair[1]
        if has_edge(user_business_dictionary[user1], user_business_dictionary[user2]):
            edge_list.append((user1, user2))
            vertex_set.add(user1)
            vertex_set.add(user2)
        else:
            continue

    # Create and populate adjacency list.
    adjacency_list = dict()

    for vertex in vertex_set:
        adjacency_list[vertex] = []
        
    for edge in edge_list:
        user1 = edge[0]
        user2 = edge[1]

        adjacency_list[user1].append(user2)
        adjacency_list[user2].append(user1)

    # Calculate the betweenness of the original graph.
    original_betweenness, first_edge_to_cut = calculate_betweenness(adjacency_list, edge_list)
    
    with open(output_file_path, "w") as f:
        for item in original_betweenness:
            f.writelines(edge_to_output(item[0], item[1]))
            f.write("\n")

    m = len(edge_list)

    # Degree dictionary for vertices.
    degree_dict = dict()
    for key, value in adjacency_list.items():
        degree_dict[key] = len(value)
    
    max_modularity = 0
    final_community = [vertex_set.copy()]
    for community in final_community:
        
        max_modularity += calculate_modularity(community)
    max_modularity = max_modularity / (2*m)
    
    # Cut edges until no edges are left.
    count = 1
    while (len(edge_list) > 0):
        if count == 1:
            cut_edges = first_edge_to_cut
            count += 1
        else:
            _, cut_edges = calculate_betweenness(adjacency_list, edge_list)
        
        # Cut the edges with the highest betweenness
        for cut_edge in cut_edges:
            
            vertex_1 = cut_edge[0]
            vertex_2 = cut_edge[1]

            adjacency_list[vertex_1].remove(vertex_2)
            adjacency_list[vertex_2].remove(vertex_1)

            try:
                edge_list.remove(cut_edge)
            except:
                edge_list.remove((cut_edge[1], cut_edge[0]))

        community_list = []

        # Find communities.
        for vertex in vertex_set:
            flag = True
            for community in community_list:
                if vertex in community:
                    flag = False

            # If this node is already visited and in a community already, move to the next node.
            if (flag == False):
                continue

            q = deque()
            q.append(vertex)

            visited_vertices = set()
            visited_vertices.add(vertex)

            while (q):
                current = q.popleft()
                for vertex in adjacency_list[current]:
                    if vertex not in visited_vertices:
                        visited_vertices.add(vertex)
                        q.append(vertex)

            community_list.append(sorted(visited_vertices))

        total_modularity = 0

        # Calculate the modularity for each community and sum them up.
        for community in community_list:
            total_modularity += calculate_modularity(community)
           
        total_modularity = total_modularity / (2*m)
        
        # Keep track of the maximum modularity and its community.
        if total_modularity >= max_modularity:
            max_modularity = total_modularity
            
            final_community = community_list
   
    final_result = []

    for community in final_community:
        final_result.append(sorted(community))

    with open(output_file_path_2, "w") as f:
        for item in sorted(final_result, key=lambda x: (len(x), x[0])):
            f.writelines((", ".join(list_to_output(item))))
            f.write("\n")
            