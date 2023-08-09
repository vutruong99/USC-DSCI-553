from pyspark import SparkContext
import os
import sys
import time
import math
import random
from itertools import combinations
import csv
from graphframes import *
from pyspark.sql import SQLContext

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Function to detect an edge between two users.
def has_edge(business_list_1, business_list_2):
    if len(list(set(business_list_1) & set(business_list_2))) >= threshold:
        return True
    else:
        return False

# Function format results.
def list_to_output(l):
    sorted_l = sorted(l)
    res = []
    for item in sorted_l:
        res.append("'" + str(item) + "'")
    return res

if __name__ == "__main__":
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]

    sc = SparkContext()
    sqlContext = SQLContext(sc)
    sc.setLogLevel("ERROR")

    # Read data
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

    # Create edges list and vertices list.
    for user_pair in user_pairs:
        user1 = user_pair[0]
        user2 = user_pair[1]
        if has_edge(user_business_dictionary[user1], user_business_dictionary[user2]):
            edge_list.append((user1, user2, "friend"))
            edge_list.append((user2, user1, "friend"))
            vertex_set.add((user1,))
            vertex_set.add((user2,))
        else:
            continue

    # Create vertices and edges for GraphFrame.
    vertices = sqlContext.createDataFrame(list(vertex_set), ["id"])
    edges = sqlContext.createDataFrame(edge_list, ["src", "dst", "relationship"])

    # Create graph.
    g = GraphFrame(vertices, edges)

    # Find communities.
    result = g.labelPropagation(maxIter=5)

    result_rdd = result.rdd.map(list)
    
    result_rdd = result_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: sorted(x))

    result_list = result_rdd.sortBy(lambda x: (len(x[1]), x[1][0])).collect()

    with open(output_file_path, "w") as f:
        for item in result_list:
            users = item[1]
            f.writelines((", ".join(list_to_output(users))))
            f.write("\n")