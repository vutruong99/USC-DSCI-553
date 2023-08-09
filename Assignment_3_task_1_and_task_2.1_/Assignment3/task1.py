from pyspark import SparkContext
import os
import sys
import time
import math
import random
from itertools import combinations
import csv

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Hash user strings into ints
def users_to_index(data, users):
    res = []
    for user in data:
        res.append(hash(user))

    return res


# Hash users into signatures
def hash_users(data):
    signatures = []
    for hash_function in hash_functions_list:
        a = hash_function[0]
        b = hash_function[1]
        p = hash_function[2]
        m = hash_function[3]

        min_index = m
        for user in data:

            new_index = ((a * user + b) & p) % m
            if new_index <= min_index:
                min_index = new_index

        signatures.append(min_index)

    return signatures

# Hash each portion of a band into a bucket
def hash_horizontally(one_business_signature, b, r):
    hashed_bands = []

    for i in range(b):
        pink_column = one_business_signature[(i * r): (i + 1) * r]
        hashed_column = hash(tuple(pink_column))
        hashed_bands.append((i, hashed_column))
    return hashed_bands


# Generate pairs from candidates (guarantee no duplicates)
def get_pairs(candidates):
    res = []
    for combo in combinations(candidates, 2):
        if combo not in visited_pairs:
            visited_pairs.add(combo)
            res.append(combo)
    return res

# Calculate Jaccard similarity between candidate items
def jaccard_similarity(candidate_pairs):
    true_pairs = []
    candidate_pairs = set(candidate_pairs)

    for candidate_pair in candidate_pairs:
        business_1_users = business_user_dict[candidate_pair[0]]
        business_2_users = business_user_dict[candidate_pair[1]]

        jaccard_similarity = float(float(len(set(business_1_users) & set(business_2_users))) / float(
            len(set(business_1_users) | set(business_2_users))))

        if jaccard_similarity >= 0.5:
            sorted_pair = sorted(candidate_pair)
            true_pairs.append((sorted_pair[0], sorted_pair[1], jaccard_similarity))

    return sorted(true_pairs)


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    yelp_train_rdd = sc.textFile(input_file_path)
    yelp_headers = yelp_train_rdd.first()
    yelp_train_rdd = yelp_train_rdd.filter(lambda x: x != yelp_headers)
    yelp_train_rdd = yelp_train_rdd.map(lambda values: values.split(","))

    # (business1: (user1, user2), business2: (user2, user3))
    businesses_list = yelp_train_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: sorted(x))

    # {business1: (user1, user2), business2: (user2, user3)}
    business_user_dict = businesses_list.collectAsMap()

    # Get unique users
    distinct_users = yelp_train_rdd.map(lambda row: row[0]).sortBy(lambda x: x).distinct().collect()

    # (business1: (381293, 312565), business2: (312565, 651323))
    businesses_list = businesses_list.mapValues(lambda x: users_to_index(x, distinct_users))

    # Get number of distinct users (m)
    distinct_users_count = yelp_train_rdd.map(lambda row: row[0]).sortBy(lambda x: x).distinct().count()

    n_hash_functions = 250
    band = 125
    r = 2
    m = distinct_users_count
    hash_functions_list = []
    p = 7202218937
    
    as_and_bs = [(45, 183), (161, 217), (102, 33), (119, 111), (236, 113), (28, 226), (84, 9), (121, 55), (171, 214), (36, 153), (104, 95), (106, 112), (229, 97), (123, 40), (128, 251), (171, 120), (246, 59), (28, 225), (59, 92), (157, 81), (107, 32), (2, 249), (106, 90), (167, 178), (7, 77), (36, 175), (44, 199), (234, 115), (79, 167), (188, 110), (45, 143), (58, 206), (126, 44), (215, 3), (37, 71), (27, 91), (34, 65), (105, 104), (241, 19), (117, 189), (113, 122), (161, 169), (27, 178), (204, 2), (9, 248), (82, 194), (115, 52), (205, 141), (83, 36), (222, 29), (115, 188), (83, 67), (73, 57), (152, 169), (3, 123), (166, 211), (243, 138), (12, 89), (202, 89), (101, 16), (54, 153), (95, 138), (178, 34), (210, 113), (208, 59), (191, 50), (167, 199), (198, 222), (182, 238), (68, 59), (2, 200), (14, 214), (2, 222), (89, 134), (185, 130), (180, 220), (251, 164), (29, 193), (44, 115), (74, 193), (125, 57), (112, 207), (91, 70), (116, 111), (28, 127), (249, 60), (250, 34), (103, 85), (73, 153), (13, 114), (213, 4), (186, 247), (20, 164), (146, 107), (224, 3), (75, 138), (93, 140), (6, 86), (190, 37), (137, 5), (187, 75), (34, 233), (217, 192), (232, 195), (178, 218), (44, 54), (213, 196), (242, 139), (65, 235), (203, 5), (22, 242), (21, 54), (169, 152), (133, 73), (100, 146), (141, 244), (119, 135), (89, 30), (104, 139), (137, 59), (234, 225), (213, 171), (148, 44), (46, 61), (55, 80), (81, 18), (174, 237), (138, 159), (159, 50), (13, 80), (84, 52), (230, 110), (121, 194), (152, 4), (243, 14), (145, 216), (131, 194), (93, 195), (32, 30), (134, 195), (23, 223), (126, 10), (183, 192), (159, 232), (48, 99), (120, 86), (34, 90), (100, 8), (192, 128), (131, 37), (207, 212), (232, 7), (220, 114), (87, 144), (15, 168), (68, 36), (13, 88), (36, 149), (52, 168), (60, 230), (207, 177), (130, 245), (246, 6), (61, 138), (148, 158), (225, 187), (218, 98), (208, 165), (147, 182), (241, 219), (117, 30), (204, 172), (164, 169), (108, 243), (30, 67), (69, 208), (158, 64), (232, 186), (33, 147), (169, 81), (40, 55), (183, 187), (197, 13), (160, 98), (191, 48), (227, 63), (114, 180), (139, 108), (165, 223), (236, 233), (157, 179), (248, 209), (11, 201), (61, 177), (234, 158), (86, 63), (141, 112), (153, 22), (107, 17), (248, 180), (113, 28), (187, 129), (202, 189), (201, 181), (90, 41), (166, 135), (67, 27), (95, 103), (239, 242), (113, 169), (210, 110), (105, 84), (243, 44), (94, 162), (38, 175), (26, 16), (132, 53), (210, 108), (213, 95), (133, 192), (116, 147), (153, 199), (77, 135), (6, 127), (120, 31), (87, 127), (238, 62), (32, 35), (95, 212), (20, 94), (93, 136), (244, 155), (61, 237), (155, 90), (142, 211), (62, 14), (129, 147), (164, 194), (220, 236), (227, 179), (135, 63), (195, 158), (205, 14), (42, 184), (236, 227), (199, 62), (102, 132), (248, 229), (223, 91), (228, 102)]

    # Generate hash functions
    for i in range(n_hash_functions):
        a = as_and_bs[i][0]
        b = as_and_bs[i][1]
        
        hash_functions_list.append([a, b, p, m])
    
    # Get signatures from users
    signatures_list = businesses_list.mapValues(lambda x: hash_users(x))

    # Hash portions of bands into K buckets
    k_buckets = signatures_list.flatMap(
        lambda x: [(tuple(hased_band), x[0]) for hased_band in hash_horizontally(x[1], band, r)])

    # Get candidates from buckets with length > 1
    candidates = k_buckets.groupByKey().mapValues(lambda x: list(x)).filter(lambda x: len(x[1]) > 1)
    
    visited_pairs = set()
    # Generate pairs from candidates
    candidate_pairs = candidates.flatMap(lambda x: get_pairs(x[1])).collect()

    # Get valid pairs (Jaccard similarity >= 0.5)
    true_pairs = jaccard_similarity(candidate_pairs)

    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["business_id_1", "business_id_2", "similarity"])
        writer.writerows(true_pairs)