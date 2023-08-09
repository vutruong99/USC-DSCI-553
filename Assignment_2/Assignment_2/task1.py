from pyspark import SparkContext
import os
import sys
from itertools import combinations
import time
import math

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# First pass of SON: Apriori.
def first_pass(partitioned_baskets, length, threshold):
    scaled_down_threshold = math.ceil(threshold/length)
    baskets = []
    frequent_singletons = []
    frequent_k = []

    # Get the baskets from the partitioned data.
    for item in partitioned_baskets:
        baskets.append(item)

    singletons_count = dict()

    # Count singletons.
    for basket in baskets:
        for item in basket:
            singletons_count[item] = singletons_count.get(item, 0) + 1

    # Determine frequent singletons.
    for item, count in singletons_count.items():
        if count >= scaled_down_threshold:
            frequent_singletons.append(item)

    # Start with k = 2 (doubles).
    k = 2
    new_baskets = []
    res = []

    # Set of frequent items from the previous k.
    previous_frequents_count = frequent_singletons

    # Result
    res = res + previous_frequents_count

    # Create new baskets without infrequent singletons.
    for basket in baskets:
        temp_basket_items = []
        for item in basket:
            if item in frequent_singletons:
                temp_basket_items.append(item)
        new_baskets.append(temp_basket_items)

    # While there are still frequent itemsets
    while (previous_frequents_count):
        k_count = dict()
        for basket in new_baskets:
            # Create combinations of length k using the singletons in the basket.
            k_combinations = combinations(basket, k)
            # Loop through the combinations, if there is a subset of that combination that does not exist in the
            # previous frequent itemsets list, that combination is invalid, else we count it.
            for combination in k_combinations:
                combination = tuple(sorted(combination))
                flag = True
                subsets = combinations(combination, k-1)
                for subset in subsets:
                    if k == 2:
                        if subset[0] not in previous_frequents_count:
                            flag = False
                            break
                    else:
                        if subset not in previous_frequents_count:
                            flag = False
                            break
                if flag == True:
                    k_count[combination] = k_count.get(combination, 0) + 1

        # Reset the frequent itemsets list.
        previous_frequents_count = []

        # After going through all the baskets, check the count of the validity of the combinations.
        for item, count in k_count.items():
            if count >= scaled_down_threshold:
                previous_frequents_count.append(item)

        # Add valid combinations to the result list.
        res = res + previous_frequents_count

        # Increase k.
        k += 1

    return res

# Second pass of SON.
def second_pass(partitioned_baskets, candidates):
    baskets = []
    # Get baskets from partitioned data.
    for item in partitioned_baskets:
        baskets.append(item)

    frequent_itemsets_count = dict()
    res = []
    for candidate in candidates:
        if isinstance(candidate, str):
            candidate = (candidate, )
        candidate = frozenset(candidate)
        for basket in baskets:
            basket = frozenset(basket)
            # If the candidate exists in the basket, count it.
            if candidate.issubset(basket):
                frequent_itemsets_count[candidate] = frequent_itemsets_count.get(candidate, 0) + 1

    # Return a list of items and their counts.
    for item, count in frequent_itemsets_count.items():
        res.append((item,count))

    return res

# Function to format output.
def list_to_output(l):
    temp_list = []
    start = "("
    end = ")"
    for item in l:
        temp = "'" + str(item) + "'"
        temp_list.append(temp)
    return start + ", ".join(temp_list) + end

if __name__ == "__main__":
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    start_time = time.time()
    input_rdd = sc.textFile(input_file_path)
    headers = input_rdd.first()
    input_rdd = input_rdd.filter(lambda x : x != headers)
    input_rdd = input_rdd.map(lambda values: values.split(","))

    if case_number == 1:
        case_rdd = input_rdd.groupByKey().mapValues(lambda x: list(set(x))).values()
    else:
        case_rdd = input_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: list(set(x))).values()

    length = case_rdd.getNumPartitions()

    # Pass 1.
    candidates = case_rdd.mapPartitions(lambda x: first_pass(x, length, support)).collect()

    # Pass 2.
    frequent_itemsets = case_rdd.mapPartitions(lambda x: second_pass(x, candidates)).reduceByKey(lambda x,y: x+y).\
    filter(lambda x: x[1] >= support).map(lambda x: list(x[0])).collect()

    # Sort the frequent itemsets.
    refined_frequents = []
    for frequent in frequent_itemsets:
        refined_frequents.append(sorted(frequent))
    refined_frequents = sorted(refined_frequents, key=lambda x:(len(x), x))

    # Sort the candidates.
    refined_candidates = []
    for item in candidates:
        if isinstance(item, str):
            if [item] not in refined_candidates:
                refined_candidates.append([item])
        else:
            item = list(item)
            if item not in refined_candidates:
                refined_candidates.append(item)
    refined_candidates = sorted(refined_candidates, key=lambda x: (len(x), x))

    # Write to file.
    with open(output_file_path, "w") as f:
        f.write("Candidates:\n")
        candidate_temp = []
        candidate_len = len(refined_candidates[0])
        for i in range(len(refined_candidates)):
            candidate_temp.append(list_to_output(refined_candidates[i]))
            if (i+1 >= len(refined_candidates)):
                f.write(",".join(candidate_temp) + "\n")
                f.write("\n")

            else:
                if (len(refined_candidates[i+1]) != candidate_len):
                    candidate_len = len(refined_candidates[i+1])
                    f.write(",".join(candidate_temp) + "\n")
                    f.write("\n")
                    candidate_temp = []
        f.write("Frequent Itemsets:\n")
        frequent_itemsets_temp = []
        frequent_itemsets_len = len(refined_frequents[0])
        for i in range(len(frequent_itemsets)):
            frequent_itemsets_temp.append(list_to_output(refined_frequents[i]))
            if (i+1 >= len(refined_frequents)):
                f.write(",".join(frequent_itemsets_temp) + "\n")
                f.write("\n")
                break
            else:
                if len(refined_frequents[i + 1]) != frequent_itemsets_len:
                    frequent_itemsets_len = len(refined_frequents[i + 1])
                    f.write(",".join(frequent_itemsets_temp) + "\n")
                    f.write("\n")
                    frequent_itemsets_temp = []

    end_time = time.time()
    print("Duration:", end_time - start_time)