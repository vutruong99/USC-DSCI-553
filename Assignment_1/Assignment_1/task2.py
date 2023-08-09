from pyspark import SparkContext
import os
import sys
import json
import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == "__main__":
    result_dictionary = dict()
    default_dictionary = dict()
    custom_dictionary = dict()

    test_review_path = sys.argv[1]
    output_file_path = sys.argv[2]
    n_partition = int(sys.argv[3])

    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    
    # Default
    
    test_review_rdd = sc.textFile(test_review_path).map(lambda row: json.loads(row))
    task1F_rdd = test_review_rdd.map(lambda review: (review["business_id"], 1))

    default_start_time = time.time()
    task1F_rdd_default = task1F_rdd.reduceByKey(lambda x,y : x + y).sortBy(lambda review: (-review[1], review[0]))
    default_end_time = time.time()

    default_dictionary["n_partition"] = task1F_rdd.getNumPartitions()
    default_dictionary["n_items"] = task1F_rdd.glom().map(lambda x: len(x)).collect()
    default_dictionary["exe_time"] = default_end_time - default_start_time
    result_dictionary["default"] = default_dictionary

    # Custom

    test_review_rdd = sc.textFile(test_review_path).map(lambda row: json.loads(row))
    task1F_rdd = test_review_rdd.map(lambda review: (review["business_id"], 1)) \
        .partitionBy(n_partition, lambda x: ord(x[0][0]))

    custom_start_time = time.time()
    task1F_rdd_custom = task1F_rdd.reduceByKey(lambda x,y : x + y).sortBy(lambda review: (-review[1], review[0]))
    custom_end_time = time.time()

    custom_dictionary["n_partition"] = task1F_rdd.getNumPartitions()
    custom_dictionary["n_items"] = task1F_rdd.glom().map(lambda x: len(x)).collect()
    custom_dictionary["exe_time"] = custom_end_time - custom_start_time
    result_dictionary["customized"] = custom_dictionary

    with open(output_file_path, "w") as f:
        json.dump(result_dictionary, f)
