from pyspark import SparkContext
import os
import sys
import json
import time
import csv

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == "__main__":
    test_review_path = sys.argv[1]
    business_path = sys.argv[2]
    output_a_path = sys.argv[3]
    output_b_path = sys.argv[4]

    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    review_rdd = sc.textFile(test_review_path).map(lambda row: json.loads(row))
    business_rdd = sc.textFile(business_path).map(lambda row: json.loads(row))

    businesses_and_stars = review_rdd.map(lambda review: (review["business_id"], review["stars"]))
    businesses_and_cities = business_rdd.map(lambda business: (business["business_id"], business["city"]))
    joined_rdd = businesses_and_stars.join(businesses_and_cities).values().\
        map(lambda v: (v[1], (v[0], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).\
        mapValues(lambda v: v[0] / v[1]).sortBy(lambda key: (-key[1], key[0])).collect()

    with open(output_a_path, 'w', newline='') as f:
        f.write("city,stars\n")
        for item in joined_rdd:
            city = str(item[0])
            stars = str(item[1])
            row = ",".join([city, stars])
            f.write(row + "\n")
            
    # ////////////////////////////////////////
    result_dictionary = dict()
    python_start_time = time.time()
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    review_rdd = sc.textFile(test_review_path).map(lambda row: json.loads(row))
    business_rdd = sc.textFile(business_path).map(lambda row: json.loads(row))

    businesses_and_stars = review_rdd.map(lambda review: (review["business_id"], review["stars"]))
    businesses_and_cities = business_rdd.map(lambda business: (business["business_id"], business["city"]))
    joined_rdd = businesses_and_stars.join(businesses_and_cities).values(). \
        map(lambda v: (v[1], (v[0], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])). \
        mapValues(lambda v: v[0] / v[1]).collect()

    python_sort = sorted(joined_rdd, key = lambda x: (-x[1], x[0]))
    print(python_sort[:10])
    python_end_time = time.time()
    result_dictionary["m1"] = python_end_time - python_start_time

    # ////////////////////////////////////////////////
    pyspark_start_time = time.time()
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    review_rdd = sc.textFile(test_review_path).map(lambda row: json.loads(row))
    business_rdd = sc.textFile(business_path).map(lambda row: json.loads(row))

    businesses_and_stars = review_rdd.map(lambda review: (review["business_id"], review["stars"]))
    businesses_and_cities = business_rdd.map(lambda business: (business["business_id"], business["city"]))
    joined_rdd = businesses_and_stars.join(businesses_and_cities).values().\
        map(lambda v: (v[1], (v[0], 1))).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])).\
        mapValues(lambda v: v[0] / v[1]).sortBy(lambda key: (-key[1], key[0])).take(10)
    print(joined_rdd)
    pyspark_end_time = time.time()

    result_dictionary["m2"] = pyspark_end_time - pyspark_start_time

    result_dictionary["reason"] = "For the two datasets test_review.json and business.json, I observe that " \
                                  "Python's execution time was roughly 12.51 seconds while that of Spark was around 24.34 seconds. " \
                                  "The reason for this difference is probably due to the fact that the dataset is not big enough " \
                                  "for Spark to demonstrate its ability to divide and conquer. Some of Spark's operations might have " \
                                  "made the process unnecessarily slower. This theory was proven by the results using the larger dataset, review.json: Python's execution "\
                                  "time was significantly slower than Spark's, specifically, 68.91 vs 45.11. This is reasonable since Spark is supposed to "\
                                  "perform better on huge datasets."

    with open(output_b_path, "w") as f:
        json.dump(result_dictionary, f)