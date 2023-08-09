from pyspark import SparkContext
import os
import sys
import json
from datetime import datetime
date_format = "%Y-%m-%d %H:%M:%S"

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == "__main__":
    test_review_path = sys.argv[1]
    output_file_path = sys.argv[2]
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")
    test_review_rdd = sc.textFile(test_review_path).map(lambda row: json.loads(row))
    result_dictionary = dict()

    # Task 1A
    review_count = test_review_rdd.map(lambda review: review["review_id"]).count()

    # Task 1B
    review_2018_count = test_review_rdd.map(lambda review: review["date"]).\
        filter(lambda review: datetime.strptime(review, date_format).year == 2018).count()

    # Task 1C
    distinct_user_count = test_review_rdd.map(lambda review: review["user_id"]).distinct().count()

    # Task 1D
    top_10_user_reviews_count =  test_review_rdd.map(lambda review: (review["user_id"], 1)).\
        reduceByKey(lambda x, y: x + y).sortBy(lambda review: (-review[1], review[0])).take(10)

    # Task 1E
    distinct_business_count = test_review_rdd.map(lambda review: review["business_id"]).distinct().count()

    # Task 1F
    top_10_business_reviews_count = test_review_rdd.map(lambda review: (review["business_id"], 1)).\
        reduceByKey(lambda x, y: x + y).sortBy(lambda review: (-review[1], review[0])).take(10)

    result_dictionary["n_review"] = review_count
    result_dictionary["n_review_2018"] = review_2018_count
    result_dictionary["n_user"] = distinct_user_count
    result_dictionary["top10_user"] = top_10_user_reviews_count
    result_dictionary["n_business"] = distinct_business_count
    result_dictionary["top10_business"] = top_10_business_reviews_count

    with open(output_file_path, "w") as f:
        json.dump(result_dictionary, f)
