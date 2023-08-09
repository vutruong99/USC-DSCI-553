from pyspark import SparkContext
import os
import sys
import math
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import json
import csv

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

if __name__ == "__main__":
    folder_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    # Read the necessary features from the JSON files and convert them into dictionaries
    business_features_dictionary = sc.textFile(folder_path + "/business.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], (x["stars"], x["review_count"]))).collectAsMap()
    tip_features_dictionary = sc.textFile(folder_path + "/tip.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
    photo_features_dictionary = sc.textFile(folder_path + "/photo.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()
    check_features_dictionary = sc.textFile(folder_path + "/checkin.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], sum(list(x["time"].values())))).collectAsMap()
    user_features_dictionary = sc.textFile(folder_path + "/user.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["user_id"], x["average_stars"])).collectAsMap()

    yelp_train_path = folder_path + "/yelp_train.csv"

    yelp_train_rdd = sc.textFile(yelp_train_path)
    yelp_headers = yelp_train_rdd.first()
    yelp_train_rdd = yelp_train_rdd.filter(lambda x: x != yelp_headers)
    yelp_train_rdd = yelp_train_rdd.map(lambda values: values.split(","))

    yelp_val_rdd = sc.textFile(test_file_path)
    yelp_val_rdd = yelp_val_rdd.filter(lambda x: x != yelp_headers)
    yelp_val_rdd = yelp_val_rdd.map(lambda values: values.split(","))

    # (Business1, User1): Rating1, (Business2, User1): Rating2
    business_user_rating_list_dictionary = yelp_train_rdd.map(lambda x: ((x[1], x[0]), float(x[2]))).collectAsMap()

    val_business_user_list = yelp_val_rdd.map(lambda x: (x[0], x[1])).map(lambda x: list(x)).collect()

    # List of features
    stars = []
    review_counts = []
    tip_counts = []
    photo_counts = []
    checkin_counts = []
    user_ratings = []

    # Target rating
    true_ratings = []

    # Mapping the features to their businesses
    for business_user, rating in business_user_rating_list_dictionary.items():
        business = business_user[0]
        user = business_user[1]

        try:
            star = business_features_dictionary[business][0]
        except:
            star = 3
        try:
            review_count = business_features_dictionary[business][1]
        except:
            review_count = 0
        try:
            tip_count = tip_features_dictionary[business]
        except:
            tip_count = 0
        try:
            photo_count = photo_features_dictionary[business]
        except:
            photo_count = 0
        try:
            checkin_count = check_features_dictionary[business]
        except:
            checkin_count = 0
        try:
            user_rating = user_features_dictionary[user]
        except:
            user_rating = 3
            
        true_rating = rating
        stars.append(star)
        review_counts.append(review_count)
        tip_counts.append(tip_count)
        photo_counts.append(photo_count)
        checkin_counts.append(checkin_count)
        user_ratings.append(user_rating)
        true_ratings.append(true_rating)

    # Prepare the training data
    X_train = pd.DataFrame(list(zip(stars, review_counts, tip_counts, photo_counts, checkin_counts, user_ratings)),
                           columns=["stars", "review_counts", "tip_counts", "photo_counts", "checkin_counts",
                                    "user_ratings"])
    y_train = pd.DataFrame(true_ratings, columns=["true_ratings"]).values

    # Train the model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Prepare testing data
    val_stars = []
    val_review_counts = []
    val_photo_counts = []
    val_checkin_counts = []
    val_tip_counts = []
    val_user_ratings = []

    for business_user in val_business_user_list:
        business = business_user[1]
        user = business_user[0]
        
        try:
            star = business_features_dictionary[business][0]
        except:
            star = 3
        try:
            review_count = business_features_dictionary[business][1]
        except:
            review_count = 0
        try:
            tip_count = tip_features_dictionary[business]
        except:
            tip_count = 0
        try:
            photo_count = photo_features_dictionary[business]
        except:
            photo_count = 0
        try:
            checkin_count = check_features_dictionary[business]
        except:
            checkin_count = 0
        try:
            user_rating = user_features_dictionary[user]
        except:
            user_rating = 3

        val_stars.append(star)
        val_review_counts.append(review_count)
        val_checkin_counts.append(checkin_count)
        val_photo_counts.append(photo_count)
        val_tip_counts.append(tip_count)
        val_user_ratings.append(user_rating)

    X_val = pd.DataFrame(
        list(zip(val_stars, val_review_counts, val_checkin_counts, val_photo_counts, val_tip_counts, val_user_ratings)),
        columns=["stars", "review_counts", "tip_counts", "photo_counts", "checkin_counts", "user_ratings"])

    # Make predictions
    y_pred = model.predict(X_val)

    # Writing the data into the file
    for i, business_user in enumerate(val_business_user_list):
        business_user.append(y_pred[i])

    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        writer.writerows(val_business_user_list)