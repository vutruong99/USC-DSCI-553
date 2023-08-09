# Method Description:
# Firstly, I collected the features that I assumed to be useful such as the number of categories of a business or the positivity of the tips, and fit all of them into a baseline model (XGBoost).
# Next, I filtered out irrelevant features by removing them one by one and monitoring the RMSE. I kept the features that produced the best RMSE on the validation set.
# Finally, I experimented with 3 different regression models including CatBoost and RandomForest. I found that CatBoost performed better than the other two so I kept it as my final model.

# Error Distribution:
# >=0 and <1: 102310
# >=1 and <2: 32813
# >=2 and <3: 6112
# >=3 and <4: 809
# >=4: 0

# RMSE:
# 0.9769828608295233

# Execution Time:
# 361s (354s for training the model on the training set, 7s for predicting the validation set)

from pyspark import SparkContext
import os
import sys
import math
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import json
from sklearn import preprocessing
from catboost import CatBoostRegressor
import csv
import time
from sklearn.preprocessing import MultiLabelBinarizer

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def average(l):
    return sum(l) / len(l)

if __name__ == "__main__":
    folder_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    # Read the necessary features from the JSON files and convert them into dictionaries
    business_features_dictionary = sc.textFile(folder_path + "/business.json").map(lambda x: json.loads(x)).map(
        lambda x: (
            x["business_id"], (
                x["stars"], x["review_count"], x["attributes"], x["categories"], x["hours"], x["latitude"],
                x["longitude"]))).collectAsMap()

    tip_features_dictionary = sc.textFile(folder_path + "/tip.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], (x["text"]))).collectAsMap()

    tip_likes_dictionary = sc.textFile(folder_path + "/tip.json").map(lambda x: json.loads(x)).map(
        lambda x: (x['business_id'], x['likes'])).reduceByKey(lambda x, y: x + y).collectAsMap()

    photo_features_dictionary = sc.textFile(folder_path + "/photo.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], 1)).reduceByKey(lambda x, y: x + y).collectAsMap()

    check_features_dictionary = sc.textFile(folder_path + "/checkin.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["business_id"], sum(list(x["time"].values())))).collectAsMap()

    user_features_dictionary = sc.textFile(folder_path + "/user.json").map(lambda x: json.loads(x)).map(
        lambda x: (x["user_id"], (x["average_stars"], x["review_count"], x["yelping_since"], x["elite"],
                                  len(x["friends"].split(",")), x["useful"], x["funny"], x["cool"], x["compliment_hot"],
                                  x["compliment_cute"], x["compliment_cool"], x["compliment_funny"],
                                  x["compliment_photos"],
                                  x["compliment_more"], x["compliment_profile"], x["compliment_list"],
                                  x["compliment_note"], x["compliment_plain"], x["compliment_writer"]))).collectAsMap()

    review_features_dictionary = sc.textFile(folder_path + "/review_train.json").map(lambda x: json.loads(x)).map(
        lambda x: ((x["user_id"], x["business_id"]), x["stars"])).collectAsMap()

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
    attributes = []
    categories = []
    hours = []
    tip_counts = []
    photo_counts = []
    checkin_counts = []
    lats = []
    longs = []
    business_tip_likes = []
    ohe_cats = []

    user_ratings = []
    user_review_counts = []
    user_yelping_since = []
    user_elite = []
    user_friends = []
    user_funny = []
    user_useful = []
    user_cool = []

    user_comp_hot = []
    user_comp_cute = []
    user_comp_cool = []
    user_comp_funny = []
    user_comp_photos = []

    user_comp_more = []
    user_comp_profile = []
    user_comp_list = []
    user_comp_note = []
    user_comp_plain = []
    user_comp_writer = []

    # Target rating
    true_ratings = []

    keywords = ["great", "awesome", "wonderful", "beautiful", "good", "delicious", "yum", "love", "best", "excellent",
                "fantastic"]

    user_rating_average = dict()

    categories_dict = dict()
    for business_user, star in review_features_dictionary.items():
        user = business_user[0]
        business = business_user[1]

        if user not in user_rating_average:
            user_rating_average[user] = []
            user_rating_average[user].append(star)
        else:
            user_rating_average[user].append(star)

    avg_stars = []
    avg_review_counts = []
    avg_user_ratings = []
    avg_user_review_counts = []

    for business_user, rating in business_user_rating_list_dictionary.items():
        business = business_user[0]
        user = business_user[1]

        try:
            star = business_features_dictionary[business][0]
            review_count = business_features_dictionary[business][1]
            user_rating = user_features_dictionary[user][0]
            user_review_count = user_features_dictionary[user][1]
            avg_stars.append(star)
            avg_review_counts.append(review_count)
            avg_user_ratings.append(user_rating)
            avg_user_review_counts.append(user_review_count)
        except:
            continue

        try:
            for category in business_features_dictionary[business][3].split(", "):
                categories_dict[category] = categories_dict.get(category, 0) + 1
        except:
            continue

    sorted_categories_dict = {k: v for k, v in sorted(categories_dict.items(), key=lambda item: -item[1])}
    top_20_categories = []

    c = 0
    for key, value in sorted_categories_dict.items():
        c += 1
        top_20_categories.append(key)
        if c == 20:
            break

    average_stars = average(avg_stars)
    average_review_counts = average(avg_review_counts)
    average_user_ratings = average(avg_user_ratings)
    average_user_review_counts = average(avg_user_review_counts)

    # Mapping the features to their businesses
    for business_user, rating in business_user_rating_list_dictionary.items():
        business = business_user[0]
        user = business_user[1]

        try:
            star = business_features_dictionary[business][0]
        except:
            star = 3.5
        try:
            review_count = business_features_dictionary[business][1]
        except:
            review_count = average_review_counts
        try:

            attribute_trues = 0
            attribute = business_features_dictionary[business][2]
            for key, value in attribute.items():

                if value == "True":
                    attribute_trues += 1
        except:
            attribute_trues = 0

        try:
            lat = business_features_dictionary[business][5]
            long = business_features_dictionary[business][6]
        except:
            lat = 0
            long = 0

        try:
            category = business_features_dictionary[business][3]
            category_length = len(category.split(" "))

            ohe_cat = []
            for cat in category.split(", "):
                if cat in top_20_categories:
                    ohe_cat.append(cat)
        except:
            category_length = 0
            ohe_cat = []
        try:
            hour = business_features_dictionary[business][4]
            hour_length = len(hour)
        except:
            hour_length = 0
        try:
            tip_score = 0

            tip_text = tip_features_dictionary[business]
            text = tip_text.split(" ")
            for t in text:
                if t.lower() in keywords:
                    tip_score += 1
            for char in tip_text:
                if char == "!":
                    tip_score += 1
        except:
            tip_score = 0
        try:
            photo_count = photo_features_dictionary[business]
        except:
            photo_count = 0
        try:
            checkin_count = check_features_dictionary[business]
        except:
            checkin_count = 0
        try:
            user_rating = user_features_dictionary[user][0]
        except:
            user_rating = 3.5
        try:
            user_friend = user_features_dictionary[user][4]
        except:
            user_friend = 0
        try:
            user_yelping_time = user_features_dictionary[user][2]
            user_yelping_year = 2023 - int(user_yelping_time.split("-")[0])
        except:
            user_yelping_year = 0
        try:
            user_review_count = user_features_dictionary[user][1]
        except:
            user_review_count = average_user_review_counts
        try:
            business_like = tip_likes_dictionary[business]
        except:
            business_like = 0
        try:
            useful = user_features_dictionary[user][5]
            funny = user_features_dictionary[user][6]
            cool = user_features_dictionary[user][7]
            comp_hot = user_features_dictionary[user][8]
            comp_cute = user_features_dictionary[user][9]
            comp_cool = user_features_dictionary[user][10]
            comp_funny = user_features_dictionary[user][11]
            comp_photos = user_features_dictionary[user][12]
            comp_more = user_features_dictionary[user][13]
            comp_profile = user_features_dictionary[user][14]
            comp_list = user_features_dictionary[user][15]
            comp_note = user_features_dictionary[user][16]
            comp_plain = user_features_dictionary[user][17]
            comp_writer = user_features_dictionary[user][18]
        except:
            useful = 0
            funny = 0
            cool = 0
            comp_hot = 0
            comp_cute = 0
            comp_cool = 0
            comp_funny = 0
            comp_photos = 0
            comp_more = 0
            comp_profile = 0
            comp_list = 0
            comp_note = 0
            comp_plain = 0
            comp_writer = 0

        true_rating = rating

        business_tip_likes.append(business_like)

        stars.append(star)
        review_counts.append(review_count)
        attributes.append(attribute_trues)
        categories.append(category_length)
        hours.append(hour_length)
        tip_counts.append(tip_score)
        photo_counts.append(photo_count)
        checkin_counts.append(checkin_count)
        lats.append(lat)
        longs.append(long)
        ohe_cats.append(ohe_cat)

        user_ratings.append(user_rating)
        user_review_counts.append(user_review_count)
        user_friends.append(user_friend)
        user_yelping_since.append(user_yelping_year)
        user_useful.append(useful)
        user_funny.append(funny)
        user_cool.append(cool)

        user_comp_hot.append(comp_hot)
        user_comp_cute.append(comp_cute)
        user_comp_funny.append(comp_funny)
        user_comp_cool.append(comp_cool)
        user_comp_photos.append(comp_photos)

        user_comp_more.append(comp_more)
        user_comp_profile.append(comp_profile)
        user_comp_list.append(comp_list)
        user_comp_note.append(comp_note)
        user_comp_plain.append(comp_plain)
        user_comp_writer.append(comp_writer)

        true_ratings.append(true_rating)

    mlb = MultiLabelBinarizer()

    ohe_categories = pd.DataFrame(mlb.fit_transform(ohe_cats),
                       columns=mlb.classes_, index = None)

    # Prepare the training data
    X_train = pd.DataFrame(
        list(zip(stars, review_counts, tip_counts, checkin_counts, user_ratings, lats, longs,
                 attributes, categories, hours, user_review_counts, user_useful, user_funny, user_cool, user_comp_hot,
                 user_comp_cute, user_comp_funny, user_comp_cool, user_comp_photos,
                 user_comp_more, user_comp_profile, user_comp_list, user_comp_note, user_comp_plain, user_comp_writer)),
        columns=["stars", "review_counts", "tip_counts", "checkin_counts",
                 "user_ratings", "lats", "longs", "attributes", "categories", "hours", "user_review_counts",
                 "user_useful", "user_funny", "user_cool", "user_comp_hot", "user_comp_cute", "user_comp_funny",
                 "user_comp_cool", "user_comp_photos", "user_comp_more", "user_comp_profile", "user_comp_list",
                 "user_comp_note",
                 "user_comp_plain", "user_comp_writer"])
    
    X_train = pd.concat([X_train, ohe_categories], axis = 1)
    y_train = pd.DataFrame(true_ratings, columns=["true_ratings"]).values
    

    model = CatBoostRegressor(iterations=5000, silent=True, allow_writing_files = False)

    model.fit(X_train, y_train)

    val_users = []
    val_businesses = []
    # Prepare testing data
    val_stars = []
    val_review_counts = []
    val_attributes = []
    val_categories = []
    val_hours = []
    val_photo_counts = []
    val_checkin_counts = []
    val_tip_counts = []
    val_lats = []
    val_longs = []
    val_business_tip_likes = []
    val_ohe_cats = []

    val_user_ratings = []
    val_user_review_counts = []
    val_user_yelping_since = []
    val_user_friends = []
    val_user_useful = []
    val_user_funny = []
    val_user_cool = []

    val_user_comp_hot = []
    val_user_comp_cute = []
    val_user_comp_cool = []
    val_user_comp_funny = []
    val_user_comp_photos = []

    val_user_comp_more = []
    val_user_comp_profile = []
    val_user_comp_list = []
    val_user_comp_note = []
    val_user_comp_plain = []
    val_user_comp_writer = []

    for business_user in val_business_user_list:
        business = business_user[1]
        user = business_user[0]

        val_users.append(user)
        val_businesses.append(business)
        try:
            star = business_features_dictionary[business][0]
        except:
            star = average(user_rating_average[user])
        try:
            review_count = business_features_dictionary[business][1]
        except:
            review_count = average_review_counts
        try:

            attribute_trues = 0
            attribute = business_features_dictionary[business][2]
            for key, value in attribute.items():

                if value == "True":
                    attribute_trues += 1

        except:
            attribute_trues = 0

        try:
            lat = business_features_dictionary[business][5]
            long = business_features_dictionary[business][6]
        except:
            lat = 0
            long = 0

        try:
            category = business_features_dictionary[business][3]
            category_length = len(category.split(" "))
            
            ohe_cat = []
            for cat in category.split(", "):
                if cat in top_20_categories:
                    ohe_cat.append(cat)
        except:
            category_length = 0
            ohe_cat = []
        try:
            hour = business_features_dictionary[business][4]
            hour_length = len(hour)
        except:
            hour_length = 0

        try:
            tip_score = 0
            tip_text = tip_features_dictionary[business]
            text = tip_text.split(" ")
            for t in text:
                if t.lower() in keywords:
                    tip_score += 1
            for char in tip_text:
                if char == "!":
                    tip_score += 1
        except:
            tip_score = 0
        try:
            photo_count = photo_features_dictionary[business]
        except:
            photo_count = 0
        try:
            checkin_count = check_features_dictionary[business]
        except:
            checkin_count = 0
        try:
            user_rating = user_features_dictionary[user][0]
        except:
            user_rating = average_user_ratings
        try:
            user_friend = user_features_dictionary[user][4]
        except:
            user_friend = 0
        try:
            user_yelping_time = user_features_dictionary[user][2]
            user_yelping_year = 2023 - int(user_yelping_time.split("-")[0])
        except:
            user_yelping_year = 0
        try:
            user_review_count = user_features_dictionary[user][1]
        except:
            user_review_count = average_user_review_counts

        try:
            useful = user_features_dictionary[user][5]
            funny = user_features_dictionary[user][6]
            cool = user_features_dictionary[user][7]
            comp_hot = user_features_dictionary[user][8]
            comp_cute = user_features_dictionary[user][9]
            comp_cool = user_features_dictionary[user][10]
            comp_funny = user_features_dictionary[user][11]
            comp_photos = user_features_dictionary[user][12]
            comp_more = user_features_dictionary[user][13]
            comp_profile = user_features_dictionary[user][14]
            comp_list = user_features_dictionary[user][15]
            comp_note = user_features_dictionary[user][16]
            comp_plain = user_features_dictionary[user][17]
            comp_writer = user_features_dictionary[user][18]
        except:
            useful = 0
            funny = 0
            cool = 0
            comp_hot = 0
            comp_cute = 0
            comp_cool = 0
            comp_funny = 0
            comp_photos = 0
            comp_more = 0
            comp_profile = 0
            comp_list = 0
            comp_note = 0
            comp_plain = 0
            comp_writer = 0

        try:
            business_like = tip_likes_dictionary[business]
        except:
            business_like = 0

        val_stars.append(star)
        val_review_counts.append(review_count)
        val_attributes.append(attribute_trues)
        val_categories.append(category_length)
        val_hours.append(hour_length)
        val_tip_counts.append(tip_score)
        val_photo_counts.append(photo_count)
        val_checkin_counts.append(checkin_count)
        val_lats.append(lat)
        val_longs.append(long)
        val_ohe_cats.append(ohe_cat)

        val_user_ratings.append(user_rating)
        val_user_review_counts.append(user_review_count)
        val_user_friends.append(user_friend)
        val_user_yelping_since.append(user_yelping_year)
        val_user_useful.append(useful)
        val_user_funny.append(funny)
        val_user_cool.append(cool)
        val_business_tip_likes.append(business_like)

        val_user_comp_hot.append(comp_hot)
        val_user_comp_cute.append(comp_cute)
        val_user_comp_cool.append(comp_cool)
        val_user_comp_funny.append(comp_funny)
        val_user_comp_photos.append(comp_photos)

        val_user_comp_more.append(comp_more)
        val_user_comp_profile.append(comp_profile)
        val_user_comp_list.append(comp_list)
        val_user_comp_note.append(comp_note)
        val_user_comp_plain.append(comp_plain)
        val_user_comp_writer.append(comp_writer)
    
    val_ohe_categories = pd.DataFrame(mlb.fit_transform(val_ohe_cats),
                       columns=mlb.classes_, index = None)
    X_val = pd.DataFrame(
        list(zip(val_stars, val_review_counts, val_tip_counts, val_checkin_counts, val_user_ratings,
                 val_lats, val_longs, val_attributes, val_categories, val_hours, val_user_review_counts,
                 val_user_useful, val_user_funny, val_user_cool, val_user_comp_hot, val_user_comp_cute,
                 val_user_comp_funny, val_user_comp_cool, val_user_comp_photos,
                 val_user_comp_more, val_user_comp_profile, val_user_comp_list, val_user_comp_note, val_user_comp_plain,
                 val_user_comp_writer)),
        columns=["stars", "review_counts", "tip_counts", "checkin_counts",
                 "user_ratings", "lats", "longs", "attributes", "categories", "hours", "user_review_counts",
                 "user_useful", "user_funny", "user_cool", "user_comp_hot", "user_comp_cute", "user_comp_funny",
                 "user_comp_cool", "user_comp_photos", "user_comp_more", "user_comp_profile", "user_comp_list",
                 "user_comp_note", "user_comp_plain", "user_comp_writer"])
    
    X_val = pd.concat([X_val, val_ohe_categories], axis = 1)
    # Make predictions
    y_pred = model.predict(X_val)

    # Writing the data into the file
    for i, business_user in enumerate(val_business_user_list):
        business_user.append(y_pred[i])

    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        writer.writerows(val_business_user_list)