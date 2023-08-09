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

def calculate_average_ratings(x):
    count = 0
    total = 0
    for item in x:
        count = count + 1
        total = total + item[1]
    if count != 0:
        return total / count
    else:
        return 0

def pearson_similarity(list_of_businesses):
    # list_of_business = (business1, bussiness2, bussiness4)
    new_list_of_businesses = []
    similarity_dict = set()

    # Only calculate pearson similarity for valid businesses
    for business in list_of_businesses:
        if business in valid_business:
            new_list_of_businesses.append(business)

    # All valid pairs of businesses
    combos = combinations(new_list_of_businesses, 2)

    for combo in combos:
        # If we already calculated similarity for this pair, skip it
        if (combo) not in visited_pairs:
            visited_pairs.add(combo)
        else:
            continue

        business1 = combo[0]
        business2 = combo[1]

        # Get the average for each business
        business_1_average = business_average_rating_dictionary[business1]
        business_2_average = business_average_rating_dictionary[business2]

        # Get the lists of users for each business
        business_1_users = business_user_dictionary[business1]
        business_2_users = business_user_dictionary[business2]

        co_rated_users = list(set(business_1_users) & set(business_2_users))

        # If this pair has less than 50 co-rated user, skip them
        if len(co_rated_users) <= 50:
            continue
        else:
            numerator = 0
            denom_business_1 = 0
            denom_business_2 = 0
            for i in range(len(co_rated_users)):
                user = co_rated_users[i]
                numerator += (business_user_rating_dictionary[business1][user] - business_1_average) * (
                        business_user_rating_dictionary[business2][user] - business_2_average)
                denom_business_1 += (business_user_rating_dictionary[business1][user] - business_1_average) ** 2
                denom_business_2 += (business_user_rating_dictionary[business2][user] - business_2_average) ** 2

            if denom_business_1 != 0 and denom_business_2 != 0:
                similarity = numerator / ((math.sqrt(denom_business_1)) * math.sqrt(denom_business_2))
            else:
                similarity = 0

            if similarity > 0:
                similarity_dict.add((combo, similarity))

    return similarity_dict

def predict(user, target_business):
    # If both user and business do not exist in train dataset, return 3
    if user not in user_business_rating_dictionary.keys() and target_business not in business_average_rating_dictionary.keys():
        return 3

    # If user does not exist, return the average rating of that business
    if user not in user_business_rating_dictionary.keys():
        return business_average_rating_dictionary[target_business]

    # If business does not exist, return the average rating of that user
    if target_business not in business_average_rating_dictionary.keys():
        return user_average_rating_dictionary[user]

    # Else, predict the rating using similarities
    user_ratings = user_business_rating_dictionary[user]
    numerator = 0
    denominator = 0
    for business, rating in user_ratings.items():
        if (target_business, business) in pearson_similarities_dictionary.keys():
            similarity = pearson_similarities_dictionary[(target_business, business)]
        else:
            try:
                similarity = pearson_similarities_dictionary[(business, target_business)]
            except:
                similarity = 0

        numerator += rating * similarity
        denominator += abs(similarity)

    # If there is no common business between business_similarity and user_rating, return the average rating of that business
    if denominator != 0:
        return numerator / denominator
    else:
        return business_average_rating_dictionary[target_business]

if __name__ == "__main__":
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("ERROR")

    yelp_train_rdd = sc.textFile(train_file_path)
    yelp_headers = yelp_train_rdd.first()
    yelp_train_rdd = yelp_train_rdd.filter(lambda x: x != yelp_headers)
    yelp_train_rdd = yelp_train_rdd.map(lambda values: values.split(","))

    yelp_val_rdd = sc.textFile(test_file_path)
    yelp_val_rdd = yelp_val_rdd.filter(lambda x: x != yelp_headers)
    yelp_val_rdd = yelp_val_rdd.map(lambda values: values.split(","))

    # (business1: (user1: rating1, user3: rating3), business2: (user2: rating2, user3: rating3))
    business_user_rating_rdd = yelp_train_rdd.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey()

    # (user1: {business1: rating1, business2: rating2})
    user_business_rating_rdd = yelp_train_rdd.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey()
    # {business1: {user1: rating1, user3: rating3}, business2: {user2: rating2, user3: rating3}}
    business_user_rating_dictionary = yelp_train_rdd.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(
        lambda x: dict(x)).collectAsMap()

    # {user1: {business1: rating1, business2: rating2}, user3: {business4: rating4, business5: rating5}}
    user_business_rating_dictionary = yelp_train_rdd.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(
        lambda x: dict(x)).collectAsMap()

    # {business1 : (user1, user2, user3), business2: (user3, user4, user5)}
    business_user_dictionary = yelp_train_rdd.map(lambda x: (x[1], x[0])).groupByKey().collectAsMap()

    # {user1: (business1, business2, business3), user2: (business3, business4)} AKA list of businesses with at least 1 shared user.
    user_business_rdd = yelp_train_rdd.map(lambda x: (x[0], x[1])).groupByKey()

    # {business1: avg1, business2: avg2}
    business_average_rating_dictionary = business_user_rating_rdd.mapValues(
        lambda x: calculate_average_ratings(x)).collectAsMap()

    user_average_rating_dictionary = user_business_rating_rdd.mapValues(
        lambda x: calculate_average_ratings(x)).collectAsMap()

    # Global list of visited pairs
    visited_pairs = set()

    # Get businesses with >= 35 ratings
    valid_business = yelp_train_rdd.map(lambda x: (x[1], x[0])).groupByKey().filter(
        lambda x: len(x[1]) >= 35).keys().collect()

    # Get Pearson similarities between businesses {(business1, business2): similarity12, (business2, business3): similarity23}
    pearson_similarities_dictionary = user_business_rdd.flatMap(lambda x: pearson_similarity(x[1])).groupByKey() \
        .mapValues(lambda x: list(x)[0]).collectAsMap()

    # Predict test/validation set
    predictions = yelp_val_rdd.map(lambda x: (x[0], x[1])).map(lambda x: (x[0], x[1], predict(x[0], x[1]))).collect()

    # writing the data into the file
    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        writer.writerows(predictions)