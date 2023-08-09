import numpy as np
from sklearn.cluster import KMeans
import random
import math
import sys

indexes = []
cluster_indexes = []
points = []

def calculate_mahalanobis_distance(point, cluster):
    N = cluster[0]
    sum_vector = cluster[1]
    sum_sq_vector = cluster[2]

    variance_vector = (sum_sq_vector/N) - (sum_vector/N)**2
    centroid_vector = sum_vector/N

    s = 0
    for dim in range(len(point)):
        s += ((point[dim] - centroid_vector[dim])/math.sqrt(variance_vector[dim]))**2

    return math.sqrt(s)

if __name__ == '__main__':
    input_filepath = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_filepath = sys.argv[3]

    with open(input_filepath, 'r') as f:
        for l in f:
            line = l.strip().split(',')
            dim = []
            index = int(line[0])
            cluster_index = int(line[1])
            for i in range(2, len(line)):
                dim.append(float(line[i]))

            indexes.append(index)
            cluster_indexes.append(cluster_index)
            points.append(dim)


    n_clusters = n_cluster

    random.shuffle(indexes)
    step = int(len(indexes)/5)

    DS = dict()
    DS_members = dict()

    RS = []

    CS = dict()
    CS_members = []
    DS_sum = 0

    discarded_dict = dict()

    total_count = 0
    l = np.array_split(indexes, 5)

    intermediate_results = []
    for i, chunk in enumerate(l):

        twenty_percent_data = list(chunk)
        X = []
        for j in twenty_percent_data:
            X.append(points[j])

        DS_COUNT = 0
        RS_COUNT = 0
        CS_COUNT = 0

        if i >= 1:

            for index in twenty_percent_data:
                min_distance = math.inf
                closest = None
                for key, value in DS_members.items():
                    discarded_set = DS[key]
                    m_distance = calculate_mahalanobis_distance(points[index], discarded_set)
                    if m_distance <= min_distance and m_distance < 2 * math.sqrt(len(points[index])):
                        min_distance = m_distance
                        closest = key
                if closest != None:
                    DS_COUNT += 1
                    DS_members[closest].append(index)
                    DS_sum += 1
                    DS[closest][0] += 1
                    DS[closest][1] += np.array(points[index])
                    DS[closest][2] += np.square(points[index])
                    discarded_dict[closest].append(index)
                else:
                    min_distance = math.inf
                    closest = None
                    for z, cluster in enumerate(CS_members):
                        compression_set = CS[z]
                        m_distance = calculate_mahalanobis_distance(points[index], compression_set)
                        if m_distance <= min_distance and m_distance < 2 * math.sqrt(len(points[index])):
                            min_distance = m_distance
                            closest = z
                    if closest != None:
                        CS_COUNT += 1
                        CS_members[closest].append(index)
                        CS[closest][0] += 1
                        CS[closest][1] += np.array(points[index])
                        CS[closest][2] += np.square(points[index])
                    else:
                        RS_COUNT += 1
                        RS.append(index)

            RS_items = []

            for item in RS:
                RS_items.append(points[item])
    
            if len(RS_items) > 1:
                kmeans = KMeans(n_clusters=len(RS_items)).fit(RS_items)

                cluster_members = dict()
                for count, cluster_index in enumerate(kmeans.labels_):
                    if cluster_index not in cluster_members:
                        cluster_members[cluster_index] = []
                        cluster_members[cluster_index].append(RS[count])
                    else:
                        cluster_members[cluster_index].append(RS[count])

                RS = []
                
                for key, value in cluster_members.items():
                    temp_cs = []
                    if len(value) > 1:
                        for val in value:
                            temp_cs.append(val)
                        CS_members.append(temp_cs)
                    elif len(value) == 1:
                        RS.append(value[0])

                
                # Generate statistics for CS set
                for z, cluster in enumerate(CS_members):
                    N = len(cluster)
                    temp_points = []
                    for point in cluster:
                        temp_points.append(points[point])

                    sum_vector = list(map(sum, zip(*temp_points)))
                    squared_list = []
                    for temp_point in temp_points:
                        squared_list.append(list(map(lambda x: x * x, temp_point)))
                    sum_sq_vector = list(map(sum, zip(*squared_list)))

                    CS[z] = [np.array(N), np.array(sum_vector), np.array(sum_sq_vector)]
                # for key, value in CS_members.items():
                #     temp_points = []
                #     N = len(value)
                #     for val in value:
                #         temp_points.append(points[val])
                #     sum_vector = list(map(sum, zip(*temp_points)))
                #     squared_list = []
                #     for temp_point in temp_points:
                #         squared_list.append(list(map(lambda x: x * x, temp_point)))
                #     sum_sq_vector = list(map(sum, zip(*squared_list)))
                #
                #     CS[key] = [np.array(N), np.array(sum_vector), np.array(sum_sq_vector)]
        else:

            kmeans = KMeans(n_clusters= 100).fit(X)

            cluster_members = dict()
            for count, cluster_index in enumerate(kmeans.labels_):
                if cluster_index not in cluster_members:
                    cluster_members[cluster_index] = []
                    cluster_members[cluster_index].append(twenty_percent_data[count])
                else:
                    cluster_members[cluster_index].append(twenty_percent_data[count])

            for key, value in cluster_members.items():
                if len(value) == 1:
                    RS.append(value[0])
                    twenty_percent_data.remove(value[0])

            X = []
            for j in twenty_percent_data:
                X.append(points[j])
            kmeans = KMeans(n_clusters=n_clusters).fit(X)

            for count, cluster_index in enumerate(kmeans.labels_):
                if cluster_index not in DS_members:
                    DS_members[cluster_index] = []
                    DS_members[cluster_index].append(twenty_percent_data[count])
                else:
                    DS_members[cluster_index].append(twenty_percent_data[count])

            for key, value in DS_members.items():
                temp_points = []
                N = len(value)
                for val in value:
                    temp_points.append(points[val])
                sum_vector = list(map(sum, zip(*temp_points)))
                squared_list = []
                for temp_point in temp_points:
                    squared_list.append(list(map(lambda x: x*x, temp_point)))
                sum_sq_vector = list(map(sum, zip(*squared_list)))
                DS_sum += N
                DS[key] = [np.array(N), np.array(sum_vector), np.array(sum_sq_vector)]

                discarded_dict[key] = value
            X = []
            for k in RS:
                X.append(points[k])
            kmeans = KMeans(n_clusters=len(X)).fit(X)

            cluster_members = dict()
            for count, cluster_index in enumerate(kmeans.labels_):
                if cluster_index not in cluster_members:
                    cluster_members[cluster_index] = []
                    cluster_members[cluster_index].append(RS[count])
                else:
                    cluster_members[cluster_index].append(RS[count])

            RS = []
            
            for key, value in cluster_members.items():
                temp_cs = []
                if len(value) > 1:
                    for val in value:
                        temp_cs.append(val)
                    CS_members.append(temp_cs)
                elif len(value) == 1:
                    RS.append(value[0])

            

            # Generate statistics for CS set
            for z,cluster in enumerate(CS_members):
                N = len(cluster)
                temp_points = []
                for point in cluster:
                    temp_points.append(points[point])

                sum_vector = list(map(sum, zip(*temp_points)))
                squared_list = []
                for temp_point in temp_points:
                    squared_list.append(list(map(lambda x: x * x, temp_point)))
                sum_sq_vector = list(map(sum, zip(*squared_list)))

                CS[z] =  [np.array(N), np.array(sum_vector), np.array(sum_sq_vector)]

            # for key, value in CS_members.items():
            #     temp_points = []
            #     N = len(value)
            #     for val in value:
            #         temp_points.append(points[val])
            #     sum_vector = list(map(sum, zip(*temp_points)))
            #     squared_list = []
            #     for temp_point in temp_points:
            #         squared_list.append(list(map(lambda x: x*x, temp_point)))
            #     sum_sq_vector = list(map(sum, zip(*squared_list)))
            #
            #     CS[key] = [np.array(N), np.array(sum_vector), np.array(sum_sq_vector)]

        sum_CS = 0
        for cluster in CS_members:
            sum_CS += len(cluster)


        intermediate_results.append([i, DS_sum, len(CS_members), sum_CS, len(RS)])
    final_dict = dict()
    for key, value in discarded_dict.items():
        for val in value:
            final_dict[val] = key

    for item in RS:
        final_dict[item] = -1

    for cluster in CS_members:
        for val in cluster:
            final_dict[val] = -1

    sorted_dict = dict(sorted(final_dict.items()))

    with open(output_filepath, "w") as f:
        f.writelines("The intermediate results:")
        f.write("\n")
        for item in intermediate_results:
            f.writelines("Round " + str(item[0]+1) + ": " + str(item[1]) + "," + str(item[2]) + "," + str(item[3]) + "," + str(item[4]))
            f.write("\n")
        f.write("\n")
        f.writelines("The clustering results:")
        f.write("\n")
        for key, value in sorted_dict.items():
            f.writelines(str(key) + "," + str(value))
            f.write("\n")
        
