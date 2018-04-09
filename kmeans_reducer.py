#!usr/bin/env python

import sys
import numpy as np

centroids = {}  	# map of all possible clusters
current_id = None  	# id of the current processing point
current_count = 0  	# current count of points in the same cluster
count = 0  			# Total number of data points
centroid_id = None  # id of the current processing cluster centroid

for line in sys.stdin:
    count += 1
    line = line.strip()
    line = line.split(' ')

    # In each line we have cluster id to which it belongs and the corresponding values of the data point
    centroid_id = int(line[0])
    point = np.array(list(map(int, line[1].split(','))))

    if centroid_id not in centroids:  # if it is a new centroid_id then add it the map of centroids
        centroids[centroid_id] = np.zeros(point.shape[0])
    centroids[centroid_id] += point  # Add the values of the points in the same cluster

    if current_id == centroid_id:
        current_count += 1
    else:
        if current_id is not None:
            print(','.join(map(str, map(int, centroids[current_id] / current_count))))
        current_count = 1
        current_id = centroid_id

if current_id == centroid_id:
    print(','.join(map(str, map(int, centroids[current_id] / current_count))))
