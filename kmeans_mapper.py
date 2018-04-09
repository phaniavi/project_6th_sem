#!usr/bin/env python

import sys
import numpy as np

if len(sys.argv) != 2:
    print("Enter centroid file name")
    exit(0)

centroid_file = sys.argv[1]
centroids = []

with open(centroid_file, 'r') as reader:
	for line in reader.readlines():
		point = list(map(int, line.split(',')))
		centroids.append(np.array(point))  # Store all possible centroids stored in the centroid file


def getNearestClusterIndex(data_point):
    distances = [np.linalg.norm(data_point - centroid) for centroid in centroids]
    return np.where(distances == np.min(distances))[0][0]


for line in sys.stdin:
    line = line.strip()
    point = list(map(int, line.split(',')))
    print('{} {}'.format(str(getNearestClusterIndex(np.array(point[1:-1]))), ",".join(map(str, point[1:-1]))))
