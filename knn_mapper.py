#!usr/bin/env python

import sys
import numpy as np

if len(sys.argv) != 3:
	print("Enter proper specifications for knn mapper")
	exit(0)
	
cluster_id = int(sys.argv[1])
point = np.array(list(map(int, sys.argv[2].split(','))))

def getDistanceFrom(dataPoint):
	return np.linalg.norm(dataPoint - point);

for line in sys.stdin:
	line = line.strip()
	line = line.split()
	if(int(line[0]) != cluster_id):
		continue
	line = line[1]
	dataPoint = np.array(list(map(int, line.split(','))))
	print('{} {}'.format(str(getDistanceFrom(dataPoint[1:-1])), str(dataPoint[-1])))
