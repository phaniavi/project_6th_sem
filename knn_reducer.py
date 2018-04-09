#!usr/bin/env python

import sys
import numpy as np

if len(sys.argv) != 2:
	print('Enter proper arguments for knn_reducer')
	exit(0)
	
k = int(sys.argv[1])
distances = []
predicted_class = 0.0

for line in sys.stdin:
	line = line.strip()
	distances.append(np.array(list(map(float, line.split()))))
	
distances = sorted(distances, key=lambda x:x[0])

for i in range(min(len(distances),k)):
	predicted_class += distances[i][1]/2-1
	
print(float(predicted_class/k))
