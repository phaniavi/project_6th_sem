import sys
from subprocess import PIPE, Popen
import argparse
import os
import numpy as np

def launch_KMeans(trainFile, centroidFile, k, maxIterations):
	
	count = 0
	with open(trainFile, 'r') as reader:
		with open(centroidFile, 'w') as writer:
			for line in reader.readlines():
				if count >= k:
					break
				point = line.split(',')
				point = '{}\n'.format(','.join(point[1:-1]))
				writer.write(point)
				count += 1
				
	for Iter in range(maxIterations):
		proc1 = Popen('python3 kmeans_mapper.py {}'.format(centroidFile).split(), stdin=open(trainFile, 'r'), stdout=open('temp.data', 'w'))
		proc1.communicate()
		proc = Popen('sort', stdin=open('temp.data', 'r'), stdout=open('temp1.data', 'w'))
		proc.communicate()
		proc2 = Popen('python3 kmeans_reducer.py'.split(), stdin=open('temp1.data', 'r'), stdout=open(centroidFile, 'w'))
		proc2.communicate()
		
def getNearestClusterIndex(dataPoint, centroids):
	distances = [np.linalg.norm(dataPoint - centroid) for centroid in centroids]
	return str(np.where(distances == np.min(distances))[0][0])
		
def update_trainFile(trainFile, updatedFile, centroidFile):

	centroids = []
	with open(centroidFile, 'r') as reader:
		for line in reader.readlines():
			centroids.append(np.array(list(map(int, line.split(',')))))
	
	with open(trainFile, 'r') as reader:
		lines = reader.readlines()
		for idx, line in enumerate(lines):
			line = line.strip()
			dataPoint = np.array(list(map(int, line.split(','))))
			dataPoint = dataPoint[1:-1]
			lines[idx] = '{} {}\n'.format(getNearestClusterIndex(dataPoint, centroids), line)
	
	with open(updatedFile, 'w') as writer:
		for line in lines:
			writer.write(line)
			
def classify_testFile(testFile, updatedFile, centroidFile, k):

	centroids = []
	with open(centroidFile, 'r') as reader:
		for line in reader.readlines():
			centroids.append(np.array(list(map(int, line.split(',')))))

	open('output.data', 'w').close()
	with open(testFile, 'r') as reader:
		for line in reader.readlines():
			line = line.strip()
			point = np.array(list(map(int, line.split(','))))
			point = point[1:]
			cluster_id = getNearestClusterIndex(point, centroids)
			proc1 = Popen('python3 knn_mapper.py {} {}'.format(cluster_id, ','.join(map(str,list(point)))).split(), stdin=open(updatedFile, 'r'), stdout=open('temp.data', 'w'))
			proc1.communicate()
			proc2 = Popen('sort', stdin=open('temp.data', 'r'), stdout=open('temp1.data', 'w'))
			proc2.communicate()
			proc3 = Popen('python3 knn_reducer.py {}'.format(k).split(), stdin=open('temp1.data', 'r'), stdout=open('output.data', 'a'))
			proc3.communicate()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Efficient KNN launcher file')
	
	# optional arguments
	parser.add_argument('-m', '--maxIters', help="Maximum number of iterations for K-Means Clustering", type=int, default=100)
	parser.add_argument('-n', '--num', help='Number of clusters to form in the clustering', type=int, default=15)
	parser.add_argument('-d', '--centroid', help='Centroid File to use in the program', type=str, default='centroids.data')
	
	# required arguments
	requiredNamed = parser.add_argument_group('Required Named arguments')
	requiredNamed.add_argument('-t', '--train', help="Training data file name", required=True,type=str)
	requiredNamed.add_argument('-c', '--test', help="Test data file name", required=True,type=str)
	
	# parse arguments
	args = parser.parse_args()
	
	# retrieve optional arguments
	maxIterations = args.maxIters
	k = args.num
	centroidFile = args.centroid
	    
	# retrieve required arguments
	trainFile = args.train
	testFile = args.test
	
	# run K-Means clustering algorithm
	launch_KMeans(trainFile, centroidFile, k, maxIterations)
	
	# prefix cluster_id to the data points in 'trainFile' and store them in 'updatedFile'
	temp = trainFile.split('.')
	updatedFile = temp[0] + '1.' + temp[1]
	update_trainFile(trainFile, updatedFile, centroidFile)
	
	# predict the class for every sample in 'testFile' using parallel KNN
	classify_testFile(testFile, updatedFile, centroidFile, k)
