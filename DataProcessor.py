import numpy
from sklearn.neighbors import KDTree
from sklearn.neighbors import LSHForest
from collections import Counter
import csv
import spacy
import codecs

TOP_N_COUNT = 1 # Number of categories to describe the channel

#from scipy import spatial

# ToDo: benchmark (1 - scipy.spatial.distance.cosine(one.vector, three.vector))
def cosine_similarity(vec1, vec2):
    return numpy.dot(vec1, vec2)/(numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))

def GetVerticalsListFromFile():
    verticals = []
    with open('/home/pkonovalov/PycharmProjects/ChannelClassifier/Data/verticals.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            verticals.append(row[0].replace("&", ""))
    return verticals

def GetVerticalsVectorsDict(nlp, verticals):
    verticalsDict = {}

    for category in verticals:
        text = nlp(category.replace("&", ""))
        verticalsDict[category] = text.vector
    return verticalsDict

def Classify(nlp, keywords, categories): #keywords  - string; categories - dict: {name; vector}

    keywordsVec = nlp(keywords).vector
    catArray = numpy.array(list(categories.values()))
    catKeys = list(categories.keys())

    print("Creating LSHForest...")

    lshf = LSHForest(n_candidates=70, n_estimators=30, n_neighbors=TOP_N_COUNT)
    lshf.fit(catArray)
    print("LSHForest was created")

    print("Getting neighbors...")
    distances, indices = lshf.kneighbors(keywordsVec.reshape(1, -1))
    print("Got neighbors.")

    curIter = 0
    for curIndex in numpy.nditer(indices):
        print("Found category: " + str(catKeys[curIndex]))
        distance = distances[0][curIter]
        print("with similarity: " + str(1 - distance))
        curIter += 1


#catArray = numpy.array([ 1.,  2.,  3., 4., 5.]).reshape(-1, 1)
#target = numpy.array([ 3.7]).reshape(-1, 1)

# tree = KDTree(catArray, leaf_size=2)

# catArray = [ [21, 5, -5], [50, 50, 50], [-6, 10, 2]]
# target = [[-30, 40, 60]]
#
# lshf = LSHForest(n_candidates=70, n_estimators=30, n_neighbors=1)
# lshf.fit(catArray)
# print("LSHForest was created")
#
# print("Getting neighbors...")
# distances, indices = lshf.kneighbors(target)
# print("Got neighbors.")
#
# curIter = 0
# for curIndex in numpy.nditer(indices):
#     print("Found category: " + str(curIndex))
#     print("with distance: " + str(distances[curIter]))
#     curIter += 1
#
# ar1 = numpy.array([-30, 40, 60])
# ar2 = numpy.array([-6, 10, 2])
# dist = 1 - cosine_similarity(ar1, ar2)
# print("distN =" + str(dist))
# dist, ind = tree.query(target, k=2)

# curIter = 0
# for curIndex in numpy.nditer(ind):
#     print("Found category: " + str(curIndex))
#     print("with distance: " + str(dist[curIter]))
#     curIter += 1