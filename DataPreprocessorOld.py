import numpy
#from sklearn.neighbors import KDTree
from sklearn.neighbors import LSHForest
from collections import Counter
import csv
import spacy
import codecs

TOP_N_COUNT = 3 # Number of categories to describe the channel

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

    sumVector = numpy.zeros(nlp.vocab.vectors_length)

    for category in verticals:
        words = nlp(category.replace("&", ""))
        for word in words:
            sumVector += word.vector
        verticalsDict[category] = sumVector
    return verticalsDict

def Classify(nlp, keywords, categories): #keywords  - list; categories - dict: {name; vector}
    counterDict = Counter(keywords) #optimization for keywords duplicates
    sumVector = numpy.zeros(nlp.vocab.vectors_length)

#temp
    text =  ' '.join(keywords)

    for word, repCount in counterDict.items(): #summurizing words vectors
        curVect = nlp(word).vector
        sumVector += (curVect * repCount)

    vec = nlp(text).vector
    sim = cosine_similarity(vec, sumVector)
    print("Sim: " + str(sim))

    catArray = numpy.array(list(categories.values()))
    catKeys = list(categories.keys())
    #tree = KDTree(catArray, metric='pyfunc', func=cosine_similarity)
    #dist, ind = tree.query(sumVector, k=TOP_N_COUNT) #.reshape(-1, 1)

    print("Creating LSHForest...")

    lshf = LSHForest(n_candidates=70, n_estimators=30, n_neighbors=TOP_N_COUNT)
    lshf.fit(catArray)
    print("LSHForest was created")

    print("Getting neighbors...")
    distances, indices = lshf.kneighbors(sumVector.reshape((1, -1)))
    print("Got neighbors.")

    for curIndex in numpy.nditer(indices):
        print("Found category: " + str(catKeys[curIndex]))
        print("with distance: " + str(distances))

#
# def Process(nlp):
#     words = nlp(u'BMW Mercedes Toyota Lexus Ford window number')
#     car = nlp(u'car')
#     motorcycle = nlp(u'motorcycle')
#
#     shape = nlp.vocab.vectors_length
#     sumVector = numpy.zeros(shape)
#
#     for word in words:
#         sumVector += word.vector
#
#     sim1 = cosine_similarity(sumVector, car.vector)
#     sim2 = cosine_similarity(sumVector, motorcycle.vector)
#
#     print(sim1)
#     print(sim2)


#X = numpy.array([[ 1.,  2.], [ 10.,  20]])

#tree = KDTree(X, leaf_size=2, metric='pyfunc', func=cosine_similarity)

#target = numpy.array([ 3.7]).reshape(-1, 1)
#target = numpy.array( [ 9.,  11.])

#dist, ind = tree.query(target, k=1)
#print (dist)  # distances to 3 closest neighbors
#print (ind)










#X = numpy.array([ 1.,  2.,  3., 4., 5.]).reshape(-1, 1)

#X1 = numpy.random.random((4, 1))
# tree = KDTree(X, leaf_size=2, metric='pyfunc', func=cosine_similarity)
#
# #target = numpy.array([ 3.7]).reshape(-1, 1)
#
# dist, ind = tree.query(target, k=1)
# print (dist)  # distances to 3 closest neighbors
# print (ind)
# Load English tokenizer, tagger, parser, NER and word vectors
#nlp = spacy.load('en')
#Process(nlp)

# s1 = one.similarity(three)
# s1_test = 1 - spatial.distance.cosine(one.vector, three.vector)
# s1_test2 = cosine_similarity(numpy.float64(one.vector), numpy.float64(three.vector))


# s2 = two.similarity(three)
#
# sum1 = (s1 + s2)/2
#
# vecSum = one.vector + two.vector
# sum2 = 1 - spatial.distance.cosine(three.vector, vecSum)
#
#
#
# print("s1 = " + str(s1))
# print("s1_test = " + str(s1_test))
# print("s1_test2 = " + str(s1_test2))
#
# print("s2 = " + str(s2))
#
#
# print("Sum1 = " + str(sum1))
# print("Sum2 = " + str(sum2))


#sum2 = car.similarity()




#print(vecSum)
#print(oranges.similarity(oranges))


