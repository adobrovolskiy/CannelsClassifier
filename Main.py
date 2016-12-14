import numpy
#from sklearn.neighbors import KDTree
from sklearn.neighbors import LSHForest
from collections import Counter
import csv
import spacy
import DataProcessor as dp


print("Loading dictinary...")
nlp = spacy.load('en')
print("Dictinary loaded.")
#verticals = GetVerticalsListFromFile()

verticals = { 'fruits', 'cars', 'bikes', 'animals'}
verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
keywords = 'BMW'
print("Classifiing keywords: " + keywords)
dp.Classify(nlp, keywords, verticalsDict)


verticals = { 'fruits', 'cars', 'bikes', 'animals'}
verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
keywords = 'BMW NISSAN TOYOTA'
print("Classifiing keywords: " + keywords)
dp.Classify(nlp, keywords, verticalsDict)


verticals = { 'fruits', 'cars', 'bikes', 'animals'}
verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
keywords = 'apple plum banana'
print("Classifiing keywords: " + keywords)
dp.Classify(nlp, keywords, verticalsDict)

verticals = { 'fruits', 'cars', 'bikes', 'animals'}
verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
keywords = 'Bunny Bear Wolf'
print("Classifiing keywords: " + keywords)
dp.Classify(nlp, keywords, verticalsDict)


verticals = { 'fruits', 'cars', 'bikes', 'animals'}
verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
keywords = 'Bunny Bear Wolf BMW NISSAN'
print("Classifiing keywords: " + keywords)
dp.Classify(nlp, keywords, verticalsDict)

breakS1 = 1
breakS1 = 1
breakS1 = 1
breakS1 = 1