import numpy
from sklearn.neighbors import LSHForest

class KeywordsClassifier(object):
    N_CANDIDATES = 70
    N_ESTIMATORS = 30

    def __init__(self, lshf, categories, topN):
        self.lshf = lshf # congigured instance of the LSHForest
        self.categories = categories # list of string - categories to classify
        self.topN = topN # the number of top N tochose

    @classmethod
    def CreateKeywordsClassifier(cls, nlp, categories, topNCount):

        catDic = KeywordsClassifier.GetVerticalsVectorsDict(nlp, categories)
        catArray = numpy.array(list(catDic.values()))
        lshf = LSHForest(n_candidates=cls.N_CANDIDATES, n_estimators=cls.N_ESTIMATORS, n_neighbors=topNCount)
        lshf.fit(catArray)
        return KeywordsClassifier(lshf, list(catDic.keys()), topNCount)

    @staticmethod
    def GetVerticalsVectorsDict(nlp, verticals):
        verticalsDict = {}

        for category in verticals:
            text = nlp(category.replace("&", ""))
            verticalsDict[category] = text.vector

        return verticalsDict


    def ClassifyKeywords(self, keywordsVector):
        result = {}
        distances, indices = self.lshf.kneighbors(keywordsVector.reshape(1, -1))
        curIter = 0

        for curIndex in numpy.nditer(indices):
            category = self.categories[curIndex]
            cosDistance = distances[0][curIter]
            result[category] = 1 - cosDistance
            curIter += 1

        return result