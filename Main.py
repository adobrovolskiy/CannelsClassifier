import numpy as np
#from sklearn.neighbors import KDTree
from sklearn.neighbors import LSHForest
from collections import Counter
import csv
import spacy
import DataProcessor as dp
from KeywordsClassifier import KeywordsClassifier
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from Video import Video
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print ('%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0))
        return ret
    return wrap

def Visualise(resultDic):
    ind = np.arange(len(res))  # the x locations for the groups
    width = 0.3  # the width of the bars
    fig = plt.figure(figsize=(15,20))
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, res.values(), width,
                    color='black',
                    error_kw=dict(elinewidth=2, ecolor='red'))

    # axes and labels
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Cossine similaritiy')
    ax.set_title('Cossine similaritiy by category')
    xTickMarks = res.keys()
    ax.set_xticks(ind + width / 2)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    plt.show()

def GetVerticalsListFromFile():
    verticals = []
    with open('/home/pkonovalov/PycharmProjects/ChannelClassifier/Data/verticals.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            verticals.append(row[0].replace("&", ""))
    return verticals

def GetLargestKeywordsDb():

    keywords = []
    print("Loading keywords...")
    with open('/home/pkonovalov/Documents/dump/235227_keywords.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            keywords.append(row[0])


    print("List of keywords was loaded. Len = " + str(len(keywords)))
    print("Joining string..")
    resultStr = ' '.join(keywords)
    print("Strings was joined")
    return resultStr


def GetLargestKeywordsDb2(nlp):
    sumVector = np.zeros(nlp.vocab.vectors_length)

    keywords = []
    print("Loading keywords...")
    with open('/home/pkonovalov/Documents/dump/235227_keywords.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            keywords.append(row[0])

    counterDict = Counter(keywords) #optimization for keywords duplicates
    for word, repCount in counterDict.items():  # summurizing words vectors
        curVect = nlp(word).vector
        sumVector += (curVect * repCount)

    return sumVector

def GetLargestKeywordsDb3(nlp):
    print("Loading keywords...")
    with open('/home/pkonovalov/Documents/dump/235227_keywords.csv', 'rt') as csvfile:
        data = csvfile.read()

    return  nlp(data).vector

def GetVideosFromFile():
    videos = []
    with open('/home/pkonovalov/Documents/test.csv', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        #next(reader, None)  # skip the headers


        for row in reader:

            title = row[0]
            description = row[1]
            channel_id = row[2]
            category = row[3]
            tags = row[4]

            currentVideo = Video(title, description, channel_id, category, tags)
            videos.append(currentVideo)

    return videos


samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
neigh = NearestNeighbors(2, 0.4, algorithm='brute', metric='cosine')
neigh.fit(samples)
res = neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)


videos = GetVideosFromFile()

print("Loading dictinary...")
nlp = spacy.load('en')
print("Dictinary loaded.")


verticals = GetVerticalsListFromFile()
classifier = KeywordsClassifier.CreateKeywordsClassifier( nlp, verticals, 5)


#kwB = GetLargestKeywordsDb()
#keywordsVect = nlp(kwB).vector

keywordsVect = GetLargestKeywordsDb2(nlp)

start = time.clock()
res = classifier.ClassifyKeywords(keywordsVect)
end = time.clock()
elapsed = end - start
print ('Took %0.3f ms', elapsed *1000.0)



for curVideo in videos:
    keywords = curVideo.tags
    keywordsVect = nlp(keywords).vector
    print("Keywords: " + keywords)
    print("title: " + curVideo.title)
    #print("description: " + curVideo.description)
    print("channelId: " + curVideo.channelId)
    print("category: " + curVideo.category)

    res = classifier.ClassifyKeywords(keywordsVect)
    g = 5
    #Visualise(res)

#verticals = { 'fruits', 'cars', 'bikes', 'animals'}
# keywords = """ car"""





# curI = 0
# for x in res.values():
#     ax.annotate(str(res[curI]), xy=(x, res[curI]))
#     curI+=1



#verticals = GetVerticalsListFromFile()
#
# verticals = { 'fruits', 'cars', 'bikes', 'animals'}
# verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
# keywords = 'BMW'
# print("Classifiing keywords: " + keywords)
# dp.Classify(nlp, keywords, verticalsDict)
#
#
# verticals = { 'fruits', 'cars', 'bikes', 'animals'}
# verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
# keywords = 'BMW NISSAN TOYOTA'
# print("Classifiing keywords: " + keywords)
# dp.Classify(nlp, keywords, verticalsDict)
#
#
# verticals = { 'fruits', 'cars', 'bikes', 'animals'}
# verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
# keywords = 'apple plum banana'
# print("Classifiing keywords: " + keywords)
# dp.Classify(nlp, keywords, verticalsDict)
#
# verticals = { 'fruits', 'cars', 'bikes', 'animals'}
# verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
# keywords = 'Bunny Bear Wolf'
# print("Classifiing keywords: " + keywords)
# dp.Classify(nlp, keywords, verticalsDict)
#
#
# verticals = { 'fruits', 'cars', 'bikes', 'animals'}
# verticalsDict = dp.GetVerticalsVectorsDict(nlp, verticals)
# keywords = 'Bunny Bear Wolf BMW NISSAN'
# print("Classifiing keywords: " + keywords)
# dp.Classify(nlp, keywords, verticalsDict)

breakS1 = 1
breakS1 = 1
breakS1 = 1
breakS1 = 1