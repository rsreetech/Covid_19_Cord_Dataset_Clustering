# -*- coding: utf-8 -*-
"""
Created on Sun May 24 14:37:04 2020

@author: Rithesh
"""


from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
import numpy as np
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.cluster import KMeans 
from sklearn import metrics 
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt  


cord19Df = pd.read_csv('C:\\NLPExperiments\\metadata.csv')

#Remove rows whose 'abstract' column in empty
cord19FilDf = cord19Df[ cord19Df.abstract.isna() != True ]

def clean_text(text ): 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text1 = ''.join([w for w in text if not w.isdigit()]) 
    return text1.lower()


stop_words = set(stopwords.words('english')) 
stop_words.add('abstract')
stop_words.add('doi')
stop_words.add('biorxiv')
stop_words.add('medrxiv')

def remove_Stopwords(text ):
    words = word_tokenize( text.lower() ) 
    sentence = [w for w in words if not w in stop_words]
    return " ".join(sentence)
    

def lemmatize_text(text):
    wordlist=[]
    lemmatizer = WordNetLemmatizer() 
    sentences=sent_tokenize(text)
    for sentence in sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    return ' '.join(wordlist) 


#Preprcess abstract column
cord19FilDf['abstract'] = cord19FilDf['abstract'].apply(clean_text)
cord19FilDf['abstract'] = cord19FilDf['abstract'].apply(remove_Stopwords)
cord19FilDf['abstract'] = cord19FilDf['abstract'].apply(lemmatize_text)

#Created tf-idf features for abstract column.Select only top 500 features
absList = cord19FilDf['abstract'].tolist()
vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',max_df=0.50,use_idf=True,smooth_idf=True,max_features=500)
tfIdfMat  = vectorizer.fit_transform(absList )
feature_names = sorted(vectorizer.get_feature_names())



distortions = [] 
inertias = [] 
scores=[]
K = range(2,40) 
scores=[]
#Compute PCA features on tf-idf matrix
tfIdfMatrix = tfIdfMat.todense()
pca = PCA(n_components=0.95)
tfIdfMat_reduced= pca.fit_transform(tfIdfMat.toarray())
tfIdfMat_reduced.shape

#Select 20000 randoom papers
idx = np.random.randint(105564,size=20000)

tfIdfKmeans = tfIdfMat_reduced[idx,:]

#Perform Kmeans clustering using different values of k
for k in K: 
    
    kmModel =  KMeans(n_clusters=k,max_iter=50).fit(tfIdfKmeans )
    preds = kmModel.fit_predict(tfIdfKmeans)
    centers = kmModel.cluster_centers_
    distortions.append(sum(np.min(cdist(tfIdfKmeans, centers,'euclidean'),axis=1)) / tfIdfKmeans .shape[0]) 
    
    

plt.figure( figsize=(20,10))  
plt.plot(K,distortions, 'gx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortions') 
plt.title('The Elbow Method using distortions') 
plt.show() 
plt.savefig('KmeansElbow.png')
    
#Select K=25 and perform Kmeans on reduced PCA matrix
kmeanModel = KMeans(n_clusters=25).fit(tfIdfKmeans ) 

#Predict clusters on entire PA matrix
preds = kmeanModel.fit_predict(tfIdfMat_reduced)

cord19FilDf['cluster'] =preds

cord19FilDf.to_csv("ClusteredCord19Data.csv")

# Group papers by cluster ID. For each cluster create wordcloud using titles of papers in cluster
cord19TitleFilDf = cord19FilDf[ cord19Df.title.isna() != True ]
cord19Clusters = cord19TitleFilDf.groupby('cluster')
globalClusterTitles = []
k=1
for name,cluster in cord19Clusters:
    
    globalClusterTitles.append(cluster['title'].tolist())
    wordcloud = WordCloud(background_color="white",width=1600, height=800).generate(' '.join(cluster['title'].tolist()))
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    #save the image
    plt.savefig('ClusterNo'+str(k)+'wordcloud.png', facecolor='k', bbox_inches='tight')
    k = k+1



    







