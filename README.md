# Covid_19_Cord_Dataset_Clustering
Placed here is code to perform text clustering  on Covid 19 CORD-19  Dataset.
Inspired from:https://github.com/MaksimEkin/COVID19-Literature-Clustering.
Here we cluster the abstracts and not the full text and then look at wordclouds based on titles in each cluster to infer cluster topics

Donwload the Covid 19 CORD-19 Dataset here:
https://www.semanticscholar.org/cord19/

Pre-requisites:
1)NLTK: https://github.com/nltk/nltk 
2)Pandas: https://pandas.pydata.org/
3)Matplotlib: https://matplotlib.org/ 
4)wordcloud: https://pypi.org/project/wordcloud/
5)scikit-learn : https://scikit-learn.org/stable/index.html

Methodolgy explained in: CORD-19LiteratureClustering.doc

All wordcloud figues, KMeans distortion plot and clsutered data is attached in the repository

Run this code on the downloaded metadata.csv file to generate  clustered data
