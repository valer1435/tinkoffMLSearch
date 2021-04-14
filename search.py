import random
import pandas as pd
import re
from functools import reduce
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


stemmer = SnowballStemmer(language='english')


#https://www.kaggle.com/neisse/scrapped-lyrics-from-6-genres?select=lyrics-data.csv
class Document:
    def __init__(self, author, title, text, rating):
        # можете здесь какие-нибудь свои поля подобавлять
        self.author = author
        self.title = title
        self.text = text
        self.rating = rating
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title + f"({self.author})", self.text + ' ...']

index = {}
inverse_index = {}
file_name = "data.csv"
author_column = "ALink"
title_column = "SName"
lyric_column = "Lyric"
rating_column = "rating"
def build_index():
    df =pd.read_csv(file_name)[[author_column, title_column, lyric_column, rating_column]]
    print(len(df))
    df.apply(add_index, axis=1)

def add_index(row):
    for col in [author_column, title_column]:
        lem_words = list(map(stem, re.sub(r'[^A-Za-z0-9А-Яа-я\s-]', '', row[col].lower().replace("-", " ")).split()))
        for word in lem_words:
            if word not in inverse_index:
                inverse_index[word] = []
                inverse_index[word].append(row.name)
            elif row.name not in inverse_index[word]:
                inverse_index[word].append(row.name)
    index[row.name] = Document(
        row[author_column].replace("-", " "),
        row[title_column],
        row[lyric_column],
        row[rating_column]
        )

def stem(x):
    return stemmer.stem(x)
            
def count_all(query):
    return CountVectorizer().fit([query])
    
def evaluate_performance(candidates):
    if len(candidates) == 0:
        return 0
    DCG = 0
    iDCG = 0
    for i in range(len(candidates)):
        c = candidates[i]
        DCG += c.rating/np.log(i+1+1)

    ideal_cand = sorted(candidates, key=lambda x: -x.rating)
    for i in range(len(ideal_cand)):
        c = ideal_cand[i]
        iDCG += c.rating/np.log(i+1+1)

    return DCG/iDCG
    
def score(query, document, vectorizer):
    author_k = 2
    title_k = 2
    lyric_k = 0.05
    rat_k = 0.5

    score = 0
    score += author_k * np.sum(vectorizer.transform([document.author]).todense())
    score += title_k * np.sum(vectorizer.transform([document.title]).todense())
    score += lyric_k * np.sum(vectorizer.transform([document.text]).todense())
    score += rat_k * document.rating
    return score

def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    if query=="":
        return []
    candidates = []
    idx_sets = []
    lem_words =  list(map(lambda x: stemmer.stem(x), re.sub(r'[^A-Za-z0-9А-Яа-я\s-]', '', query.lower().replace("-", " ")).split()))
    for word in lem_words:
        if word not in inverse_index:
            continue
        idx_sets.append(set(inverse_index[word]))
        candidates = candidates + inverse_index[word]
    if not idx_sets:
        return []
    set_all = list(reduce(lambda x,y: x.intersection(y), idx_sets))
    candidates = candidates + set_all
    
    l = list(frozenset(reversed(candidates[-50:])))
    
    return list(map(lambda x: index[x], l))
