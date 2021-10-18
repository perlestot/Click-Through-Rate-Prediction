# -*- coding: utf-8 -*-

# dev_func.py

#!/usr/bin/env python3 

# Basic library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm


# Tuning and Develop Model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import scipy as sp
from sklearn.externals import joblib

# Set parameter for tuning
score_param = 'balanced_accuracy' # Score for tune model
n_iter_search = 2 # Max candidate parameter for RandomizedSearchCV
cv = 2 # Number of k-fold cross validation


# MultinomialNB
def MNB_HPTune(X, y, verbose = 0):
    tfidf = TfidfVectorizer(min_df=2)
    lda = LatentDirichletAllocation(random_state=7)
    clf = MultinomialNB()
    pipe = Pipeline([('TFIDF', tfidf),
                     ('LDA', lda),
                     ('MNB', clf)])
    
    # Define Search Param
    param_dist = dict(TFIDF__ngram_range = ((1,2), (1,3)),
                      LDA__n_components = np.arange(10,200,10),
                      MNB__alpha= np.append(np.logspace(-3,2,num=20),0)
                     )
    rs = RandomizedSearchCV(estimator=pipe,
                            param_distributions=param_dist,
                            refit=True,
                            scoring=score_param,
                            n_iter=n_iter_search,
                            cv=cv,
                            n_jobs=-1,
                            random_state=7,
                            iid=True)
    if verbose == 1:
        start = time()    
        rs.fit(X,y)
        print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." 
              % (time() - start, len(rs.cv_results_['params'])))

    elif verbose == 0:
        rs.fit(X,y)
    
    # Best parameter from RandomizedSearchCV
    
    bs_var_ncom = rs.best_params_['LDA__n_components']
    bs_var_alpha = rs.best_params_['MNB__alpha']
    bs_var_alpha = np.log10(bs_var_alpha)
    
    param_grid = dict(TFIDF__ngram_range = ((1,2), (1,3)),
                      LDA__n_components = np.arange(bs_var_ncom-20,bs_var_ncom+20,5),
                      MNB__alpha= np.append(np.logspace(bs_var_alpha-1,bs_var_alpha+1,num=5),0)
                     )
    gs = GridSearchCV(estimator=pipe, 
                      param_grid=param_grid,
                      refit=True,
                      scoring=score_param,
                      cv=cv,
                      n_jobs=-1, 
                      iid=True)
    
    if verbose == 1:
        start = time()
        gs.fit(X,y)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(gs.cv_results_['params'])))

    elif verbose == 0:
        gs.fit(X,y)
    
    return rs, gs

# Decision Tree
def DT_HPTune(X, y, verbose = 0):
    tfidf = TfidfVectorizer(min_df=2)
    lda = LatentDirichletAllocation(random_state=7)
    clf = DecisionTreeClassifier(random_state=7)
    pipe = Pipeline([('TFIDF', tfidf),
                     ('LDA', lda),
                     ('DT', clf)])
    # Define Search Param
    param_dist = dict(TFIDF__ngram_range = ((1,2), (1,3)),
                      LDA__n_components = np.arange(40,200,10),
                      DT__criterion = ['gini', 'entropy'],
                      DT__max_depth = np.arange(6,20,1)
                     )
    rs = RandomizedSearchCV(estimator=pipe,
                            param_distributions=param_dist,
                            refit=True,
                            scoring=score_param,
                            n_iter=n_iter_search,
                            cv=cv,
                            n_jobs=-1,
                            random_state=7,
                            iid=True)
    if verbose == 1:
        start = time()    
        rs.fit(X,y)
        print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." 
              % (time() - start, len(rs.cv_results_['params'])))

    elif verbose == 0:
        rs.fit(X,y)
    
    # Best parameter from RandomizedSearchCV
    bs_var_ncom = rs.best_params_['LDA__n_components']
    bs_var_cri = rs.best_params_['DT__criterion']
    bs_var_maxd = rs.best_params_['DT__max_depth']
    
    param_grid = dict(TFIDF__ngram_range = ((1,2), (1,3)),
                        LDA__n_components = np.arange(bs_var_ncom-10,bs_var_ncom+10,5),
                        DT__criterion = [bs_var_cri],  
                        DT__max_depth = np.arange(bs_var_maxd-3,bs_var_maxd+3,1)
                     )
    gs = GridSearchCV(estimator=pipe, 
                      param_grid=param_grid,
                      refit=True,
                      scoring=score_param,
                      cv=cv,
                      n_jobs=-1, 
                      iid=True)
    
    if verbose == 1:
        start = time()
        gs.fit(X,y)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(gs.cv_results_['params'])))

    elif verbose == 0:
        gs.fit(X,y)
    
    return rs, gs

# Support Vector Machine
def SVM_HPTune(X, y, verbose = 0):
    tfidf = TfidfVectorizer(min_df=2)
    lda = LatentDirichletAllocation(random_state=7)
    clf = SVC(probability=True)
    pipe = Pipeline([('TFIDF', tfidf),
                     ('LDA', lda),
                     ('SVM', clf)])
    
    # Define Search Param
    param_dist = dict(TFIDF__ngram_range = ((1,2), (1,3)),
                      LDA__n_components = np.arange(10,200,10),
                      SVM__C = np.logspace(-10,5,num=30),
                      SVM__gamma = np.logspace(-10,5,num=30),
                      SVM__kernel = ['sigmoid', 'rbf','linear','poly'],
                      #SVM__decision_function_shape = ('ovo','ovr')
                     )
    rs = RandomizedSearchCV(estimator=pipe,
                            param_distributions=param_dist,
                            refit=True,
                            scoring=score_param,
                            n_iter=n_iter_search,
                            cv=cv,
                            n_jobs=-1,
                            random_state=7,
                            iid=True)
    if verbose == 1:
        start = time()    
        rs.fit(X,y)
        print("RandomizedSearchCV took %.2f seconds for %d candidate parameter settings." 
              % (time() - start, len(rs.cv_results_['params'])))

    elif verbose == 0:
        rs.fit(X,y)
    
    # Best parameter from RandomizedSearchCV
    bs_ngram = rs.best_params_['TFIDF__ngram_range']
    bs_var_ncom = rs.best_params_['LDA__n_components']
    bs_C = rs.best_params_['SVM__C'] 
    bs_gamma = rs.best_params_['SVM__gamma'] 
    bs_kernel = rs.best_params_['SVM__kernel']
    #bs_dfs = rs.best_params_['SVM__decision_function_shape'] 
    d_C = np.log10(bs_C)
    d_gamma = np.log10(bs_gamma)
    
    param_grid = dict(TFIDF__ngram_range = [bs_ngram],
                      LDA__n_components = np.arange(bs_var_ncom-5,bs_var_ncom+5,5),
                      SVM__C = np.logspace(d_C-1,d_C+4,num=5),
                      SVM__gamma = np.logspace(d_gamma-1,d_gamma+1,num=5),
                      SVM__kernel = [bs_kernel,'rbf'],
                      #SVM__decision_function_shape = [bs_dfs]
                     )
    gs = GridSearchCV(estimator=pipe, 
                      param_grid=param_grid,
                      refit=True,
                      scoring=score_param,
                      cv=cv,
                      n_jobs=-1, 
                      iid=True)
    
    if verbose == 1:
        start = time()
        gs.fit(X,y)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(gs.cv_results_['params'])))

    elif verbose == 0:
        gs.fit(X,y)
    
    return rs, gs