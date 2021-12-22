from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from flask_pymongo import PyMongo
from flask_bootstrap import Bootstrap

from .db import get_db
from .search.documents import Abstract
from .search.index import Index
from .auth import login_required
from .fetch import index

from pymongo import MongoClient
from bson.json_util import dumps, loads
import json
import requests
import os
import os.path
import time
from lxml import etree
import gzip
from math import ceil
import pandas as pd
import numpy as np

import py_stringsimjoin as ssj
import py_stringmatching as sm
import py_entitymatching as em

def preprocess_title(title):
	title = title.lower()
	title = title.replace(',', ' ')
	title = title.replace("'", '')    
	title = title.replace('&', 'and')
	title = title.replace('?', '')
	title = title.encode('utf-8', 'ignore')
	return title.strip()



bp = Blueprint('integrate', __name__, url_prefix='/integrate')

@bp.route("/index")
def integrate():
    # itviec_url = ""
    # vnwork_url = ""
    # itviec_data = requests.get(url=itviec_url).json()
    # vnwork_data = requests.get(url=vnwork_url).json()
    #db = get_db()
    #itviec_data = list(db.itviec.find())
    #vnwork_data = list(db.vietnamwork.find())
    # Data integration function
    
    # itviec_df = pd.DataFrame.from_records(itviec_data)
    # vnwork_df = pd.DataFrame.from_records(vnwork_data)

    itviec_df = pd.read_json('flaskr\hanoi_itviec.json')
    vnwork_df = pd.read_json('flaskr\hanoi_vietnamwork.json')
    itviec_df['id'] = range(itviec_df.shape[0])
    vnwork_df['id'] = range(vnwork_df.shape[0])

    itviec_df['mixture'] = itviec_df['title'] + itviec_df['company']
    vnwork_df['mixture'] = vnwork_df['title'] + vnwork_df['company']

    C = ssj.overlap_coefficient_join(itviec_df, vnwork_df, 'id', 'id', 'mixture', 'mixture', sm.WhitespaceTokenizer(), 
								 l_out_attrs=['title', 'company', 'salary'],
								 r_out_attrs=['title', 'company', 'maxSalary'],
								 threshold=0.6)

    em.set_key(itviec_df, 'id')   # specifying the key column in the itviec dataframe
    em.set_key(vnwork_df, 'id')     # specifying the key column in the vietnamwork dataframe
    em.set_key(C, '_id')            # specifying the key in the candidate set
    em.set_ltable(C, itviec_df)   # specifying the left table 
    em.set_rtable(C, vnwork_df)     # specifying the right table
    em.set_fk_rtable(C, 'r_id')     # specifying the column that matches the key in the right table 
    em.set_fk_ltable(C, 'l_id')     # specifying the column that matches the key in the left table

    #C.to_csv('flaskr\sampled.csv', encoding='utf-8')
    labeled = em.read_csv_metadata('flaskr\labeled.csv', ltable=itviec_df, rtable=vnwork_df,
                                fk_ltable='l_id', fk_rtable='r_id', key='_id')

    
    labeled['l_title'] = labeled['l_title'].map(preprocess_title)
    labeled['r_title'] = labeled['r_title'].map(preprocess_title)
    labeled['l_company'] = labeled['l_company'].map(preprocess_title)
    labeled['r_company'] = labeled['r_company'].map(preprocess_title)
    # hn_itviec = em.read_csv_metadata('data_3005/hn_itviec.csv', key='id')
    # hn_vnwork = em.read_csv_metadata('data_3005/hn_vnwork.csv', key='id')

    split = em.split_train_test(labeled, train_proportion=0.6, random_state=0)
    train_data = split['train']
    test_data = split['test']

    dt = em.DTMatcher(name='DecisionTree', random_state=0)
    svm = em.SVMMatcher(name='SVM', random_state=0)
    rf = em.RFMatcher(name='RF', random_state=0)
    lg = em.LogRegMatcher(name='LogReg', random_state=0)
    ln = em.LinRegMatcher(name='LinReg')
    nb = em.NBMatcher(name='NaiveBayes')

    attr_corres = em.get_attr_corres(itviec_df, vnwork_df)
    # attr_corres['corres'] = [('title', 'title'),
    # 						('company', 'company')]

    l_attr_types = em.get_attr_types(itviec_df)
    r_attr_types = em.get_attr_types(vnwork_df)

    tok = em.get_tokenizers_for_matching()
    sim = em.get_sim_funs_for_matching()

    F = em.get_features(itviec_df, vnwork_df, l_attr_types, r_attr_types, attr_corres, tok, sim)

    train_features = em.extract_feature_vecs(train_data, feature_table=F, attrs_after='label', show_progress=False) 
    train_features = em.impute_table(train_features,  exclude_attrs=['_id', 'l_id', 'r_id', 'label'], strategy='mean', missing_val = np.nan)

    result = em.select_matcher([dt, rf, svm, ln, lg, nb], table=train_features, 
                            exclude_attrs=['_id', 'l_id', 'r_id', 'label'], k=5,
                            target_attr='label', metric_to_select_matcher='f1', random_state=0)
    result['cv_stats']

    best_model = result['selected_matcher']
    best_model.fit(table=train_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], target_attr='label')

    test_features = em.extract_feature_vecs(test_data, feature_table=F, attrs_after='label', show_progress=False)
    test_features = em.impute_table(test_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], strategy='mean', missing_val = np.nan)

    # Predict on the test data
    predictions = best_model.predict(table=test_features, exclude_attrs=['_id', 'l_id', 'r_id', 'label'], 
                                    append=True, target_attr='predicted', inplace=False)

    # Evaluate the predictions
    eval_result = em.eval_matches(predictions, 'label', 'predicted')
    em.print_eval_summary(eval_result)

    candset_features = em.extract_feature_vecs(C, feature_table=F, show_progress=True)
    candset_features = em.impute_table(candset_features, exclude_attrs=['_id', 'l_id', 'r_id'], strategy='mean', missing_val = np.nan)
    predictions = best_model.predict(table=candset_features, exclude_attrs=['_id', 'l_id', 'r_id'],
                                    append=True, target_attr='predicted', inplace=False)
    matches = predictions[predictions.predicted == 1]

    from py_entitymatching.catalog import catalog_manager as cm
    matches = matches[['_id', 'l_id', 'r_id', 'predicted']]
    matches.reset_index(drop=True, inplace=True)
    cm.set_candset_properties(matches, '_id', 'l_id', 'r_id', itviec_df, vnwork_df)
    matches = em.add_output_attributes(matches, l_output_attrs=['title', 'salary', 'company', 'address'],
                                    r_output_attrs=['title', 'maxSalary', 'company', 'address'],
                                    l_output_prefix='l_', r_output_prefix='r_',
                                    delete_from_catalog=False)
    matches.drop('predicted', axis=1, inplace=True)
    matches = matches.to_dict('records')
    return render_template("jobs/integrate.html", duplicates=matches)