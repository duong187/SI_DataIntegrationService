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

import json
import requests
import os
import os.path
import time
from lxml import etree
import gzip
from math import ceil


def load_documents():
    start = time.time()
    db = get_db()
    jobs = db.jobs_info.find()
    jobs = list(jobs)
    print(len(jobs))
    doc_id = 0
    for job in jobs:
        print(doc_id)
        if(job['title']):
            title = job['title']
        else:
            title = " "
        url = job['url']
        address = job['address']
        company = job['company']
        maxSalary = job['salary']
        yield Abstract(ID=doc_id, title=title, url=url, address=address, company=company, maxSalary=maxSalary)
        doc_id += 1

    end = time.time()
    print(f'Parsing XML took {end - start} seconds')

def index_documents(documents, index):
    for i, document in enumerate(documents):
        index.index_document(document)
        # if i % 50 == 0:
        #     print(f'Indexed {i} documents', end='\r')
    return index

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev'
    )
    #app.config["MONGO_URI"] = "mongodb://localhost:27017/data_intergration"
    app.config["MONGO_URI"] = "mongodb+srv://duongnb:18071999@cluster0.9r4fz.mongodb.net/data_intergration?retryWrites=true&w=majority"
    Bootstrap(app)
    mongo = PyMongo(app)
    app.db = mongo.db

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Register blueprint
    from . import auth
    app.register_blueprint(auth.bp)
    from . import fetch
    app.register_blueprint(fetch.bp)
    from . import integrate
    app.register_blueprint(integrate.bp)
    
    app.add_url_rule('/', endpoint='index')
    app.add_url_rule('/home', endpoint='home')

    with app.app_context():
        index = index_documents(load_documents(), Index())
        print(f'Index contains {len(index.documents)} documents')

    # @app.route('/home', methods=('GET', 'POST'))
    # def home():
    #     if request.method == 'GET':
    #         text = request.args.get('search_text')
    #         jobs = index.search(text, search_type='AND')
    #         return render_template("jobs/home.html", jobs=jobs, page=1)

    @app.route("/home", methods=('GET', 'POST'))
    @login_required
    def home():
        global db
        page_num = 1
        db = get_db()
        text_s = ""
        jobs = db.jobs_info.find()
        if request.args.get('page_num'):
            page_num = int(request.args.get('page_num'))
        if request.args.get('search_text'):
            text = request.args.get('search_text')
            text_s = request.args.get('search_text')
            if(text != None):
                jobs = index.search(text, search_type='AND')

        jobs = list(jobs)
        jobs = jobs[(page_num-1)*20:page_num*20]
        jobs_count = db.jobs_info.count()
        page_count = ceil(jobs_count/20)
        page = f"{page_num}/{page_count}"
        return render_template("jobs/home.html", jobs=jobs, page=page, jobs_count=jobs_count, text_s=text_s, page_count=page_count)
    return app