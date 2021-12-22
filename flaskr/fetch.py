from math import ceil
import functools
#from flask import current_app, g
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash
from pymongo import InsertOne, DeleteOne, ReplaceOne

from .db import get_db
from .auth import login_required

from bson.json_util import dumps

bp = Blueprint('fetch', __name__)


@bp.route("/")
def index():
    return render_template("jobs/index.html")


# @bp.route("/home", methods=('GET', 'POST'))
# @login_required
# def fetch_all():
#     page_num = 1
#     if request.method == 'POST':
#         page_num = int(request.form['page_num'])
#     if request.args.get('search_text'):
#         text = request.args.get('search_text')
#         jobs = index.search(text, search_type='AND')
#     else:
#         db = get_db()
#         jobs = db.jobs_info.find()
#     jobs = list(jobs)
#     jobs = jobs[(page_num-1)*20:page_num*20]

#     jobs_count = db.jobs_info.count()
#     page_count = ceil(jobs_count/20)
#     page = f"{page_num}/{page_count}"
#     return render_template("jobs/home.html", jobs=jobs, page=page)

# @bp.route("/fetch_all")
# def fetch_all():
#     db = get_db()
#     jobs = db.jobs_info.find()
#     data = dumps(list(jobs))
#     return data
