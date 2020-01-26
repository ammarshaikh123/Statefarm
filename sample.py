
from flask import Flask, request, render_template

import pandas as pd

app = Flask(__name__, template_folder='template')
df=[]
@app.route('/')
def my_form():
    print("Inside")
    return render_template('new.html')

@app.route('/new1', methods=['POST','GET'])
def my_form_post_1():

    for key,val in request.form.items():
        df.append(val)

@app.route('/new2', methods=['POST','GET'])
def my_form_post_2():

    for key,val in request.form.items():
        df.append(val)

@app.route('/new3', methods=['POST','GET'])
def my_form_post_3():

    for key,val in request.form.items():
        df.append(val)
@app.route('/new4', methods=['POST','GET'])
def my_form_post_4():

    for key,val in request.form.items():
        df.append(val)
@app.route('/new5', methods=['POST','GET'])
def my_form_post_5():

    for key,val in request.form.items():
        df.append(val)
@app.route('/new6', methods=['POST','GET'])
def my_form_post_6():

    for key,val in request.form.items():
        df.append(val)
@app.route('/new7', methods=['POST','GET'])
def my_form_post_7():

    for key,val in request.form.items():
        df.append(val)
@app.route('/new8', methods=['POST','GET'])
def my_form_post_8():

    for key,val in request.form.items():
        df.append(val)
@app.route('/new9', methods=['POST','GET'])
def my_form_post_9():

    for key,val in request.form.items():
        df.append(val)
        return pd.DataFrame(df)



if(__name__ ==  "__main__"):
    app.run()