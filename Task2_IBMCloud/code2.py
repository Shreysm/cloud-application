from flask import Flask,render_template,request
from werkzeug import secure_filename
import sqlite3 as sql
import pandas as pd
#import ibm_db
from math import sin, cos, sqrt, atan2, radians
import os
import sqlite3
import csv
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


app = Flask(__name__)
folder='./static'
app.config['UPLOAD_FOLDER']=folder
port = int(os.getenv("PORT", 5000))
#con=ibm_db.connect("DATABASE=BLUDB;HOSTNAME=dashdb-txn-sbox-yp-dal09-03.services.dal.bluemix.net;PORT=50000;PROTOCOL=TCPIP;UID=vlv63547;PWD=@Charya18;", "", "")


@app.route('/')
def home():
   return render_template('home.html')


@app.route('/cluster')
def cluster():
   con = sql.connect("database.db")
   cur = con.cursor()
   cur.execute("select latitude,longitude from Earthquake ")    
   #ref:https://mubaris.com/2017/10/01/kmeans-clustering-in-python/
   X = cur.fetchall()
   X = pd.DataFrame(X)
   kmeans = KMeans(n_clusters=3)
   kmeans = kmeans.fit(X)
   labels = kmeans.predict(X)
   centroids = kmeans.cluster_centers_
   fig, ax = plt.subplots()
   ax.scatter(X.iloc[:, 0], X.iloc[:, 1])
   ax.scatter(centroids[:, 0], centroids[:, 1], marker='*')
   fig.savefig('static/img.png')
   return render_template('cluster.html',centroids = centroids)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

