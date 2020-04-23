#Author: Shreyas Mohan

from flask import Flask,render_template,request
from werkzeug import secure_filename
import sqlite3 as sql
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import os
import numpy as np
import csv
import time
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib as mp
mp.use('Agg')
import matplotlib.pyplot as plotter

application = Flask(__name__)
scaler = StandardScaler()
onehotencoder = OneHotEncoder(categorical_features = "all")
le = LabelEncoder()

folder='./static'
application.config['UPLOAD_FOLDER']=folder
port = int(os.getenv("PORT", 5000))


@application.route('/')
def home():
   return render_template('home.html')


@application.route('/enternew')
def upload_csv():
   return render_template('upload.html')

@application.route('/upload_file',methods = ['POST', 'GET'])
def upload_file():
  if request.method == 'POST':
       con = sql.connect("database.db")
       csv = request.files['myfile']
       filename=secure_filename(csv.filename)
       file = pd.read_csv(csv)
       file.to_sql('Earthquake', con, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)   
       con.close()

  return render_template("file_upload.html",msg = filename)


@application.route('/records')      
def records():
  con = sql.connect("database.db")
   
  cur = con.cursor()
  cur.execute('SELECT * from Earthquake')
  rows = cur.fetchall()
  cur.execute("select count(*) from Earthquake ")
  count = cur.fetchall()
  '''
  count=0
  minval=100
  for row in rows:
    count=count+1
    mag=row[5]
    place=row[7]
    if mag is not None and mag<minval and mag>2:
        minval=mag
        mine=row
      
  '''
  con.close()
  return render_template("displayrec.html",data=rows,count=count)

@application.route('/range1')
def range1():
   return render_template('Range1.html')

@application.route('/values1',methods = ['POST', 'GET'])
def values1():
  #Connecting to database
  conn = sql.connect('database.db')
  curs = conn.cursor()
  #Selecting column names
  curs.execute("SELECT age, fare FROM Earthquake")
  rows = curs.fetchall()
  #Convert list into dataframe
  df_unscaled = pd.DataFrame(rows)
  #Removing null values
  print(df_unscaled.info())
  print(df_unscaled)
  df_unscaled = df_unscaled.dropna(axis=0, how='any')
  #df_unscaled[0]= le.fit_transform(df_unscaled[0])
  #print(df_unscaled)
  df = scaler.fit_transform(df_unscaled)
  df = pd.DataFrame(df)
  
  print(df[0])
  plotter.title('Clustering of people\n', fontsize=15)
  #Input the number of clusters
  k_clusters = request.form['clusterinput']
  #count no of NaN isnull().sum()
  kmeans= KMeans(n_clusters=int(k_clusters)).fit(df)
  #plotter.figure(figsize=(900/150, 900/150), dpi=150)
  plotter.xlabel('Age', fontsize=14)
  plotter.ylabel('Fare price', fontsize=14)
  plotter.scatter(df.iloc[:,0], df.iloc[:,1], c=[i.astype(float) for i in kmeans.labels_])#, label = ["cluster"+str(i) for i in range(1,int(k_clusters))])
  centroids = kmeans.cluster_centers_
  centroid_pts = kmeans.labels_
  df2= pd.DataFrame(centroids)  
  count1 = df2.shape[0]#No of centroids
  count2 = df.dropna(axis=0, how='any').shape[0]#No of data points
  plotter.scatter(df2.iloc[:,0], df2.iloc[:,1], c='r',marker = "^", label ='Centroids')
  #plotter.legend(loc="upper right")
  plotter.savefig('static/clusterplot1')
  plotter.clf()
  count = 0
  #count clusters
  counter = {}
  for i in range(0,int(k_clusters)):
    for j in centroid_pts:
      if j == i:
        count=count+1
    counter[i] = count
    count=0
  #distance
  distance ={}
  for i in range(0,int(k_clusters)):
    for j in range(i,int(k_clusters)):
      if i != j:
        dist = np.linalg.norm(centroids[i]-centroids[j])
        key = '('+ str(i) +',' + str(j) +')'
        distance[str(key)] = dist
  #elbow
  distortions = {}
  c_range = range(2,10)
  #silhouette score
  score_stats = {}
  '''
  for i in c_range:
    clust = KMeans(i).fit(pd.DataFrame(rows).dropna(axis=0, how='any'))
    distortions[i] = clust.inertia_ 
    k_means = KMeans(i).fit(pd.DataFrame(rows).dropna(axis=0, how='any'))
    silhouette_avg_score = silhouette_score(pd.DataFrame(rows).dropna(axis=0, how='any'), clust.labels_)
    score_stats[i] = "The average silhouette score value is : "+ str(silhouette_avg_score)
  plotter.plot(list(distortions.keys()),list(distortions.values()))
  plotter.xlabel("Number of clusters")
  plotter.ylabel("Explained Variance")
  plotter.title("Elbow method results")
  plotter.savefig('static/elbow')
  plotter.clf()
  '''
  return render_template("rangelist2.html",zip = zip(centroids, centroid_pts), count1 = count1, count2 = count2,counter = counter,score_stats = score_stats, distance = distance)

@application.route('/ranges')
def ranges():
   return render_template('Range.html')

@application.route('/values',methods = ['POST', 'GET'])
def values():
  #Connecting to database
  conn = sql.connect('database.db')
  curs = conn.cursor()
  #Selecting column names
  input1=request.form['col1']
  input2=request.form['col2']
  query = 'SELECT '+str(input1)+','+str(input2)+' FROM Earthquake ' 
  curs.execute(query)
  rows = curs.fetchall()
  print(rows)
  #Convert list into dataframe
  df_unscaled = pd.DataFrame(rows)
  #Removing null values
  df_unscaled = df_unscaled.dropna(axis=0, how='any')
  df = scaler.fit_transform(df_unscaled)
  df = pd.DataFrame(df)
  print(df)
  plotter.title('Clustering of people\n', fontsize=15)
  #Input the number of clusters
  k_clusters = request.form['clusterinput']
  #count no of NaN isnull().sum()
  kmeans= KMeans(n_clusters=int(k_clusters)).fit(df)
  #plotter.figure(figsize=(900/150, 900/150), dpi=150)
  plotter.xlabel(input1, fontsize=14)
  plotter.ylabel(input2, fontsize=14)
  plotter.scatter(df.iloc[:,0], df.iloc[:,1], c=[i.astype(float) for i in kmeans.labels_])#, label = ["cluster"+str(i) for i in range(1,int(k_clusters))])
  centroids = kmeans.cluster_centers_
  centroid_pts = kmeans.labels_
  df2= pd.DataFrame(centroids)  
  count1 = df2.shape[0]#No of centroids
  count2 = df.dropna(axis=0, how='any').shape[0]#No of data points
  plotter.scatter(df2.iloc[:,0], df2.iloc[:,1], c='r',marker = "^", label ='Centroids')
  #plotter.legend(loc="upper right")
  plotter.savefig('static/clusterplot1')
  plotter.clf()
  count = 0
  #count clusters
  counter = {}
  for i in range(0,int(k_clusters)):
    for j in centroid_pts:
      if j == i:
        count=count+1
    counter[i] = count
    count=0
  #distance
  distance ={}
  for i in range(0,int(k_clusters)):
    for j in range(i,int(k_clusters)):
      if i != j:
        dist = np.linalg.norm(centroids[i]-centroids[j])
        key = '('+ str(i) +',' + str(j) +')'
        distance[str(key)] = dist
  #elbow
  distortions = {}
  c_range = range(2,10)
  #silhouette score
  score_stats = {}
  for i in c_range:
    clust = KMeans(i).fit(pd.DataFrame(rows).dropna(axis=0, how='any'))
    distortions[i] = clust.inertia_ 
    k_means = KMeans(i).fit(pd.DataFrame(rows).dropna(axis=0, how='any'))
    silhouette_avg_score = silhouette_score(pd.DataFrame(rows).dropna(axis=0, how='any'), clust.labels_)
    score_stats[i] = "The average silhouette score value is : "+ str(silhouette_avg_score)
  plotter.plot(list(distortions.keys()),list(distortions.values()))
  plotter.xlabel("Number of clusters")
  plotter.ylabel("Explained Variance")
  plotter.title("Elbow method results")
  plotter.savefig('static/elbow')
  plotter.clf()
  
  return render_template("rangelist.html",zip = zip(centroids, centroid_pts), count1 = count1, count2 = count2,counter = counter,score_stats = score_stats, distance = distance)


@application.route('/barc')
def queries():
   return render_template('barc.html')

@application.route('/baroptions' , methods = ['POST', 'GET'])
def baroptions():
  conn = sql.connect('database.db')
  curs = conn.cursor()
  #Selecting column names
  curs.execute("SELECT sex, age FROM Earthquake")
  rows = curs.fetchall()
  #Convert list into dataframe
  df_unscaled = pd.DataFrame(rows)
  #Removing null values
  #print("Before")
  #print(df_unscaled[0])
  #print(df_unscaled[1])
  df_unscaled = df_unscaled.dropna(axis=0, how='any')
  df_unscaled[0]= le.fit_transform(df_unscaled[0])
  '''
  j=0

  for i in range(len(df_unscaled[1])):
      df_unscaled[1][i]=j
      j=j+1
  print("AFter")
  print(df_unscaled[0])
  print(df_unscaled[1])
  print(df_unscaled)'''
  df = scaler.fit_transform(df_unscaled)
  df = pd.DataFrame(df)
  
  
  plotter.title('Clustering of people\n', fontsize=15)
  #Input the number of clusters
  k_clusters = request.form['clusterinput']
  #count no of NaN isnull().sum()
  kmeans= KMeans(n_clusters=int(k_clusters)).fit(df)
  #plotter.figure(figsize=(900/150, 900/150), dpi=150)
  plotter.xlabel('Sex', fontsize=14)
  plotter.ylabel('Fare price', fontsize=14)
  plotter.scatter(df.iloc[:,0], df.iloc[:,1], c=[i.astype(float) for i in kmeans.labels_])#, label = ["cluster"+str(i) for i in range(1,int(k_clusters))])
  centroids = kmeans.cluster_centers_
  centroid_pts = kmeans.labels_
  df2= pd.DataFrame(centroids)  
  count1 = df2.shape[0]#No of centroids
  count2 = df.dropna(axis=0, how='any').shape[0]#No of data points
  plotter.scatter(df2.iloc[:,0], df2.iloc[:,1], c='r',marker = "^", label ='Centroids')
  #plotter.legend(loc="upper right")
  plotter.savefig('static/clusterplot1')
  plotter.clf()
  count = 0
  #count clusters
  counter = {}
  for i in range(0,int(k_clusters)):
    for j in centroid_pts:
      if j == i:
        count=count+1
    counter[i] = count
    count=0
  #distance
  distance ={}
  for i in range(0,int(k_clusters)):
    for j in range(i,int(k_clusters)):
      if i != j:
        dist = np.linalg.norm(centroids[i]-centroids[j])
        key = '('+ str(i) +',' + str(j) +')'
        distance[str(key)] = dist
  #elbow
  distortions = {}
  c_range = range(2,10)
  #silhouette score
  score_stats = {}
  '''
  for i in c_range:
    clust = KMeans(i).fit(pd.DataFrame(rows).dropna(axis=0, how='any'))
    distortions[i] = clust.inertia_ 
    k_means = KMeans(i).fit(pd.DataFrame(rows).dropna(axis=0, how='any'))
    silhouette_avg_score = silhouette_score(pd.DataFrame(rows).dropna(axis=0, how='any'), clust.labels_)
    score_stats[i] = "The average silhouette score value is : "+ str(silhouette_avg_score)
  plotter.plot(list(distortions.keys()),list(distortions.values()))
  plotter.xlabel("Number of clusters")
  plotter.ylabel("Explained Variance")
  plotter.title("Elbow method results")
  plotter.savefig('static/elbow')
  plotter.clf()
  '''
  return render_template("bardisplay.html",zip = zip(centroids, centroid_pts), count1 = count1, count2 = count2,counter = counter,score_stats = score_stats, distance = distance)


@application.route('/options2', methods = ['POST', 'GET'])
def options2():
  con = sql.connect("database.db")
  cur = con.cursor()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (mag BETWEEN 2 and 4 ) and ( depth between 5 and 10) ')
  row1 = cur.fetchall()
  cur.execute('SELECT mag FROM Earthquake WHERE (mag BETWEEN 2 and 4 ) and ( depth between 5 and 10) ')
  row2 = cur.fetchall()
  cur.execute('SELECT depthError FROM Earthquake WHERE (mag BETWEEN 2 and 4 ) and ( depth between 5 and 10) ')
  row3 = cur.fetchall()
  
  res=[]
  iter=['Magnitude','Depth Error']
  res.append(iter)
  count=row1[0][0]
  for i in range(count):
    iter1=[]
    iter1.append(row2[i][0])
    iter1.append(row3[i][0])
    res.append(iter1)

  cur.execute('SELECT count(*) FROM Earthquake WHERE (place LIKE "%CA") ')
  row4 = cur.fetchall()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (place LIKE "% Hawaii") ')
  row5 = cur.fetchall()
  con.close()
  return render_template("randomdisplay.html",result=res,row4=row4[0][0],row5=row5[0][0])

@application.route('/hist')
def hist():
   return render_template('hist.html')

@application.route('/histoption' , methods = ['POST', 'GET'])
def histoption():
  con = sql.connect("database.db")
  cur = con.cursor()
  loc1=request.form['loc1']
  loc2=request.form['loc2']
  loc3=request.form['loc3']
  loc4=request.form['loc4']

  
  cur.execute('SELECT count(*) FROM Earthquake WHERE place LIKE ?',('%'+loc1+'%',))
  row4 = cur.fetchall()
  cur.execute('SELECT count(*) FROM Earthquake WHERE place LIKE ?',('%'+loc2+'%',))
  row5 = cur.fetchall()
  cur.execute('SELECT count(*) FROM Earthquake WHERE place LIKE ?',('%'+loc3+'%',))
  row6 = cur.fetchall()
  cur.execute('SELECT count(*) FROM Earthquake WHERE place LIKE ?',('%'+loc4+'%',))
  row7 = cur.fetchall()
                  
  con.close()
      
  return render_template("randomdisplay2.html",loc1=loc1,loc2=loc2,loc3=loc3,loc4=loc4,row4=row4[0][0],row5=row5[0][0],row6=row6[0][0],row7=row7[0][0])


@application.route('/magniform')
def magniform():
   return render_template('Magnitude.html')

@application.route('/options' , methods = ['POST', 'GET'])
def options():
    con = sql.connect("database.db")

    cur = con.cursor()
    range1=int(request.form['range1'])
    range2=int(request.form['range2'])
    #mag2=int(request.form['mag2'])
    
    cur.execute("SELECT count(*) from Earthquake WHERE fare BETWEEN ? and ?" ,(range1,range2,))
    count = cur.fetchall()
    cur.execute("SELECT * from Earthquake WHERE fare BETWEEN ? and ? limit 5" ,(range1,range2,))
    row = cur.fetchall()

    cur.execute("DELETE from Earthquake WHERE fare BETWEEN ? and ?" ,(range1,range2,))
    row1=cur.fetchall()
    print(row1)
    con.commit()
    con.close()
    return render_template("maglist.html",rows = row,count=count)

@application.route('/timeach',methods = ['POST', 'GET'])
def timeach():
  con = sql.connect("database.db")
  dep1=float(request.form['dep1'])
  dep2=float(request.form['dep2'])

  num =int(request.form['num'])
  timeall=[]
  row2=[]
  
  
  
  rows = []
  c=[]

  for i in range(num):
    dpe1 = round(random.uniform(dep1,dep2),1)
    dpe2 = round(random.uniform(dep1,dep2),1)
    loc=str(dpe1)+str(dpe2)
    cur = con.cursor()
    result = r.get(loc)
    start_time=time.time()
    

    if result is None:
      cur.execute('SELECT latitude,longitude,"time",depthError from Earthquake WHERE (depthError BETWEEN ? and ?) ' ,(dpe1,dpe2, ))
      result = cur.fetchall()
      c.append('Not in Cache!')
      rows.append(result)
      r.set(loc, str(result))     
    else:
      c.append('Cache')
    end_time = time.time()
    elapsed_time = end_time - start_time
    timeall.append(elapsed_time)
  
  con.close()
  

  if(len(rows)==0):
    row2=["No data available!Enter appropriate values"]  

  return render_template("loctime.html",rows = c,result=row2,timeall=timeall)

@application.route('/show',methods = ['POST', 'GET'])
def show():
   con = sql.connect("database.db")
   val1=(request.form['val1'])
   cur = con.cursor()
   rows1=[]
   cur.execute("select PictureCap from Earthquake WHERE CabinNum = ? ",(request.form['val1'],))
   m1 = cur.fetchall()
   print(m1)
   print (m1[0])
   print (m1[0][0])
   """ cur.execute("select PictureCap from Earthquake WHERE Lname = ? ",(request.form['val2'],))
   m2 = cur.fetchall()"""
   con.close()
   return render_template("list1.html",rows1=m1) #,rows2=m2)

@application.route('/location')
def location():
   return render_template('Location.html')

@application.route('/distance',methods = ['POST', 'GET'])
def distance():
  con = sql.connect("database.db")
  start_time = time.time()
  cur = con.cursor()
  num =int(request.form['num'])
  lat = float(request.form['lat1'])
  lon = float(request.form['lon1'])
  dist= float(request.form['kms'])
  deg=float(dist/220)
  lat1=lat-deg
  lat2=lat+deg
  lon1=lon-deg
  lon2=lon+lat-deg
  rows=[]
  for i in range(num):
    cur = con.cursor()
    cur.execute("SELECT latitude,longitude,place from Earthquake WHERE (latitude BETWEEN ? and ?) and (longitude BETWEEN ? and ? )" ,(lat1,lat2,lon1,lon2, ))         
    get = cur.fetchall()
    rows.append(get)
                     
  con.close()
  if(len(rows)==0):
    rows=["No data available!Enter appropriate values"]   

  end_time = time.time()
  elapsed_time = end_time - start_time

  return render_template("distance.html",rows = rows,lat=lat,lon=lon,dis=dist,time=elapsed_time)

@application.route('/enight')
def enight():
  con = sql.connect("database.db")
  cur = con.cursor()
  cur.execute("select COUNT(*) from Earthquake where mag > 4  ")
  count1 = cur.fetchall();
  cur.execute('select COUNT(*) from Earthquake where mag > 4 and ("time" Like "%T20:%" or "time" Like "%T21:%" or "time" Like "%T22:%" or "time" Like "%T23:%" or "time" Like "%T00:%" or "time" Like "%T01:%" or "time" Like "%T02:%" or "time" Like "%T03:%" or "time" Like "%T04:%" or "time" Like "%T05:%")')
  result = cur.fetchall();
   #percent=(result[0]/count1[0])*100
  con.close()
  return render_template("night.html",mag4=count1,count=result)
@application.route('/loc2')
def loc2():
  return render_template('Location2.html')
@application.route('/loct')
def loct():
  return render_template('Locationt.html')

@application.route('/locsrc',methods = ['POST', 'GET'])
def locsrc():
  con = sql.connect("database.db")
  dep1=float(request.form['dep1'])
  dep2=float(request.form['dep2'])

  num =int(request.form['num'])
  timeall=[]
  row2=[]
  start_time=time.time()
  
  
  rows = []
  c=[]
  depth=[]
  for i in range(num):
    dep=[]
    dpe1 = round(random.uniform(dep1,dep2),1)
    dpe2 = round(random.uniform(dep1,dep2),1)
    dep.append(dpe1)
    dep.append(dpe2)
    depth.append(dep)
    loc=str(dpe1)+str(dpe2)
    cur = con.cursor()
    result = r.get(loc)
    if result is None:
      cur.execute('SELECT count(*) from Earthquake WHERE (depthError BETWEEN ? and ?) ' ,(dpe1,dpe2, ))
      result = cur.fetchall()
      c.append('Not in Cache!')
      rows.append(result)
      r.set(loc, str(result))     
    else:
      c.append('Cache')
  
  con.close()
  end_time = time.time()
  elapsed_time = end_time - start_time

  if(len(rows)==0):
    row2=["No data available!Enter appropriate values"]  

  return render_template("locsrc.html",rows = c,result=rows,time=elapsed_time,depth=depth)

@application.route('/loc3')
def loc3():
  return render_template('Location3.html')

@application.route('/locdis',methods = ['POST', 'GET'])
def locdis():
  con = sql.connect("database.db")
  start_time = time.time()
  cur = con.cursor()
  num =int(request.form['num'])
  lat = float(request.form['lat1'])
  lon = float(request.form['lon1'])
  dist= float(request.form['kms'])
  deg=float(dist/220)
  lat1=lat-deg
  lat2=lat+deg
  lon1=lon-deg
  lon2=lon+lat-deg
  rows=[]
  loc=request.form['lat1']+request.form['lon1']+request.form['kms']
  c=[]
  for i in range(num):
    cur = con.cursor()
    result = r.get(loc)
    if result is None:
      cur.execute("SELECT latitude,longitude,place from Earthquake WHERE (latitude BETWEEN ? and ?) and (longitude BETWEEN ? and ? )" ,(lat1,lat2,lon1,lon2, ))         
      result = cur.fetchall()
      rows.append(result)
      c.append('Not in Cache!')
      r.set(loc, str(result))
    else:
      c.append('Cache')
                     
  con.close()
  end_time = time.time()
  if(len(rows)==0):
    rows=["No data available!Enter appropriate values"]   

  
  elapsed_time = end_time - start_time

  return render_template("locdis.html",rows = c,result=rows,time=elapsed_time)

@application.route('/datesp')
def datesp():
  return render_template('date.html')

@application.route('/datespecific',methods = ['POST', 'GET'])
def datespecific():
  con = sql.connect("database.db")
  cur = con.cursor()
  loc=request.form['date1']

  cur.execute("select * from Earthquake where time like ?",('%'+loc+'%',))
  result = cur.fetchall()
   #percent=(result[0]/count1[0])*100
  con.close()
  return render_template("datelist.html",result=result)
'''
@application.route('/cluster')
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
'''
if __name__ == '__main__':
    application.run()
    #application.run(host='0.0.0.0', port=port)

