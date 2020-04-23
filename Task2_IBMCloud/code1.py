from flask import Flask,render_template,request
from werkzeug import secure_filename
import sqlite3 as sql
import pandas as pd
#import ibm_db
from math import sin, cos, sqrt, atan2, radians
import os
import sqlite3
import csv
#from sklearn.cluster import KMeans
#from matplotlib import pyplot as plt


app = Flask(__name__)
folder='./static'
app.config['UPLOAD_FOLDER']=folder
port = int(os.getenv("PORT", 5000))
#con=ibm_db.connect("DATABASE=BLUDB;HOSTNAME=dashdb-txn-sbox-yp-dal09-03.services.dal.bluemix.net;PORT=50000;PROTOCOL=TCPIP;UID=vlv63547;PWD=@Charya18;", "", "")


@app.route('/')
def home():
   return render_template('home.html')

@app.route('/enternew')
def upload_csv():
   return render_template('upload.html')

@app.route('/upload_file',methods = ['POST', 'GET'])
def upload_file():
   if request.method == 'POST':
       con = sql.connect("database.db")
       csv = request.files['myfile']
       filename=secure_filename(csv.filename)
       file = pd.read_csv(csv)
       file.to_sql('Earthquake', con, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)	  
       con.close()

       return render_template("file_upload.html",msg = filename)

@app.route('/records')      
def records():
  con = sql.connect("database.db")
   
  cur = con.cursor()
  cur.execute("select * from Earthquake ")
  rows = cur.fetchall()
  count=0
  minval=100
  for row in rows:
    count=count+1
    mag=row[5]
    place=row[7]
    if mag<minval and mag>2:
        minval=mag
        mine=row
      

  con.close()
  return render_template("displayrec.html",rows = mine,mag=minval,count=count)

@app.route('/magniform')
def magniform():
   return render_template('Magnitude.html')

@app.route('/options' , methods = ['POST', 'GET'])
def options():
    con = sql.connect("database.db")

    cur = con.cursor()

    mag=request.form['mag']
    if request.form['1']== 'greater':
      cur.execute("SELECT Count(*) from earthquake where mag > ?",(request.form['mag'],)) 
      rows = cur.fetchall()
      cur.execute("SELECT * from Earthquake where mag > ?",(request.form['mag'],))
      rows_2 = cur.fetchall()
    elif request.form['1']== 'lesser':
      cur.execute("SELECT Count(*) from Earthquake where MAG < ?",(request.form['mag'],))
      rows = cur.fetchall()
      cur.execute("SELECT * from Earthquake where MAG < ?",(request.form['mag'],))
      rows_2 = cur.fetchall()
    else :
      cur.execute("SELECT Count(*) from Earthquake where MAG = ?",(request.form['mag'],))
      rows = cur.fetchall()
      cur.execute("SELECT * from Earthquake where MAG = ?",(request.form['mag'],))
      rows_2 = cur.fetchall()

    if(len(rows_2)==0):
      rows=1
      rows_2=["No data available!Enter appropriate values"]
      return render_template("maglist.html",rows = [rows,rows_2])
    else:
      return render_template("maglist.html",rows = [rows,rows_2])

@app.route('/ranges')
def ranges():
   return render_template('Range.html')

@app.route('/values',methods = ['POST', 'GET'])
def values():
  con = sql.connect("database.db")
  m1=request.form['mag1']
  m2=request.form['mag2']
  cur = con.cursor()
  cur.execute("SELECT* from Earthquake WHERE mag < ?" ,(m1,))
  row1 = cur.fetchall()
  cur.execute("SELECT count(*) from Earthquake WHERE mag < ?" ,(m1,))
  bc= cur.fetchall()

  cur.execute("SELECT * from Earthquake WHERE mag BETWEEN ? and ?" ,(request.form['mag1'],request.form['mag2'], ))
  row2 = cur.fetchall()
  cur.execute("SELECT count(*) from Earthquake WHERE mag BETWEEN ? and ?" ,(request.form['mag1'],request.form['mag2'], ))
  rc = cur.fetchall()
  
  cur.execute("SELECT * from Earthquake WHERE mag > ?" ,(request.form['mag2'],))
  row3 = cur.fetchall()
  cur.execute("SELECT count(*) from Earthquake WHERE mag > ?" ,(request.form['mag2'],))
  ac= cur.fetchall()

  con.close()
  if(len(row1)==0):
    row1=["No data available!Enter appropriate values"]
  if(len(row2)==0):
    row2=["No data available!Enter appropriate values"]
  if(len(row3)==0):
    row3=["No data available!Enter appropriate values"]
  return render_template("rangelist.html",row1 = row1,row2=row2,row3=row3,mag1=m1,mag2=m2,count1=bc,count2=rc,count3=ac)

@app.route('/location')
def location():
   return render_template('Location.html')

@app.route('/distance',methods = ['POST', 'GET'])
def distance():
  con = sql.connect("database.db")
  cur = con.cursor()
  cur.execute("select * from Earthquake ")   
  rows = cur.fetchall();
   #ref:https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
  R = 6373.0
  lat1 = radians(float(request.form['lat1']))
  lon1 = radians(float(request.form['lon1']))
  dist =[]
  for row in rows:
    lat2 = radians(row[2])
    lon2 = radians(row[3])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance =float(R * c)
    if distance <= (float(request.form['kms'])):
      dist.append(row)
    con.close()
  if(len(dist)==0):
    dist=["No data available!Enter appropriate values"]
  return render_template("distance.html",rows = dist,lat=request.form['lat1'],lon=request.form['lon1'],dis=request.form['kms'])

@app.route('/enight')
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
@app.route('/loc2')
def loc2():
  return render_template('Location2.html')

@app.route('/locsrc',methods = ['POST', 'GET'])
def locsrc():
  mag=request.form['mag']
  loc=request.form['loc']
  con = sql.connect("database.db")
  cur = con.cursor()
  cur.execute("select time,latitude,longitude,place from Earthquake WHERE locationSource=? and mag >? and magNst*2 >= nst" ,(request.form['loc'],request.form['mag'] ))    
  rows = cur.fetchall();

  if(len(rows)==0):
    rows=["No data available!Enter appropriate values"]
  return render_template("locsrc.html",rows = rows)

@app.route('/loc3')
def loc3():
  return render_template('Location3.html')

@app.route('/locdis',methods = ['POST', 'GET'])
def locdis():
  con = sql.connect("database.db")
  cur = con.cursor()
  cur.execute("select * from Earthquake ")   
  rows = cur.fetchall();
   #ref:https://stackoverflow.com/questions/19412462/getting-distance-between-two-points-based-on-latitude-longitude?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
  R = 6373.0
  lat1 = radians(float(-13))
  lon1 = radians(float(45))
  mine=[]
  maxe=[]
  minval=100
  maxval=0
  inputdis=float(request.form['kms'])
  print(inputdis)
  count=0
  for row in rows:
    lat2 = radians(row[2])
    lon2 = radians(row[3])
    mag=row[5]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance =float(R * c)
    print(distance)
    if distance <= (inputdis):
      count=count+1
      if mag<minval:
        minval=mag
        mine=row
      if mag>maxval:
        maxval=mag
        maxe=row
  print(count)
  if(len(mine)==0):
    mine=["No data available!Enter appropriate values"]
  if(len(maxe)==0):
    maxe=["No data available!Enter appropriate values"]
  con.close()
  return render_template("locdis.html",dis=request.form['kms'],count=count,maxeq=maxe,mineq=mine,min=minval,max=maxval)

@app.route('/date')
def date():
  return render_template('date.html')

@app.route('/datespecific')
def datespecific():
  con = sql.connect("database.db")
  cur = con.cursor()

  cur.execute('select * from Earthquake where time like "%/?/%"',request.form['date'])
  result = cur.fetchall();
   #percent=(result[0]/count1[0])*100
  con.close()
  return render_template("datelist.html",result=result)
'''
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
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)

