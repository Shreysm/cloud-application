from flask import Flask,render_template,request
from werkzeug import secure_filename
import sqlite3 as sql
#import pandas as pd
#import ibm_db
from math import sin, cos, sqrt, atan2, radians
import os
#import sqlite3
import csv
import time
import random

#from sklearn.cluster import KMeans
#from matplotlib import pyplot as plt

app = Flask(__name__)
folder='./static'
app.config['UPLOAD_FOLDER']=folder
port = int(os.getenv("PORT", 5000))


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
       #file = pd.read_csv(csv)
       #file.to_sql('Earthquake', con, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)   
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
    if mag is not None and mag<minval and mag>2:
        minval=mag
        mine=row
      

  con.close()
  return render_template("displayrec.html",rows = mine,mag=minval,count=count,place=place,data=rows)

@app.route('/range1')
def range1():
   return render_template('Range1.html')

@app.route('/values1',methods = ['POST', 'GET'])
def values1():
  con = sql.connect("database.db")
  lat1=request.form['lat1']
  lat2=request.form['lat2']
  lon1=request.form['lon1']
  lon2=request.form['lon2']
  result=[]

  cur = con.cursor()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (latitude < ? ) and ( longitude < ?) ',(lat1,lon1, ))
  row1 = cur.fetchall()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (latitude BETWEEN ? and ? ) and ( longitude between ? and ?) ',(lat1,lat2,lon1,lon2, ))
  row2 = cur.fetchall()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (latitude > ? ) and ( longitude > ?) ',(lat2,lon2, ))
  row3 = cur.fetchall()
  

  con.close()
  
  return render_template("rangelist2.html",row1=row1[0][0],row2=row2[0][0],row3=row3[0][0])

@app.route('/ranges')
def ranges():
   return render_template('Range.html')

@app.route('/values',methods = ['POST', 'GET'])
def values():
  con = sql.connect("database.db")
  mag1=request.form['mag1']
  mag2=request.form['mag2']
  dep1=request.form['dep1']
  dep2=request.form['dep2']

  cur = con.cursor()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (mag BETWEEN ? and ? ) and ( depth between ? and ?) ',(mag1,mag2,dep1,dep2, ))
  row1 = cur.fetchall()
  cur.execute('SELECT mag FROM Earthquake WHERE (mag BETWEEN ? and ? ) and ( depth between ? and ?) ',(mag1,mag2,dep1,dep2, ))
  row2 = cur.fetchall()
  cur.execute('SELECT depthError FROM Earthquake WHERE (mag BETWEEN ? and ? ) and ( depth between ? and ?) ',(mag1,mag2,dep1,dep2, ))
  row3 = cur.fetchall()
  con.close()
  res=[]
  iter=['Magnitude','Depth Error']
  res.append(iter)
  count=row1[0][0]
  for i in range(count):
    iter1=[]
    iter1.append(row2[i][0])
    iter1.append(row3[i][0])
    res.append(iter1)

  return render_template("rangelist.html",result=res)


@app.route('/barc')
def queries():
   return render_template('barc.html')

@app.route('/baroptions' , methods = ['POST', 'GET'])
def baroptions():
  con = sql.connect("database.db")
  
  mag =request.form['mag']
  rows = []

  cur = con.cursor()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (mag < ?) and (place LIKE "%CA") ',(mag, ))
  row1 = cur.fetchall()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (mag >= ?) and (place LIKE "%CA") ',(mag, ))
  row2 = cur.fetchall()

  cur.execute('SELECT count(*) FROM Earthquake WHERE (mag < ?) and (place LIKE "% Hawaii") ',(mag, ))
  row3 = cur.fetchall()
  cur.execute('SELECT count(*) FROM Earthquake WHERE (mag >= ?) and (place LIKE "% Hawaii") ',(mag, ))
  row4 = cur.fetchall()

  
  con.close()
  
  return render_template("bardisplay.html",row1=row1[0][0],row2=row2[0][0],row3=row3[0][0],row4=row4[0][0])


@app.route('/options2', methods = ['POST', 'GET'])
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

@app.route('/hist')
def hist():
   return render_template('hist.html')

@app.route('/histoption' , methods = ['POST', 'GET'])
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


@app.route('/magniform')
def magniform():
   return render_template('Magnitude.html')

@app.route('/options' , methods = ['POST', 'GET'])
def options():
    con = sql.connect("database.db")

    cur = con.cursor()
    mag1=int(request.form['mag1'])
    mag2=int(request.form['mag2'])
    interval=1
    flag=0
    result=[]
    i=[]
   
    for inte in range(mag1,mag2,interval):
      i.append(inte)
      flag=flag+1
    j=0

    while (j<flag and j!=flag-1):
      cur.execute("SELECT * from Earthquake WHERE mag BETWEEN ? and ?" ,(i[j],i[j+1], ))
      row = cur.fetchall()
      cur.execute("SELECT count(*) from Earthquake WHERE mag BETWEEN ? and ?" ,(i[j],i[j+1], ))
      rc = cur.fetchall()
      result.append(rc)
      j=j+1

    res=[]
    iter=['Magnitude','Count']
    res.append(iter)
    
    for i in range(j):
      iter1=[]
      iter1.append('Group'+str(i))
      iter1.append(result[i][0][0])
      res.append(iter1)

    return render_template("maglist.html",rows = res)

@app.route('/timeach',methods = ['POST', 'GET'])
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


@app.route('/location')
def location():
   return render_template('Location.html')

@app.route('/distance',methods = ['POST', 'GET'])
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
@app.route('/loct')
def loct():
  return render_template('Locationt.html')

@app.route('/locsrc',methods = ['POST', 'GET'])
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

@app.route('/loc3')
def loc3():
  return render_template('Location3.html')

@app.route('/locdis',methods = ['POST', 'GET'])
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

@app.route('/datesp')
def datesp():
  return render_template('date.html')

@app.route('/datespecific',methods = ['POST', 'GET'])
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

