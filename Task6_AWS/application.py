#Author: Shreyas Mohan

from flask import Flask,render_template,request
from werkzeug import secure_filename
import sqlite3 as sql
import pandas as pd
#import ibm_db
from math import sin, cos, sqrt, atan2, radians
import os
#import sqlite3
import csv
import time
import random
#from pymemcache.client import base
#from sklearn.cluster import KMeans
#from matplotlib import pyplot as plt
import redis
timelist=[]
myHostname = "shreyasm.redis.cache.windows.net"
myPassword = "vpbSR5REB24mLu2Fzr4GeH+8WzdX7gPdh6Q7+AYRWRA="

#r = redis.StrictRedis(host=myHostname, port=6380,password=myPassword,ssl=True)
var=0
app = application =Flask(__name__)
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
    cur.execute("select * from Earthquake")
    rows = cur.fetchall()
    cur.execute("select count(*) from Earthquake ")
    count = cur.fetchall()

    cur.execute("select * from students")
    rows1 = cur.fetchall()
    cur.execute("select count(*) from students ")
    count1 = cur.fetchall()
    '''count=0
    minval=100
    for row in rows:
      count=count+1
      mag=row[5]
      place=row[7]
      if mag<minval and mag>2:
        minval=mag
        mine=row'''
      

    con.close()
    return render_template('dissix.html',count=count,data=rows,count1=count1,data1=rows1)

@app.route('/year', methods = ['POST', 'GET'])      
def year():
  con = sql.connect("database.db")
  cur = con.cursor()
  #num=request.form['1']
  start_time = time.time()
  global var
  global timelist

  #q='SELECT '+str(str(year))+' from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"'
  #cur.execute(q)
  if var == 0:
      cur.execute('SELECT state,"2010" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      var=var+1
      timelist.append(elapsed_time)
      con.close()
      
  elif var == 1:
      cur.execute('SELECT state,"2011" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      var=var+1
      timelist.append(elapsed_time)
      con.close()
  elif var == 2:
      cur.execute('SELECT state,"2012" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      timelist.append(elapsed_time)
      var=var+1
      con.close()
  elif var == 3:
      cur.execute('SELECT state,"2013" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      var=var+1
      timelist.append(elapsed_time)
      con.close()
  elif var == 4:
      cur.execute('SELECT state,"2014" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      var=var+1
      timelist.append(elapsed_time)
      con.close()
  elif var == 5:
      cur.execute('SELECT state,"2015" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      var=var+1
      timelist.append(elapsed_time)
      con.close()
  elif var == 6:
      cur.execute('SELECT state,"2016" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      var=var+1
      timelist.append(elapsed_time)
      con.close()
  elif var == 7:
      cur.execute('SELECT state,"2017" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      var=var+1
      timelist.append(elapsed_time)
      con.close()
  elif var == 8:
      cur.execute('SELECT state,"2018" from Earthquake where state="Texas" or state="Louisiana" or state="Oklahoma"')
      rows = cur.fetchall()
      end_time = time.time()
      elapsed_time = end_time - start_time
      var=0
      timelist.append(elapsed_time)
      con.close()
  
  end_time = time.time()
  elapsed_time = end_time - start_time

  con.close()
  return render_template("displayrec.html",data=rows,s=start_time,e=end_time,timetaken=elapsed_time)

@app.route('/queries')
def queries():
   return render_template('queries.html')

'''
@app.route('/options1' , methods = ['POST', 'GET'])
def options1():
  con = sql.connect("database.db")
  start_time = time.time()
  number =int(request.form['mag'])
  rows = []
  for i in range(number):
    magni = round(random.uniform(1,6),1)
    cur = con.cursor()
    q = str(magni)

    #cur.execute("select * from Earthquake WHERE mag = ?" ,(magni,))
    #get = cur.fetchall();
    #rows.append(get)
    result = r.get(magni)
    if result is None:
    # The cache is empty, need to get the value
    # from the canonical source:
      cur.execute("select * from Earthquake WHERE mag = ?" ,(magni,))
      result = cur.fetchall();

    # Cache the result for next time:
      r.set(q, str(result))   
  con.close()
  end_time = time.time()
  elapsed_time = end_time - start_time
  return render_template("rd.html",rows = elapsed_time)
'''
@app.route('/queries2')
def queries2():
   return render_template('queries2.html') 

@app.route('/options2' , methods = ['POST', 'GET'])
def options2():
      con = sql.connect("database.db")
      start_time = time.time()     
      num =int(request.form['num'])
      loc = (request.form['loc'])
      rows = []
      for i in range(num):
              cur = con.cursor()
              #b = 'select * from Earthquake WHERE place LIKE ?', ('%'+loc+'%',)
              cur.execute("select * from Earthquake WHERE place LIKE ? ", ('%'+loc+'%',))
              get = cur.fetchall()
              rows.append(get)
                     
      con.close()
      end_time = time.time()
      elapsed_time = end_time - start_time
      return render_template("randomdisplay.html",time = elapsed_time,rows=rows)

@app.route('/queries3')
def queries3():
   return render_template('queries3.html')
'''
@app.route('/options3' , methods = ['POST', 'GET'])
def options3():
      con = sql.connect("database.db")
      start_time = time.time()     
      num =int(request.form['num'])
      loc = (request.form['loc'])
      rows = []
      c=[]
      for i in range(num):
        cur = con.cursor()
        result = r.get(loc)
        
        if result is None:
                # The cache is empty, need to get the value
                # from the canonical source:
          cur.execute("select * from Earthquake WHERE place LIKE ? ", ('%'+loc+'%',))
          result = cur.fetchall()
          c.append('Not in Cache!')
          rows.append(result)
          r.set(loc, str(result))     
        else:
          c.append('Cache')               
      con.close()
      end_time = time.time()
      elapsed_time = end_time - start_time
      return render_template("randomdisplay2.html",time = elapsed_time,rows=c,result=rows)
'''
@app.route('/magniform')
def magniform():
   return render_template('Magnitude.html')

@app.route('/options' , methods = ['POST', 'GET'])
def options():
    con = sql.connect("database.db")

    cur = con.cursor()
    i=[]
    depth1=int(request.form['d1'])
    depth2=int(request.form['d2'])
    interval=int(request.form['interval'])
    flag=0
    result=[]
   
    for inte in range(depth1,depth2,interval):
      i.append(inte)
      flag=flag+1
    j=0
    while (j<flag and j!=flag-1):
      cur.execute("SELECT * from Earthquake WHERE mag BETWEEN ? and ?" ,(i[j],i[j+1], ))
      row = cur.fetchall()
      cur.execute("SELECT count(*) from Earthquake WHERE mag BETWEEN ? and ?" ,(i[j],i[j+1], ))
      rc = cur.fetchall()
      result.append(row)
      j=j+1


    if(len(result)==0):
      result=["No data available!Enter appropriate values"]
      return render_template("maglist.html",rows = result)
    else:
      return render_template("maglist.html",rows = result)

@app.route('/ranges')
def ranges():
   return render_template('Range.html')

@app.route('/range1')
def range1():
   return render_template('Range1.html')

@app.route('/values',methods = ['POST', 'GET'])
def values():
  con = sql.connect("database.db")
  dep1=float(request.form['dep1'])
  dep2=float(request.form['dep2'])

  num =int(request.form['num'])
  timeall=[]
  row2=[]
  for i in range(num):
    dpe1 = round(random.uniform(dep1,dep2),1)
    dpe2 = round(random.uniform(dep1,dep2),1)
    print(dpe1)
    print(dpe2)
    start_time=time.time()
    cur = con.cursor()
    cur.execute('SELECT latitude,longitude,"time",depthError from Earthquake WHERE (depthError BETWEEN ? and ?) ' ,(dpe1,dpe2, ))
    get = cur.fetchall()
    row2.append(get)
    end_time = time.time()
    elapsed_time = end_time - start_time
    #row2.append(elapsed_time)
    timeall.append(elapsed_time)
  
  con.close()
  
  
  if(len(row2)==0):
    row2=["No data available!Enter appropriate values"]

  return render_template("rangelist.html",row2=row2,timeall=timeall)
'''
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
    end_time =time.time()
    elapsed_time = end_time - start_time
    timeall.append(elapsed_time)
  
  con.close()
  

  if(len(rows)==0):
    row2=["No data available!Enter appropriate values"]  

  return render_template("loctime.html",rows = c,result=row2,timeall=timeall)
'''
@app.route('/values1',methods = ['POST', 'GET'])
def values1():
  con = sql.connect("database.db")
  course=request.form['course']
  section=request.form['section']
  
  cur = con.cursor()
  cur.execute('SELECT "Max" FROM Earthquake WHERE "Course " = ? and "Section" = ?',(course,section, ))
  rows = cur.fetchall()

  cap1=int(rows[0][0])
  print(cap1)
  cap=0
  if(cap1>0):
    cap=cap1-1
    print(cap)
    cur.execute('UPDATE Earthquake set "Max" = ? WHERE "Course " = ? and "Section" = ?',(cap,course,section, ))
    row2 = cur.fetchall()
    print(row2)
    con.commit()

  
  
  con.close()
  '''
  if(len(row2)==0):
    row2=["No data available!Enter appropriate values"]
  '''
  return render_template("rangelist2.html",rows=rows,cap=cap1,new=cap);

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
'''
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
'''
@app.route('/loc3')
def loc3():
  return render_template('Location3.html')
'''
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
'''
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

