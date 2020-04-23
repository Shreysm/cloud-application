
from flask import Flask,render_template,request
from werkzeug import secure_filename
import os
import csv

imagefilelist=[]

app = Flask(__name__)
folder='./static'
app.config['UPLOAD_FOLDER']=folder
port = int(os.getenv("PORT", 5000))

with open('static/people.csv') as csvfile:
      readCSV = csv.reader(csvfile, delimiter=',')
      names= []
      points= []
      states=[]
      rooms= []
      pictures= []
      favorites=[]
      for row in readCSV:
         name=row[0]
         point=row[1]
         state=row[2]
         room=row[3]
         picture=row[4]
         favorite=row[5]

         names.append(name)
         points.append(point)
         states.append(state)
         rooms.append(room)
         pictures.append(picture)
         favorites.append(favorite)

@app.route('/')
def home():
   return render_template('home.html')


@app.route('/enternew')
def upload_csv():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   
   if request.method == 'POST':
      f = request.files['myfile']
      filename=secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
      if filename.endswith(".jpg"):
      	return render_template("file_upload.html",msg = filename)
      elif filename.endswith(".csv"):
         return render_template("file_upload.html",msg = " CSV ")

@app.route('/imagelist')
def imagelist():
   imagefilelist=["ab.jpg","bob.jpg","dar.jpg","jason.jpg","jees.jpg","pepe.jpg"]
   size=[109.6,129.4,43.1,201.6,836.4,81.0]
   return render_template("imagelist.html", result = imagefilelist,size=size)

@app.route('/displaypic')
def displaypic():
   return render_template('display.html')

@app.route('/stateimage', methods = ['GET', 'POST'])
def stateimage():
      name=[]
      pic=[]
      fav=[]
      pointsin=request.form['points']
      for i in range(len(points)):
         if points[i] == pointsin:
            name.append(names[i])
            pic.append(pictures[i])
            fav.append(favorites[i])
      return render_template("state_imaage.html",name=name,pic=pic,fav=fav)

@app.route('/records')
def records():    
   return render_template("displayrec.html",names=names,points=points,states=states,rooms=rooms,pictures=pictures,favorites=favorites)

@app.route('/modifyrec')
def modifyrec():
   return render_template('modifyrec.html')

@app.route('/twopointrec')
def twopointrec():
   return render_template('displaytwop.html')
@app.route('/modifypoint', methods = ['GET', 'POST'])
def modifypoint():
   
      pointin=request.form['point']
      namein=request.form['name']
      favin=request.form['fav']
      pnew=request.form['pointnew']
      for i in range(len(points)):
         if points[i] == pointin:
            names[i]=namein
            favorites[i]=favin
            points[i]=pnew
      
      return render_template("modified.html")
'''
@app.route('/twop', methods = ['GET', 'POST'])
def stateimage():
      
      pointsin=request.form['points']
      for i in range(len(points)):
         if points[i] == pointsin:
            name.append(names[i])
            pic.append(pictures[i])
            fav.append(favorites[i])
      return render_template("state_imaage.html",name=name,pic=pic,fav=fav)'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
