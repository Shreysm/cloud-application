
from flask import Flask,render_template,request
from werkzeug import secure_filename
import os

imagefilelist=[]

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

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   
   if request.method == 'POST':
      f = request.files['myfile']
      filename=secure_filename(f.filename)
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
      if filename.endswith(".jpg"):
         return render_template("file_upload.html",msg = filename)
      elif filename.endswith(".csv"):
         return render_template("file_upload.html",msg = filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
