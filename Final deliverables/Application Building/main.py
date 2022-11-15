from flask import Flask,request, url_for, redirect, render_template,session
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
parkinsons_data = pd.read_csv('static/uploads/parkinsons.csv')
parkinsons_data.drop(['name'], axis=1, inplace=True)
x = parkinsons_data.drop(['status'], axis=1)
y = parkinsons_data['status']
sm = SMOTE(random_state=300)
x, y = sm.fit_resample(x, y)
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(x)
Y = y
x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.20, random_state=20)
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
filename = 'parkinson_model.pickle'
pickle.dump(rfc, open(filename, 'wb'))
app = Flask(__name__, template_folder='templates', static_folder='static')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'This is your secret key to utilize session in Flask'
@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route('/home')
def home():
    return render_template("index.html")
@app.route('/login')
def login():
    return render_template("login.html")
@app.route('/form_login',methods=['POST','GET'])
def login1():
    database={'user1':'1234','user2':'abcd','admin':'admin'}
    name1=request.form['username']
    pwd=request.form['password']
    if name1 not in database:
        return render_template('login.html',info='Invalid User')
    else:
         if database[name1]!=pwd:
             return render_template('login.html',info='Invalid password')
         else:
             # return render_template('login.html',info='login Successfull')
             return render_template('upload.html',name=name1)
@app.route('/upload')
def upload_file():
    return render_template('upload.html')
@app.route('/', methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']

        # Extracting uploaded data file name
        data_filename = secure_filename(uploaded_df.filename)

        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))

        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)

        return render_template('upload2.html')
@app.route('/show_data')
def showData():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)

    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    uploaded_df = pd.read_csv(data_file_path)
    # pandas dataframe to html table flask
    uploaded_df_html = uploaded_df.to_html()
    return render_template('preview.html', data_var=uploaded_df_html)
@app.route('/input_data',methods=['GET'])  # route to display the home page
def inputPage():
    return render_template("form1.html")
@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def predict():
    if request.method == 'POST':
        mdvp_fo = float(request.form['mdvp_fo'])
        mdvp_fhi = float(request.form['mdvp_fhi'])
        mdvp_flo = float(request.form['mdvp_flo'])
        mdvp_jitter1 = float(request.form['mdvp_jitter1'])
        mdvp_jitter2 = float(request.form['mdvp_jitter2'])
        mdvp_rap = float(request.form['mdvp_rap'])
        mdvp_ppq = float(request.form['mdvp_ppq'])
        jitter_ddp = float(request.form['jitter_ddp'])
        mdvp_shimmer = float(request.form['mdvp_shimmer'])
        mdvp_shimmer2 = float(request.form['mdvp_shimmer2'])
        shimmer_apq3 = float(request.form['shimmer_apq3'])
        shimmer_apq5 = float(request.form['shimmer_apq5'])
        mdvp_apq = float(request.form['mdvp_apq'])
        shimmer_dda = float(request.form['shimmer_dda'])
        nhr = float(request.form['nhr'])
        hnr = float(request.form['hnr'])
        rpde = float(request.form['rpde'])
        dfa = float(request.form['dfa'])
        spread1 = float(request.form['spread1'])
        spread2 = float(request.form['spread2'])
        d2 = float(request.form['d2'])
        ppe = float(request.form['ppe'])
        input_data = (mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter1, mdvp_jitter2, mdvp_rap, mdvp_ppq, jitter_ddp, mdvp_shimmer,
             mdvp_shimmer2, shimmer_apq3, shimmer_apq5, mdvp_apq, shimmer_dda, nhr, hnr, rpde, d2, dfa, spread1,
             spread2, ppe)

        # changing input data to numpy array
        input_data_numpy = np.asarray(input_data)

        # reshaping the numpy array
        input_data_reshape = input_data_numpy.reshape(1, -1)

        # standardizing the input data
        std_data = scaler.transform(input_data_reshape)
        filename = 'parkinson_model.pickle'

        # loading the model file from the storage
        loaded_model = pickle.load(open(filename, 'rb'))

        # predictions using the loaded model file
        prediction = loaded_model.predict(std_data)
        print('prediction is', prediction)

        if (prediction[0] == 0):
            return render_template('form1.html', prediction_text="The Person does not have Parkinsons Disease")

        else:
            return render_template('form1.html', prediction_text="The Person has Parkinsons")
    else:
        return render_template('form1.html')
if __name__ == '__main__':
    app.run(debug=True)

