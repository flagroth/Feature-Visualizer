
# coding: utf-8
import os
from flask import Flask, render_template, request, redirect, url_for

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

import pandas as pd
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, gridplot
from bokeh.charts import Scatter
from bokeh.models import ColumnDataSource,HoverTool
from bokeh.embed import components
from bokeh.palettes import Category20

from werkzeug.utils import secure_filename

from flask import send_from_directory

UPLOAD_FOLDER = '../data'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
_filename="kk_filename"
_target="kk_target"
_separator=","

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def data_prep(filename=_filename, target=_target, separator=_separator, features=None):
    dataset = pd.read_csv("../data/"+filename, sep=separator)
    dataset = dataset.apply(pd.to_numeric, errors='ignore')
    print(dataset.columns)
    y = dataset[target]
    #x = dataset.drop(['ID','ID.1','Name','Overall','Photo'], axis = 1)

    if features is not None:
        x = dataset[features]
    else:
        x = dataset
    data_columns = dataset.columns
    x = x.apply(pd.to_numeric, errors='ignore')
    #x = (x - x.mean()) / (x.max() - x.min())
    msk = np.random.rand(len(x)) < 0.7
    train_x = x[msk]
    train_y = y[msk]
    test_x = x[~msk]
    test_y = y[~msk]

    return ({"data_columns":data_columns,
            "train_x":train_x,
            "train_y":train_y,
            # "train_names":train_names,
            # "train_photo":train_photo,
            "test_x":test_x,
            "test_y":test_y
            # "test_names":test_names,
            # "test_photo":test_photo
            })

def make_plot(source_points, compo=None, columns=None, colors=None, method=None):
#    target_names = ['Bueno','Regular','Malo']
    
    # hover = HoverTool(tooltips=""""
    # <div>
        # <div>
            # <img
                # src="@imgs" alt="@imgs"
            # ></img>
        # </div>
         # <div>
            # <span style="font-size: 17px; font-weight: bold;">@desc</span>
            # <span style="font-size: 15px; color: #966;">[$index]</span>
        # </div>
    # """)
    
    p = figure(tools = ["pan,wheel_zoom,box_select,lasso_select,reset"], width = 300, height = 300, title=method)
    if method is None:
        return(1)
    if method=='PCA':
        x='x_pca'
        y='y_pca'
        compo_x='comp_pca_x'
        compo_y='comp_pca_y'
    if method=='LDA':
        x='x_lda'
        y='y_lda'
        compo_x='comp_lda_x'
        compo_y='comp_lda_y'
    if method=='TSNE':
        x='x_tsne'
        y='y_tsne'
    if method=='MDS':
        x='x_mds'
        y='y_mds'
               
    p.circle(x, y , source = source_points, alpha=.8, color=colors)#X_r[y == i, 0], X_r[y == i, 1], color=color)
    #p.line(compo_x, compo_y, source = source_lines, color = 'red')
    if method in ['PCA','LDA']:
        for a,b,label in zip(compo[0,:],compo[1,:],columns):
            p.line([0,a],[0,b], color = 'red')
            p.text([a],[b],text=[label], text_align="center")
    return(p)

@app.route('/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
    
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            target = request.form.get("target")
            separator = request.form.get("separator")
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',
            return redirect(url_for('features', filename=filename, target=target, separator=separator))
    return '''
    <!doctype html>
    <title>Feature visualizer</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file> *Only CSV
         Target column:<input type=text name=target>
		 CSV separator<input type=text name=separator>
         <input type=submit value=Upload>
    </form>
    '''
    
@app.route('/features?filename=<filename>&target=<target>&separator=<separator>', methods=['GET'])
def features(filename,target,separator):
    global _filename,_target, _separator
    _filename=filename
    _target=target
    _separator=separator

    data=data_prep(_filename,_target,_separator)
    #data_columns,train_x,train_y,train_names,train_photo,test_x,test_y,test_names,test_photo = data.values()
    data_columns,train_x,train_y,test_x,test_y = data.values()
    return render_template("visualizer.html", feature_names=train_x.columns, filename=filename, target=target)
    
@app.route('/visualize', methods=['POST'])
def visualize():
    features = request.form.getlist('feature')
    if len(features)==0:
        features=None
    data=data_prep(_filename, _target, _separator, features)
    data_columns,train_x,train_y,test_x,test_y = data.values()
    print(train_x.columns)
    
    vizs = request.form.getlist('viz')
    print(vizs)
    
    dataSource=dict()
    if "PCA" in vizs:
        print("Making PCA")
        pca = PCA(n_components=2)
        X_r_pca = pca.fit(train_x).transform(train_x)
        comp_pca = pca.components_
        X_r_test = pca.transform(test_x)
        dataSource['x_pca'] = X_r_pca[:,0]
        dataSource['y_pca'] = X_r_pca[:,1]
    
    if "LDA" in vizs:
        print("Making LDA")
        lda = LDA(n_components=2)
        X_r_lda = lda.fit(train_x,train_y).transform(train_x)
        comp_lda = lda.coef_
        y_predict_lda = lda.predict(test_x)
        X_r_lda_test = lda.fit(test_x,y_predict_lda).transform(test_x)
        dataSource['x_lda'] = X_r_lda[:,0]
        dataSource['y_lda'] = X_r_lda[:,1]
    
    if "TSNE" in vizs:
        print("Making t-SNE")
        tsne_iter = int(request.form.get('tsne_iter'))
        tsne = TSNE(n_components=2,n_iter=tsne_iter)
        X_r_tsne = tsne.fit_transform(train_x)
        dataSource['x_tsne'] = X_r_tsne[:,0]
        dataSource['y_tsne'] = X_r_tsne[:,1]

    if "MDS" in vizs:
        print("Making MDS")
        mds_iter = int(request.form.get('mds_iter'))
        mds = MDS(n_components=2,max_iter=mds_iter)
        X_r_mds = mds.fit_transform(train_x)
        dataSource['x_mds'] = X_r_mds[:,0]
        dataSource['y_mds'] = X_r_mds[:,1]
    
    print("Colouring")
    color = Category20[20]
    print(np.unique(train_y.astype(int)))
    print(color)
    colors = []
    for i in train_y.astype(int):
        colors.append(color[i]) 
        
    source_points = ColumnDataSource(data=dataSource)
       
    g=[]
    print("Plotting")
    if "PCA" in vizs:
        g.append(make_plot(source_points, comp_pca, train_x.columns, colors, "PCA"))
    if "LDA" in vizs:
        g.append(make_plot(source_points, comp_lda, train_x.columns, colors, "LDA"))
    if "TSNE" in vizs:    
        g.append(make_plot(source_points, colors=colors, method="TSNE"))
    if "MDS" in vizs:    
        g.append(make_plot(source_points, colors=colors, method="MDS"))
    p = gridplot([g])
    script, div = components(p)
    return render_template("visualizer.html",feature_names=data_columns, script=script, div=div)
    
    
if __name__ == '__main__':
    app.run('10.71.184.225',port=5000, debug=True)
    

