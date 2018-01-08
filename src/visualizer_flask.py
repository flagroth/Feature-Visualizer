
# coding: utf-8
from flask import Flask, render_template, request

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

import pandas as pd
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, gridplot
from bokeh.charts import Scatter
from bokeh.models import ColumnDataSource,HoverTool
from bokeh.embed import components

app = Flask(__name__)

def data_prep(features=None):
    dataset = pd.read_csv("../data/dataset_attributes.csv")
    dataset = dataset.apply(pd.to_numeric, errors='ignore')
    y = np.empty(len(dataset['Overall']), dtype='object')
    x = dataset.drop(['ID','ID.1','Name','Overall','Photo'], axis = 1)
    data_columns = x.columns
    if features is not None:
        x = x[features]
    y[dataset['Overall']> 85] = '0'
    y[dataset['Overall']<=85] = '1'
    y[dataset['Overall']<=65] = '2'
    x = x.apply(pd.to_numeric, errors='ignore')
    x = (x - x.mean()) / (x.max() - x.min())
    msk = np.random.rand(len(x)) < 0.7
    train_x = x[msk]
    train_y = y[msk]
    test_x = x[~msk]
    test_y = y[~msk]
    train_names = dataset['Name'][msk]
    train_photo = dataset['Photo'][msk]
    test_names = dataset['Name'][~msk]
    test_photo = dataset['Photo'][~msk]
    return ({"data_columns":data_columns,"train_x":train_x,"train_y":train_y,"train_names":train_names,"train_photo":train_photo,
             "test_x":test_x,"test_y":test_y,"test_names":test_names,"test_photo":test_photo})

def make_plot(source_points, compo=None, columns=None, colors=None, method=None):
    target_names = ['Bueno','Regular','Malo']
    
    hover = HoverTool(tooltips=""""
    <div>
        <div>
            <img
                src="@imgs" alt="@imgs"
            ></img>
        </div>
         <div>
            <span style="font-size: 17px; font-weight: bold;">@desc</span>
            <span style="font-size: 15px; color: #966;">[$index]</span>
        </div>
    """)
    
    p = figure(tools = [hover,"pan,wheel_zoom,box_select,lasso_select,reset"], width = 300, height = 300, title=method+" FIFA 18")
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
               
    p.circle(x, y , source = source_points, alpha=.8, color=colors)#X_r[y == i, 0], X_r[y == i, 1], color=color)
    #p.line(compo_x, compo_y, source = source_lines, color = 'red')
    if method in ['PCA','LDA']:
        for a,b,label in zip(compo[0,:],compo[1,:],columns):
            p.line([0,a],[0,b], color = 'red')
            p.text([a],[b],text=[label], text_align="center")
    return(p)
    
@app.route('/',methods=['GET'])
def index():
    data=data_prep()
    data_columns,train_x,train_y,train_names,train_photo,test_x,test_y,test_names,test_photo = data.values()
    return render_template("tfm.html", feature_names=train_x.columns)
    
@app.route('/',methods=['POST'])
def visualize():
    features = request.form.getlist('feature')
    if len(features)==0:
        features=None
    data=data_prep(features)
    data_columns,train_x,train_y,train_names,train_photo,test_x,test_y,test_names,test_photo = data.values()
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
        tsne = TSNE(n_components=2)
        X_r_tsne = tsne.fit_transform(train_x)
        dataSource['x_tsne'] = X_r_tsne[:,0]
        dataSource['y_tsne'] = X_r_tsne[:,1]
    
    print("Colouring")
    color = ['navy', 'turquoise', 'darkorange']
    colors = []
    for i in train_y.astype(int):
        colors.append(color[i]) 
    
    dataSource['desc']=train_names
    dataSource['imgs']=train_photo
    
    source_points = ColumnDataSource(data=dataSource)
  
    g=[]
    print("Plotting")
    if "PCA" in vizs:
        g.append(make_plot(source_points, comp_pca, train_x.columns, colors, "PCA"))
    if "LDA" in vizs:
        g.append(make_plot(source_points, comp_lda, train_x.columns, colors, "LDA"))
    if "TSNE" in vizs:    
        g.append(make_plot(source_points, colors=colors, method="TSNE"))
    p = gridplot([g])
    script, div = components(p)
    return render_template("tfm.html",feature_names=data_columns, script=script, div=div)
    
    
if __name__ == '__main__':
    app.run('10.71.184.225',port=5000, debug=True)
    

