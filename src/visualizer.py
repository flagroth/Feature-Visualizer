
# coding: utf-8

# In[1]:

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE

import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, gridplot
from bokeh.charts import Scatter
from bokeh.models import ColumnDataSource,HoverTool


# In[ ]:

output_notebook()


# In[2]:

def data_prep():
    dataset = pd.read_csv("../data/dataset_attributes.csv")
    dataset = dataset.apply(pd.to_numeric, errors='ignore')
    y = np.empty(len(dataset['Overall']), dtype='object')
    x = dataset.drop(['ID','ID.1','Name','Overall','Photo'], axis = 1)
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
    return ({"train_x":train_x,"train_y":train_y,"train_names":train_names,"train_photo":train_photo,
             "test_x":test_x,"test_y":test_y,"test_names":test_names,"test_photo":test_photo})


# In[ ]:

def make_plot(source_points, compo, columns, method):
    colors = ['navy', 'turquoise', 'darkorange']
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
    if method=='':
        return 1
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
    color = []
    for i in train_y.astype(int):
        color.append(colors[i])        
    p.circle(x, y , source = source_points, color = color, alpha=.8)#X_r[y == i, 0], X_r[y == i, 1], color=color)
    #p.line(compo_x, compo_y, source = source_lines, color = 'red')
    for a,b,label in zip(compo[0,:],compo[1,:],columns):
        p.line([0,a],[0,b], color = 'red')
        p.text([a],[b],text=[label], text_align="center")
    return(p)


# In[ ]:

if __name__ == '__main__':
    data=data_prep()
    train_x,train_y,train_names,train_photo,test_x,test_y,test_names,test_photo = data.values()
    
    #Making PCA
    pca = PCA(n_components=2)
    X_r_pca = pca.fit(train_x).transform(train_x)
    comp_pca = pca.components_
    X_r_test = pca.transform(test_x)
    
    #Making LDA
    lda = LDA(n_components=2)
    X_r_lda = lda.fit(train_x,train_y).transform(train_x)
    comp_lda = lda.coef_
    y_predict_lda = lda.predict(test_x)
    X_r_lda_test = lda.fit(test_x,y_predict).transform(test_x)
    
    #Making t-SNE
    tsne = TSNE(n_components=2)
    X_r_tsne = tsne.fit_transform(train_x)
    
    source_points = ColumnDataSource(data=dict(
        x_pca=X_r_pca[:,0], y_pca=X_r_pca[:,1], 
        x_lda=X_r_lda[:,0], y_lda=X_r_lda[:,1],
        desc=train_names,
        imgs=train_photo
    ))
    
    g0 = make_plot(source_points, comp_pca, train_x.columns,"PCA")
    g1 = make_plot(source_points, comp_lda, train_x.columns,"LDA")
    p = gridplot([[g0,g1]])
    show(p)
    #return 0


# In[4]:

data=data_prep()
train_x,train_y,train_names,train_photo,test_x,test_y,test_names,test_photo = data.values()


# In[ ]:

#Making t-SNE
tsne = TSNE(n_components=2,n_iter=250)
X_r_tsne = tsne.fit_transform(train_x)


# In[8]:

train_x.shape


# In[ ]:



