# coding: utf-8
import os
from flask import Flask, render_template, request, redirect, url_for

from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.random_projection import GaussianRandomProjection as GRP

from sklearn.svm import SVC

import pandas as pd
import numpy as np
from bokeh.io import output_notebook, show
from bokeh.plotting import figure, gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import components
from bokeh.palettes import Category20

from werkzeug.utils import secure_filename

from flask import send_from_directory

import numpy as np

UPLOAD_FOLDER = '../data'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
_filename="kk_filename"
_target="kk_target"
_separator=","
_train_pct=70
_sampling=100
viz_names=["PCA","LDA","RANDOM","TSNE","MDS"]

# Filename checker
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Data preparation
def data_prep(filename=_filename, target=_target, separator=_separator, train_pct=70, sampling=100, features=None):
	global _dataset 
	_dataset = pd.read_csv("../data/"+filename, sep=separator)
	
	if sampling==100:
		dataset=_dataset
	else:
		dataset = _dataset.groupby(target,group_keys=False).apply(lambda x:x.sample(min(len(x),int(len(x)*(sampling)/100))))
		print("Original ", _dataset.shape, "Sampled", dataset.shape)
	
	
	dataset = dataset.apply(pd.to_numeric, errors='coerce')
	print(dataset.columns)
	y = dataset[target]

	dataset=dataset.drop(target, axis=1)
	data_columns = dataset.columns
	
	if features is not None:
		x = dataset[features]
	else:
		x = dataset
		
	x = x.apply(pd.to_numeric, errors='coerce')
	x = (x - x.mean()) / (x.max() - x.min())
	msk = np.random.rand(len(x)) < train_pct/100
	
	train_x = x[msk]
	train_y = y[msk]
	test_x = x[~msk]
	test_y = y[~msk]
	
	return ({"data_columns":data_columns,
			"train_x":train_x,
			"train_y":train_y,
			"test_x":test_x,
			"test_y":test_y
			})
	

# Plot drawing
def make_plot(source_points, source_points_test, compo=None, columns=None, method=None, accuracy=0):  
	if method in ['PCA','LDA','RANDOM']:
		p = figure(tools = ["save,pan,wheel_zoom,box_select,lasso_select,reset"], width = 200, height = 200, title=method + ". Acc: " + str(accuracy), sizing_mode='scale_width')
	else:
		p = figure(tools = ["save,pan,wheel_zoom,box_select,lasso_select,reset"], width = 200, height = 200, title=method, sizing_mode='scale_width')
	x = 'x_' + method.lower()
	y = 'y_' + method.lower()
	t_x = 'x_test_' + method.lower()
	t_y = 'y_test_' + method.lower()
	compo_x = 'comp_' + method.lower() + '_x'
	compo_y = 'comp_' + method.lower() + '_y'
	
	p.circle(x=x, y=y , source = source_points, alpha=.8, fill_color='colors_fill', line_color='colors_line')
	if method in ['PCA','LDA','RANDOM']:	
		p.circle(x=t_x, y=t_y, source = source_points_test, alpha=.8, fill_color='colors_'+method.lower()+"_fill", line_color='colors_'+method.lower()+"_line", size=8)
		# In linear methods we can show components
		for a,b,label in zip(compo[0,:],compo[1,:],columns):
			p.line([0,a],[0,b], color = 'red')
			p.text([a],[b],text=[label], text_align="center")
	return(p)

# Upload file							   
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			print('No file part')
			return redirect(request.url)
		file = request.files['file']
		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			print('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			target = request.form.get("target")
			separator = request.form.get("separator")
			train_pct = request.form.get("train_pct")
			sampling = request.form.get("sampling")
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('features_page', filename=filename, target=target, separator=separator, train_pct=train_pct, sampling=sampling))
	return '''
	<!doctype html>
	<link rel=stylesheet type=text/css href="static/cerulean.min.css">
	<title>Feature visualizer</title>
	<h1>Upload new File</h1>
	<div class=""panel panel-default"">
		<div class="panel-body">
			<form method=post enctype=multipart/form-data>
				<p><input type=file name=file>*Only CSV</p>
				<p>Target column:<input type=text name=target>
				CSV separator<input type=text name=separator></p>
				<p>Training percentage: <input type=text name=train_pct></p>
				<p>Stratified sampling pct: <input type=text name=sampling></p> 
				<input type=submit value=Upload>
			</form>
		</div>
	</div>
	'''

# Feature selection	
@app.route('/features_page?filename=<filename>&target=<target>&separator=<separator>&train_pct=<train_pct>&sampling=<sampling>', methods=['GET'])
def features_page(filename,target,separator,train_pct,sampling):
	global _filename,_target, _separator, _train_pct, _sampling
	_filename=filename
	_target=target
	_separator=separator
	_train_pct=int(train_pct)
	_sampling=int(sampling)

	data=data_prep(_filename,_target,_separator,_train_pct,_sampling)

	data_columns,train_x,train_y,test_x,test_y = data.values()
	return render_template("visualizer.html", feature_names=train_x.columns, viz_names=viz_names, filename=filename, target=target)

# Visualizations	
@app.route('/visualize', methods=['POST'])
def visualize():
	features = request.form.getlist('feature')
	if len(features)==0:
		features=None
	data=data_prep(_filename, _target, _separator, _train_pct, _sampling, features)
	
	dataSource=dict()
	dataSource_test=dict()
	
	data_columns,train_x,train_y,test_x,test_y = data.values()
	
	vizs = request.form.getlist('viz')
	print(vizs)
	color = Category20[20]
	colors_fill = []
	colors_line = []
	for i in train_y.astype(int):
		colors_fill.append(color[i]) 
		colors_line.append(color[i]) 
		
	dataSource['colors_fill']=colors_fill
	dataSource['colors_line']=colors_line
	clf = SVC(kernel = 'linear')
	if "PCA" in vizs:
		print("Making PCA")
		pca = PCA(n_components=2)
		X_r_pca = pca.fit(train_x).transform(train_x)
		comp_pca = pca.components_
		X_r_pca_test = pca.transform(test_x)
		clf.fit(X_r_pca,train_y)
		y_predict_pca = clf.predict(X_r_pca_test)
		acc_pca = accuracy_score(test_y, y_predict_pca)
		dataSource['x_pca'] = X_r_pca[:,0]
		dataSource['y_pca'] = X_r_pca[:,1]
		dataSource_test['x_test_pca'] = X_r_pca_test[:,0]
		dataSource_test['y_test_pca'] = X_r_pca_test[:,1]
		colors_pca_fill = []
		colors_pca_line = []
		for i in test_y.astype(int):
			colors_pca_fill.append(color[i])
		for i in y_predict_pca.astype(int):
			colors_pca_line.append(color[i])
			
		dataSource_test['colors_pca_fill'] = colors_pca_fill
		dataSource_test['colors_pca_line'] = colors_pca_line
	
	if "LDA" in vizs:
		print("Making LDA")
		lda = LDA(n_components=2)
		X_r_lda = lda.fit(train_x,train_y).transform(train_x)
		comp_lda = lda.coef_
		y_predict_lda = lda.predict(test_x)
		acc_lda = accuracy_score(test_y, y_predict_lda)
		X_r_lda_test = lda.fit(test_x,y_predict_lda).transform(test_x)
		dataSource['x_lda'] = X_r_lda[:,0]
		dataSource['y_lda'] = X_r_lda[:,1]
		dataSource_test['x_test_lda'] = X_r_lda_test[:,0]
		dataSource_test['y_test_lda'] = X_r_lda_test[:,1]
		colors_lda_fill = []
		colors_lda_line = []
		for i in test_y.astype(int):
			colors_lda_fill.append(color[i])
		for i in y_predict_lda.astype(int):
			colors_lda_line.append(color[i])
			
		dataSource_test['colors_lda_fill'] = colors_lda_fill
		dataSource_test['colors_lda_line'] = colors_lda_line

	if "RANDOM" in vizs:
		print("Making RANDOM")
		randomProjection = GRP(n_components=2)
		randomProjection.fit(train_x,train_y)
		X_r_random = randomProjection.transform(train_x)
		comp_random = randomProjection.components_
		X_r_random_test = randomProjection.transform(test_x)
		clf.fit(X_r_random,train_y)
		y_predict_random = clf.predict(X_r_random_test)
		acc_random = accuracy_score(test_y, y_predict_random)	   
		dataSource['x_random'] = X_r_random[:,0]
		dataSource['y_random'] = X_r_random[:,1]
		dataSource_test['x_test_random'] = X_r_random_test[:,0]
		dataSource_test['y_test_random'] = X_r_random_test[:,1]
		colors_random_fill = []
		colors_random_line = []
		for i in test_y.astype(int):
			colors_random_fill.append(color[i])
		for i in y_predict_random.astype(int):
			colors_random_line.append(color[i])
			
		dataSource_test['colors_random_fill'] = colors_random_fill
		dataSource_test['colors_random_line'] = colors_random_line
		
	
	if "TSNE" in vizs:
		print("Making t-SNE")
		tsne_iter = int(request.form.get('tsne_iter'))
		tsne = TSNE(n_components=2,n_iter=tsne_iter)
		X_r_tsne = tsne.fit_transform(train_x)
#		X_r_tsne_test = tsne.fit_transform(test_x)
#		clf.fit(X_r_tsne,train_y)
#		y_predict_tsne = clf.predict(X_r_tsne_test)
#		acc_tsne = accuracy_score(test_y, y_predict_tsne)	 
		dataSource['x_tsne'] = X_r_tsne[:,0]
		dataSource['y_tsne'] = X_r_tsne[:,1]

	if "MDS" in vizs:
		print("Making MDS")
		mds_iter = int(request.form.get('mds_iter'))
		mds = MDS(n_components=2,max_iter=mds_iter)
		X_r_mds = mds.fit_transform(train_x)
		dataSource['x_mds'] = X_r_mds[:,0]
		dataSource['y_mds'] = X_r_mds[:,1]


	source_points = ColumnDataSource(data=dataSource)
	source_points_test = ColumnDataSource(data=dataSource_test)
	   
	g=[]
	print("Plotting")
	if "PCA" in vizs:
		g.append(make_plot(source_points, source_points_test, comp_pca, train_x.columns, "PCA", acc_pca))
	if "LDA" in vizs:
		g.append(make_plot(source_points, source_points_test, comp_lda, train_x.columns, "LDA", acc_lda))
	if "RANDOM" in vizs:	
		g.append(make_plot(source_points, source_points_test, comp_random, train_x.columns, "RANDOM", acc_random))		
	if "TSNE" in vizs:	
		g.append(make_plot(source_points, source_points_test, method="TSNE", accuracy = 0))
	if "MDS" in vizs:	
		g.append(make_plot(source_points, source_points_test, method="MDS", accuracy = 0))
	
	p = gridplot(g, ncols=2, plot_width=200, plot_height=200, sizing_mode='scale_width', merge_tools = False)
	script, div = components(p)
	return render_template("visualizer.html", feature_names=data_columns, selected_features=features, viz_names=viz_names, selected_viz=vizs, script=script, div=div)
	
	
if __name__ == '__main__':
	app.run('10.71.176.149',port=5000, debug=True)
	

