#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code reads configuration file 'config.json' and solves a classification problem for tabular data. Makes a gridSearch and looks for best parameters for each model selected
in config file. Makes ROC curve and Precision-Recall plots for each model and save them in img/ directory.
"""
import os
import json
#from boruta import BorutaPy
import logging
import argparse
from time import time
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split, cross_validate
from sklearn.metrics import make_scorer, recall_score, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import utils as us
import time
import pylab as plt
#from stratified_group_data_splitting import StratifiedGroupKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle
#from BorutaShap import BorutaShap
import shap


if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('-j', '--jsonfile',
			dest='JSONfilename',
			help='JSON configuration file',
			metavar='FILE',
			default='config.json')

	args = parser.parse_args()

	with open(args.JSONfilename) as f:
		config_data = json.load(f)


	dir_path = os.path.dirname(os.path.realpath(__file__))
	src = os.path.basename(__file__)

	# starting time
	start = time.time()

	####### SETTING VARIABLES FROM CONFIGURATION FILE ########################
	experiment_name = config_data["experiment_name"]
	csv_filename = config_data["input_file"]
	outcome = config_data["outcome"]
	numerical_features = config_data["numerical_features"]
	cathegorical_features = config_data["cathegorical_features"]
	unwanted_features = config_data["unwanted_features"]
	models = config_data["models"]
	bshap = eval(config_data["bshap"])
	num_trials = config_data["num_trials"]
	config_imputer = eval(config_data["imputer"])
	config_scaler = eval(config_data["scaler"])
	num_splits = config_data["num_splits"]
	save = eval(config_data["save"])
	###########################################################################

	log, dir_name = us.version_control(src, dir_path, experiment_name, save)

	log.info("experiment : "  + str(experiment_name))
	log.info("models: "+ str(list(models.keys())))
	log.info("outcome: "  + str(outcome))
	log.info("Reading input data file: "  + str(csv_filename) + "...")
	log.info("Numerical features: " + str(numerical_features))
	log.info("Cathegorical features: " + str(cathegorical_features))
	log.info("Unwanted features: " + str(unwanted_features))
	log.info("boruta shap: " + str(bshap))
	log.info("Number of trials per model: " + str(num_trials))
	log.info("Number of splits for the cross validation: " + str(num_splits))
	###########################################################################

	try:
		data = pd.read_csv(csv_filename)
	except:
		data = pd.read_excel(csv_filename)

	data = pd.get_dummies(data, columns = ['Diagnosis'])
	#print(data)

	data.drop(['Diagnosis_No_Known_Disorder'], axis=1,inplace=True)
	#data = data.query('Age <= 10')
	#print(len(data))
	#groups = np.array(data["ID"])
	#print(len(groups))
	for f in unwanted_features:
		del data[f]


	data.to_csv(dir_name + '/' + "data.csv", float_format="%.3f")
	features = list(data.columns)
	if outcome in features: features.remove(outcome)


    ######################################## CATEGORICAL FEATURES #################################
	for cat in cathegorical_features:
		if cat in features: features.remove(cat)
    ###########################################################################################

	us.print_and_log('All Data input shape:' + str(data.shape) , log)
	us.print_and_log('Number of input features:' + str(len(features)), log)


	X = data[features]
	y = data[outcome]

	##################### PERFORMANCE VARIABLES ###############################################
	df_Results = pd.DataFrame(index = models, columns = ['N','train_bal_acc_mean', 'train_bal_acc_std',
								'train_roc_auc_mean', 'train_roc_auc_std',
								'train_ave_pre_mean', 'train_ave_pre_std',
								'test_bal_acc_mean' , 'test_bal_acc_std' ,
								'test_roc_auc_mean' , 'test_roc_auc_std' ,
								'test_ave_pre_mean' , 'test_ave_pre_std'] )


	mean_fpr = np.linspace(0, 1, 1000)
	tprs = []
	precs = []
	mean_recall = np.linspace(0,1,1000)
	mean_shap_values = []
	############################################################################################

	for model_name in models.keys():

		df_Results.loc[model_name, 'N'] = len(y)
		us.print_and_log("*** Model " + str(model_name),log)
		train_scores = np.zeros(num_trials)
		test_scores = np.zeros(num_trials)

		bal_acc_train_scores = np.zeros(num_trials)
		roc_auc_train_scores = np.zeros(num_trials)
		ave_pre_train_scores = np.zeros(num_trials)
		sensitivity_train_scores = np.zeros(num_trials)
		specificity_train_scores = np.zeros(num_trials)

		bal_acc_test_scores = np.zeros(num_trials)
		roc_auc_test_scores = np.zeros(num_trials)
		ave_pre_test_scores = np.zeros(num_trials)
		sensitivity_test_scores = np.zeros(num_trials)
		specificity_test_scores = np.zeros(num_trials)

		myscoring = {'bal_acc': 'balanced_accuracy',
				'roc_auc': 'roc_auc',
				'ave_pre': 'average_precision',
				'sensitivity': 'recall',
				'specificity': make_scorer(recall_score,pos_label=0)
				}

		######## MODEL SETTING ##############################
		param_grid = models[model_name]
		model = eval(model_name)
		for param in models[model_name]:
			param_grid[param] = eval(param_grid[param])
		#####################################################

		for i in range(num_trials):

			X = data[features]
			y = data[outcome]
			X = config_imputer.fit_transform(X)
			y = np.array(y)

			#pipe = Pipeline(steps=[('imputer',config_imputer), ('classifier', model)])
			pipe = Pipeline(steps=[('classifier', model)])

			us.print_and_log("Iteration:" + str(i),log)

			inner_cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=i)
			outer_cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=i)
			#outer_cv = LeaveOneOut()

			#pipe = Pipeline(steps=[('imputer', imputer),  ('scaler', scaler), ('classifier', model)])
			clf = GridSearchCV(pipe, param_grid=param_grid,cv=inner_cv, refit=True, scoring='roc_auc')
			# Take a look at the best params

			nested_score = cross_validate(clf, X=X, y=y, cv=outer_cv, return_train_score=True, return_estimator=True, scoring = myscoring)


			bal_acc_train_scores[i] = np.mean(nested_score['train_bal_acc'])
			roc_auc_train_scores[i] = np.mean(nested_score['train_roc_auc'])
			ave_pre_train_scores[i] = np.mean(nested_score['train_ave_pre'])

			us.print_and_log('Train: bal_acc ' + str( bal_acc_train_scores[i]) ,log)
			us.print_and_log('Train: roc_auc ' + str(roc_auc_train_scores[i]) ,log)
			us.print_and_log('Train: ave_pre ' + str(ave_pre_train_scores[i]),log)

			bal_acc_test_scores[i] = np.mean(nested_score['test_bal_acc'])
			roc_auc_test_scores[i] = np.mean(nested_score['test_roc_auc'])
			ave_pre_test_scores[i] = np.mean(nested_score['test_ave_pre'])

			us.print_and_log('Test: bal_acc ' + str(bal_acc_test_scores[i]),log)
			us.print_and_log('Test: roc_auc ' + str(roc_auc_test_scores[i]),log)
			us.print_and_log('Test: ave_pre ' + str(ave_pre_test_scores[i]),log)

			X = data[features]
			y = data[outcome]
			X = config_imputer.fit_transform(X)
			y = np.array(y)

			############# SHAP VALUES COMPUTATION FOR EACH FOLD ##################
			iter_shap = 0
			list_shap_values = []
			list_test_sets = []
			shuffled_X = pd.DataFrame(columns = features)
			for train_index, test_index in inner_cv.split(X, y):
				print("Split:", iter_shap)

				## TRUE POSITIVE RATE COMPUTATION FOR EACH OUTER LOOP (TEST SET)
				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = y[train_index], y[test_index]
				X_train = pd.DataFrame(X_train,columns=features)
				X_test = pd.DataFrame(X_test,columns=features)

				classifier_shap = nested_score['estimator'][iter_shap].best_estimator_["classifier"]
				#classifier_shap = XGBClassifier(use_label_encoder=False)
				classifier_shap.fit(X_train, y_train)

				#classifier_shap = clf
				#classifier_shap.fit(X_train, y_train,groups[train_index])
				#model = classifier_shap.best_estimator_["classifier"]
				##################################  ROC AREA #################

				y_pred = classifier_shap.predict_proba(X_test)[:, 1]

				fpr, tpr, thresholds = roc_curve(y_test, y_pred)
				roc_auc = auc(fpr, tpr)
				interp_tpr = np.interp(mean_fpr, fpr, tpr)
				interp_tpr[0] = 0.0
				tprs.append(interp_tpr)
				#print("roc auc: ", roc_auc)
				#print("roc_auc_score: ",roc_auc_score(y_test, y_pred))
				#print("nested_score auc: ", nested_score['test_roc_auc'][iter_shap])

				explainer = shap.TreeExplainer(classifier_shap)

				list_shap_values.append(explainer.shap_values(X_test))
				shuffled_X = shuffled_X.append(X_test)

				iter_shap += 1
			#######################################################################

			array_shap_values = list_shap_values[0]
			for i in range(1,len(list_shap_values)):
				array_shap_values = np.append(array_shap_values,list_shap_values[i],axis=0)


			tmp_tpr, tmp_prec = us.procedure_validation(outer_cv, X, y,nested_score,tprs,precs, mean_fpr, mean_recall)
			tprs += tmp_tpr
			precs += tmp_prec
			#mean_shap_values.append(array_shap_values)


		#tmp = mean_shap_values[0]
		#for i in range(1, num_trials):
		#		tmp += mean_shap_values[i]
		#mean_shap_values = tmp/num_trials
		########################################################
		print("############################## PLOTTING")
		######### PLOTTING AND SAVING ROC CURVE #############
		#us.plotting(mean_fpr,tprs,x_label="False Positive Rate",y_label = "True Positive Rate",num_trials=num_trials, num_splits=num_splits,file_name = dir_name + "/img/"+ model_name + "_ROCcurve.png",random_classifier=True)
		plt.plot([0, 1], [0, 1], '--', color='r', label='Random classifier', lw=2, alpha=0.8)
		mean_tpr = np.mean(tprs, axis=0)
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		plt.title('AUC=%0.3f' % mean_auc)
		plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC', lw=2, alpha=0.8)

		## Standard deviation computation
		std_tpr = np.std(tprs, axis=0)
		tprs_upper_std = np.minimum(mean_tpr + std_tpr, 1)
		tprs_lower_std = np.maximum(mean_tpr - std_tpr, 0)
		plt.fill_between(mean_fpr, tprs_lower_std, tprs_upper_std, color='green', alpha=.2,label=r'$\pm$ 1 SD')

		## 99.9% CI computation
		z = 3.291
		SE = std_tpr / np.sqrt(num_trials * num_splits)
		tprs_upper_95CI = mean_tpr + (z * SE)
		tprs_lower_95CI = mean_tpr - (z * SE)
		plt.fill_between(mean_fpr, tprs_lower_95CI, tprs_upper_95CI, color='grey', alpha=.5,label=r'$\pm$ 99.9% CI')

		plt.xlim([-0.05, 1.05])
		plt.ylim([-0.05, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend(loc="lower right")
		#plt.axis('square')
		plt.savefig(dir_name + "/img/"+model_name+"_ROCcurve.png", dpi=600)

		######## PLOTTING PRECISION-RECALL CURVE ##################
		us.plotting(mean_recall,precs,x_label="Recall",y_label="Precision",num_trials=num_trials, num_splits=num_splits,file_name = dir_name + "/img/"+model_name+"_PRcurve.png",random_classifier=False)
		us.print_and_log('Plots saved in ' + str(dir_name) + "/img/",log)
		##########################################################

		############ RESULTS TABLE ###############################
		quantities = {}
		quantities_names = ["train_bal_acc","train_roc_auc","train_ave_pre","test_bal_acc","test_roc_auc","test_ave_pre"]
		for i, j in zip(quantities_names,[bal_acc_train_scores,roc_auc_train_scores,ave_pre_train_scores,bal_acc_test_scores,roc_auc_test_scores,ave_pre_test_scores]):
			quantities[i] = us.mean_std(j)

		for i, q in enumerate(list(quantities.keys())):
			df = us.results_to_df(df_Results,model_name,quantities[q],q)
		###########################################################

	us.print_and_log('*** Average performance',log)

	for i in quantities.keys():
		us.print_and_log(i + "(mean " + str(quantities[i][0])+ ", std "+ str(quantities[i][1])+ ")",log)

	df.to_csv(dir_name + '/' + "ResultsTable.csv", float_format="%.3f")

	# end time
	end = time.time()
	# total time taken

	us.print_and_log(f"Runtime is {end - start}",log)
	end_time = datetime.now()
	#log.info('Ended at %s', end_time)
	log.info('End')

	### START FINAL CLASSIFIER ########################

	us.print_and_log('Training the final model...',log)
	X = data[features]
	y = data[outcome]

	inner_cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=2)
	clf_final = GridSearchCV(pipe, param_grid=param_grid, cv=inner_cv, refit=True, scoring='roc_auc').fit(X, y)

	weights = clf_final.best_estimator_
	log.info(clf_final)

	########## SAVE THE MODEL TO DISK ###############

	filename = dir_name + '/' + 'model.pkl'
	pickle.dump(clf_final, open(filename, 'wb'))
	pickle.dump(mean_shap_values, open(dir_name + '/' + "shap_values.pkl",'wb'))
	pickle.dump(shuffled_X, open(dir_name + '/' + "test_set.pkl","wb"))
	logging.shutdown()
