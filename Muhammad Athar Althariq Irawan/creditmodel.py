import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# binary classification spot check script
import warnings
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report


feather_filename = "loan_data_2007_2014.feather"
if not os.path.exists(feather_filename):
    # Baca melalui csv
    df = pd.read_csv("loan_data_2007_2014.csv")

    # Ubah ke feather
    df.to_feather("loan_data_2007_2014.feather")
    
rawdf = pd.read_feather(feather_filename)

#Cek duplikat
duplicate_data = len(rawdf[rawdf.duplicated(keep=False)])
print(f'Jumlah duplikat = {duplicate_data}')

df = rawdf.copy()

#membuang kolom yang tidak digunakan
drop = [
    'Unnamed: 0', 'id', 'member_id', 'annual_inc_joint', 'dti_joint', 'verification_status_joint', 'open_acc_6m',
    'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m',
    'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'url', 'desc'
]

df.drop(drop, axis=1, inplace=True)

print(f'Jumlah Kolom = {len(df.columns)}')

df['loan_status'].value_counts()

def loan_status_transform(data, col):
    data[col] = data[col].str.replace('Late (31-120 days)', '1_bad_loan', regex=False)
    data[col] = data[col].str.replace('Late (16-30 days)', '1_bad_loan', regex=False)
    data[col] = data[col].str.replace('In Grace Period', '1_bad_loan', regex=False)
    data[col] = data[col].str.replace('Does not meet the credit policy. Status:Fully Paid', '0_good_loan', regex=False)
    data[col] = data[col].str.replace('Does not meet the credit policy. Status:Charged Off', '1_bad_loan', regex=False)
    data[col] = data[col].str.replace('Current', '0_good_loan', regex=False)
    data[col] = data[col].str.replace('Fully Paid', '0_good_loan', regex=False)
    data[col] = data[col].str.replace('Charged Off', '1_bad_loan', regex=False)
    data[col] = data[col].str.replace('Default', '1_bad_loan', regex=False)
    return data[col]

df['loan_status'] = loan_status_transform(df, 'loan_status')

plt.figure(figsize=(7,5))
ax = sns.countplot(x='loan_status', data=df)
plt.show()

df['loan_status'].value_counts(normalize=True)

cat_df = df.select_dtypes(include='object')
num_df = df.select_dtypes(exclude='object')

cat_df.isnull().mean()

cat_df.columns

plt.figure(figsize=(7,5))
ax = sns.countplot(x='term', data=df, hue='loan_status')
plt.show()

term36 = 35558/(35558+305395)*100
term60 = 19774/(19774+108558)*100
print(f'Term 36 = {round(term36, 2)}% bad loan')
print(f'Term 60 = {round(term60, 2)}% bad loan')

plt.figure(figsize=(7,5))
sns.countplot(x='grade', data=df.sort_values('grade'), hue='loan_status')
plt.show()

df['emp_title'] = df['emp_title'].str.lower()
plt.figure(figsize=(7,5))
sns.countplot(y='emp_title', data=df, order=df['emp_title'].value_counts()[:15].index, hue='loan_status')
plt.show()

df['emp_length'].value_counts()

def transform_home_ownership(data=df['home_ownership']):
    data = data.str.replace('NONE', 'OTHER')
    data = data.str.replace('ANY', 'OTHER')
    return data
df['home_ownership'] = transform_home_ownership()
df['home_ownership'].value_counts()


plt.figure(figsize=(7,5))
sns.countplot(x='home_ownership', data=df, hue='loan_status')
plt.show()

df.groupby(['home_ownership', 'loan_status']).size()

mortgage = 24708/(24708+211167)*100
other = 46/(46+187)*100
own = 4952/(4952+36752)*100
rent = 25626/(25626+162847)*100

print(f'Pinjaman Buruk Mortgage = {round(mortgage, 2)}%')
print(f'Pinjaman Buruk Other = {round(other, 2)}%')
print(f'Pinjaman Buruk Own = {round(own, 2)}%')
print(f'Pinjaman Buruk Rent = {round(rent, 2)}%')

#Format tanggal ke tipe date
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%y').dt.to_period('M')
df['issue_d'][:5]

date_data = df.groupby(['loan_status', 'issue_d']).size().reset_index().pivot(index='issue_d', columns='loan_status', values=0)
date_data.plot(kind='line', figsize=(8,5))
plt.show()

df['issue_m'] = df['issue_d'].dt.month
df['issue_m']

plt.figure(figsize=(7,5))
sns.countplot(x='issue_m', data=df, hue='loan_status')
plt.show()

df['purpose'].value_counts()

plt.figure(figsize=(10,8))
sns.countplot(y='purpose', data=df, hue='loan_status')
plt.show()

plt.figure(figsize=(7,5))
sns.countplot(x='addr_state', data=df, order=df['addr_state'].value_counts()[:15].index, hue='loan_status')
plt.show()


plt.figure(figsize=(7,5))
sns.countplot(x='initial_list_status', data=df, hue='loan_status')
plt.show()

#heatmap
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), square=True, cmap='coolwarm')
plt.show()

for x in num_df.columns:
    print(x, '=', len(num_df[x].unique()))

num_df = num_df
num_df.hist(bins=50, layout=(11,3), figsize=(15,30))
plt.show()

## Missing values numerical

plt.figure(figsize=(10,8))
msno.matrix(df.select_dtypes(exclude='object'))
plt.show()

drop = [
    'sub_grade', 'emp_title', 'issue_d', 'pymnt_plan',
    'title', 'zip_code', 'earliest_cr_line', 'last_pymnt_d',
    'next_pymnt_d', 'last_credit_pull_d', 'application_type',
    'funded_amnt', 'funded_amnt_inv', 'installment', 
    'out_prncp_inv', 'total_pymnt_inv', 'total_rec_prncp',
    'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog'
]

df.drop(drop, axis=1, inplace=True)

df_sample = df.head(n=10000)

X = df_sample.drop('loan_status', axis=1).copy()
y = df_sample['loan_status'].copy()


le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

numeric_transformer = Pipeline([
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
])

categoric_transformer = Pipeline([
                                  ('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numeric_features = [
                    'loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs',
                    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                    'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_int',
                    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                    'last_pymnt_amnt', 'collections_12_mths_ex_med', 'acc_now_delinq',
                    'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'issue_m'
]

categoric_features = [col for col in X.select_dtypes(include='object').columns]

preprocessor = ColumnTransformer([
    ('numeric', numeric_transformer, numeric_features),
    ('categoric', categoric_transformer, categoric_features)
])
from sklearn.datasets import make_classification

 # load the dataset, returns X and y elements
def load_dataset():
	return make_classification(n_samples=1000, n_classes=2, random_state=1)
 
# create a dict of standard models to evaluate {name:object}
def define_models(models=dict()):
	# linear models
	models['logistic'] = LogisticRegression()
	alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for a in alpha:
		models['ridge-'+str(a)] = RidgeClassifier(alpha=a)
	models['sgd'] = SGDClassifier(max_iter=1000, tol=1e-3)
	models['pa'] = PassiveAggressiveClassifier(max_iter=1000, tol=1e-3)
	# non-linear models
	n_neighbors = range(1, 21)
	for k in n_neighbors:
		models['knn-'+str(k)] = KNeighborsClassifier(n_neighbors=k)
	models['cart'] = DecisionTreeClassifier()
	models['extra'] = ExtraTreeClassifier()
	models['svml'] = SVC(kernel='linear')
	models['svmp'] = SVC(kernel='poly')
	c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	for c in c_values:
		models['svmr'+str(c)] = SVC(C=c)
	models['bayes'] = GaussianNB()
	# ensemble models
	n_trees = 100
	models['ada'] = AdaBoostClassifier(n_estimators=n_trees)
	models['bag'] = BaggingClassifier(n_estimators=n_trees)
	models['rf'] = RandomForestClassifier(n_estimators=n_trees)
	models['et'] = ExtraTreesClassifier(n_estimators=n_trees)
	models['gbm'] = GradientBoostingClassifier(n_estimators=n_trees)
	print('Defined %d models' % len(models))
	return models
 
# define gradient boosting models
def define_gbm_models(models=dict(), use_xgb=True):
	# define config ranges
	rates = [0.001, 0.01, 0.1]
	trees = [50, 100]
	ss = [0.5, 0.7, 1.0]
	depth = [3, 7, 9]
	# add configurations
	for l in rates:
		for e in trees:
			for s in ss:
				for d in depth:
					cfg = [l, e, s, d]
					if use_xgb:
						name = 'xgb-' + str(cfg)
						models[name] = XGBClassifier(learning_rate=l, n_estimators=e, subsample=s, max_depth=d)
					else:
						name = 'gbm-' + str(cfg)
						models[name] = GradientBoostingClassifier(learning_rate=l, n_estimators=e, subsample=s, max_depth=d)
	print('Defined %d models' % len(models))
	return models
 
# create a feature preparation pipeline for a model
def make_pipeline(model):
  steps = list()
  steps.append(('preprocessor', preprocessor))
  steps.append(('model', model))
  pipeline = Pipeline(steps=steps)
  return pipeline
 
# evaluate a single model
def evaluate_model(X, y, model, folds, metric):
	# create the pipeline
	pipeline = make_pipeline(model)
	# evaluate model
	scores = cross_val_score(pipeline, X, y, scoring=metric, cv=folds, n_jobs=-1)
	return scores
 
# evaluate a model and try to trap errors and and hide warnings
def robust_evaluate_model(X, y, model, folds, metric):
	scores = None
	try:
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore")
			scores = evaluate_model(X, y, model, folds, metric)
	except:
		scores = None
	return scores
 
# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(X, y, models, folds=10, metric='f1'):
	results = dict()
	for name, model in models.items():
		# evaluate the model
		scores = robust_evaluate_model(X, y, model, folds, metric)
		# show process
		if scores is not None:
			# store a result
			results[name] = scores
			mean_score, std_score = mean(scores), std(scores)
			print('>%s: %.3f (+/-%.3f)' % (name, mean_score, std_score))
		else:
			print('>%s: error' % name)
	return results
 
# print and plot the top n results
def summarize_results(results, maximize=True, top_n=10):
	# check for no results
	if len(results) == 0:
		print('no results')
		return
	# determine how many results to summarize
	n = min(top_n, len(results))
	# create a list of (name, mean(scores)) tuples
	mean_scores = [(k,mean(v)) for k,v in results.items()]
	# sort tuples by mean score
	mean_scores = sorted(mean_scores, key=lambda x: x[1])
	# reverse for descending order (e.g. for accuracy)
	if maximize:
		mean_scores = list(reversed(mean_scores))
	# retrieve the top n for summarization
	names = [x[0] for x in mean_scores[:n]]
	scores = [results[x[0]] for x in mean_scores[:n]]
	# print the top n
	print()
	for i in range(n):
		name = names[i]
		mean_score, std_score = mean(results[name]), std(results[name])
		print('Rank=%d, Name=%s, Score=%.3f (+/- %.3f)' % (i+1, name, mean_score, std_score))
	# boxplot for the top n
	plt.boxplot(scores, labels=names)
	_, labels = plt.xticks()
	plt.setp(labels, rotation=90)
	plt.savefig('spotcheck.png')
 
# get model list
models = define_models()
# add gbm models
models = define_gbm_models(models)
# evaluate models
results = evaluate_models(X_train, y_train, models)
# summarize results
summarize_results(results)

X = df.drop('loan_status', axis=1).copy()
y = df['loan_status'].copy()

y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

xgb_pipeline = Pipeline([
                         ('preprocessor', preprocessor),
                         ('xgb_clf', XGBClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, max_depth=9))
])

xgb_cv = cross_val_score(xgb_pipeline, X_train, y_train, scoring='f1', cv=3)

xgb_model = xgb_pipeline.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

f1 = f1_score(y_test, y_pred)

print(f'Train Score = {np.mean(xgb_cv)}')
print(f'Test Score = {f1}')


plot_confusion_matrix(xgb_model, X_test, y_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
numeric = df.select_dtypes(exclude='object').copy()

X = numeric.copy()
y = df['loan_status'].copy()

y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

numeric_transformer = Pipeline([
                                ('imputer', SimpleImputer(strategy='median')),
                                ('scaler', StandardScaler())
])

numeric_features = [
                    'loan_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs',
                    'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util',
                    'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_int',
                    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
                    'last_pymnt_amnt', 'collections_12_mths_ex_med', 'acc_now_delinq',
                    'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim', 'issue_m'
]

preprocessor = ColumnTransformer([
    ('numeric', numeric_transformer, numeric_features)
])

xgb_pipeline = Pipeline([
                         ('preprocessor', preprocessor),
                         ('xgb_clf', XGBClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0, max_depth=9))
])

xgb_cv = cross_val_score(xgb_pipeline, X_train, y_train, scoring='f1', cv=3)

xgb_model = xgb_pipeline.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
f1 = f1_score(y_test, y_pred)

print(f'Train Score = {np.mean(xgb_cv)}')
print(f'Test Score = {f1}')

plot_confusion_matrix(xgb_model, X_test, y_test)


print(classification_report(y_test, y_pred))

predictions = xgb_model.predict_proba(X_test)

thresholds = np.linspace(0, 1, 101)
precision_scores = []
recall_scores = []
for threshold in thresholds:
    adjusted_predictions = [1 if p > threshold else 0 for p in predictions[:,1]]
    precision_scores.append(precision_score(y_test, adjusted_predictions))
    recall_scores.append(recall_score(y_test, adjusted_predictions))
plt.plot(thresholds[:-1], precision_scores[:-1], label="precision")
plt.plot(thresholds[:-1], recall_scores[:-1], label="recall")
plt.xlabel('Threshold')
plt.legend()
plt.grid()
plt.show()

df['loan_status'].value_counts(normalize=True)

df_dict = {
    'precision':precision_scores,
    'recall':recall_scores,
    'threshold':thresholds
}

df_pr = pd.DataFrame(df_dict)
df_pr.head(11)



























