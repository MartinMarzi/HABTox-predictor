#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
import pandas as pd
import numpy as np
import os
import pickle
from dateutil.parser import parse
import datetime
from dateutil.parser import parse
import math
from numpy import mean

from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline as SKLpipeline
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.tree import export_text
from dtreeviz.trees import dtreeviz 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as IMBLpipeline

from sklearn.inspection import permutation_importance
import shap
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import warnings

from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"

import ipynbname
notebook_name = "HAB_modelling_6_4"

import seaborn as sns

pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 35)


# In[2]:


# Define directory path name and timestamp

# Get the current date and time as a string
timestamp = datetime.datetime.now().strftime('%d%m%Y_%H%M')

# Construct the directory path with the models folder and timestamp
dir_path = f"models/{timestamp}"
    
# Create the directory if it doesn't exist
if not os.path.exists(dir_path):
    os.makedirs(dir_path)


# In[3]:


# read df pickle
df_alg = pd.read_pickle("objects/df_alg-HAB_preprocessing_5_1")
# data = pd.read_pickle("data/preprocessed/hab_org-data-HAB_part2-preprocessing-5_2")
data = pd.read_pickle("data/preprocessed/hab_interp_data-HAB_part2-preprocessing-5_2")

data.drop(columns=["sampling station", "date"], inplace=True)
# data.set_index('date', inplace=True)


# slice by station and time
# data = data[data["sampling station"] == "Debeli_rtic"].loc["2008-01-01" : "2021-12-31"]

data.isnull().sum()


# In[4]:


# Class distribution
data["lipophylic_toxins"].value_counts(dropna=False)


# In[5]:


# move month to first place
cols = data.columns.tolist()  # Get a list of column names
cols = [cols[-2]] + cols[:-2] + [cols[-1]]  # Move the one before the last column to the first position
data = data[cols]


# In[6]:


data.drop(columns=["Chl-a","PO4-P","DIN","SECCHI"], inplace=True)
# data.drop(columns=["SECCHI"], inplace=True)
data.isnull().sum()


# # Descriptive analysis

# In[7]:


# describe
description = data.describe(include='all').round(0)

# Calculate the number of missing values for each column
missing_values = data.isna().sum()
missing_values.name = 'missing_values'

# Append the missing_values row to the description DataFrame
description_with_missing = description.append(missing_values)
description_with_missing = description_with_missing.drop(['unique', 'top', 'freq'])

description_with_missing

# # Descriptive statistics without decimals.
# # Generate summary statistics for the dataset and format all numbers as integers
# description = data.describe(include='all').apply(lambda x: x.apply(lambda y: "{:.0f}".format(y) if isinstance(y, (int, float)) else y))

# # Calculate the number of missing values for each column
# missing_values = data.isna().sum()
# missing_values.name = 'missing_values'

# # Append the missing_values row to the description DataFrame
# description_with_missing = description.append(missing_values)
# description_with_missing = description_with_missing.drop(['unique', 'top', 'freq'])

# # Output the modified DataFrame
# description_with_missing


# In[8]:


# Visualize the distribution of each variable with boxplot
num_columns = len(data.columns)
subplots_per_row = 4
num_rows = math.ceil(num_columns / subplots_per_row)

fig, axes = plt.subplots(num_rows, subplots_per_row, figsize=(20, num_rows * 4))

for i, column in enumerate(data.columns):
    row = i // subplots_per_row
    col = i % subplots_per_row
    
    if data[column].dtype == 'float64' or data[column].dtype == 'int64':
        sns.boxplot(data=data, y=column, ax=axes[row, col], palette="pastel", showmeans=True)
        axes[row, col].set_title(f'Box plot of {column}')
    else:
        # Skip categorical columns for box plots
        axes[row, col].set_visible(False)

# Hide unused subplots
for i in range(num_columns, num_rows * subplots_per_row):
    row = i // subplots_per_row
    col = i % subplots_per_row
    axes[row, col].set_visible(False)

plt.tight_layout()
plt.savefig(f"{dir_path}/feature_distribution_boxplots-{notebook_name}-{timestamp}.pdf", format="pdf", dpi=300)
plt.show()


# In[9]:


# Create a figure with a more narrow width
fig, ax = plt.subplots(figsize=(4, 5))

# Create a custom  pastel color palette
custom_pastel = sns.color_palette("bright")

# Set the color for class 1 to orange
custom_pastel[1] = sns.color_palette("bright")[2]
custom_pastel[0] = sns.color_palette("bright")[3]


# Create a count plot
sns.countplot(data=data, x="lipophylic_toxins", ax=ax, palette=custom_pastel, alpha=0.6)
ax.set_title('Distribution of lipophylic toxins test counts')

# Add numbers on top of the bars
for p in ax.patches:
    ax.text(p.get_x() + p.get_width() / 2., p.get_height() + 0.5,
            p.get_height(), ha='center', fontsize=10)

# Show the plot
plt.savefig(f"{dir_path}/lipophylic_toxins_test_counts-{notebook_name}-{timestamp}.pdf", format="pdf", dpi=300)
plt.show()


# ## Descriptive analysis by month

# In[10]:


# table of mean values for each feature by month 
import calendar

grouped_means = data.groupby('month').mean()

# Count binary values for the categorical feature grouped by month
binary_counts = data.groupby('month')['lipophylic_toxins'].value_counts().unstack()

# Calculate the ratio of positive values for each month
sum_positive = binary_counts["poz"].sum()
positive_ratios = [i for i in (binary_counts["poz"] / sum_positive)] 
positive_ratios = [round(v * 100, 1) for v in positive_ratios]

# Change month names
month_names = {i: calendar.month_name[i] for i in range(1, 13)}

# Update the index using the month_names dictionary
grouped_means.index = grouped_means.index.map(month_names)
binary_counts.index = binary_counts.index.map(month_names)

# Concatenate the grouped_means and binary_counts DataFrames
result = pd.concat([grouped_means, binary_counts], axis=1).round(2)

# Add the positive_ratios Series as a new column to the result DataFrame
result["poz %"] = positive_ratios
result.to_csv(f"{dir_path}/features_by_month_A-{notebook_name}-{timestamp}.csv", index=True)

result


# In[11]:


# table of mean values for each feature by month
import calendar

grouped_means = data.groupby('month').mean()

# Count binary values for the categorical feature grouped by month
binary_counts = data.groupby('month')['lipophylic_toxins'].value_counts().unstack()

# Calculate the ratio of positive values for each month
sum_positive = binary_counts["poz"].sum()
positive_ratios = [i for i in (binary_counts["poz"] / sum_positive)] 
positive_ratios = [round(v * 100, 1) for v in positive_ratios]

# Calculate the ratio of negative values for each month
sum_negative = binary_counts["neg"].sum()
negative_ratios = [i for i in (binary_counts["neg"] / sum_negative)]
negative_ratios = [round(v * 100, 1) for v in negative_ratios]

# Change month names
month_names = {i: calendar.month_name[i] for i in range(1, 13)}

# Update the index using the month_names dictionary
grouped_means.index = grouped_means.index.map(month_names)
binary_counts.index = binary_counts.index.map(month_names)

# Concatenate the grouped_means and binary_counts DataFrames
result = pd.concat([grouped_means, binary_counts], axis=1).round(2)

# Drop the 'neg' and 'poz' columns
result = result.drop(columns=['neg', 'poz'])

# Add the positive_ratios and negative_ratios Series as new columns to the result DataFrame
result["poz %"] = positive_ratios
result["neg %"] = negative_ratios

result.to_csv(f"{dir_path}/features_by_month_B-{notebook_name}-{timestamp}.csv", index=True)
result


# In[12]:


# Plot phytoplantkon distribution by month
# Get the phytoplankton columns
phytoplankton = data.iloc[:, 1:7].columns

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the distribution of each of the first 6 numerical variables by month on the same plot
for column in phytoplankton:
    sns.lineplot(data=data, x='month', y=column, ax=ax, label=column, palette="pastel")

# Customize the plot
ax.set_title('Distribution of phytoplankton by Month')
ax.set_ylabel('Abundance')  # Set the y-axis label to "Abundance"
ax.legend()

# Customize the x-axis ticks and labels to show only month names
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(range(1, 13), month_names)

# Save and show the plot
plt.tight_layout()
plt.savefig(f"{dir_path}/phytoplankton_abundance_distribution_by_month-{notebook_name}-{timestamp}.pdf", format="pdf", dpi=300)
plt.show()


# In[13]:


# Visualise target variable distribution by month
# Create a displot with hue
custom_palette = {"neg": sns.color_palette("bright")[2], "poz": sns.color_palette("bright")[3]}

sns.displot(data=data, x="month", hue="lipophylic_toxins", kde=True, height=6, palette=custom_palette, legend=False)

# Customize the x-axis ticks and labels
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# Customize the plot if desired
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.title('Distribution of Lipophilic Toxins test results by Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend(title="Lipophylic Toxins", loc='upper left', labels=['neg', 'poz'], bbox_to_anchor=(0.05, 0.7))


# Show the plot
plt.savefig(f"{dir_path}/target_distribution_month-{notebook_name}-{timestamp}.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()


# # Data preprocessing for modelling

# ## Removing instances with unlabeled target, label encoding

# In[14]:


# Prepare for ML in scikit-learn
# labeled and unlabeled part
data_l = data[data['lipophylic_toxins'].notnull()]
data_ul = data[data['lipophylic_toxins'].isnull()]

# Remove missing values
data_l = data_l.dropna(how="any")
print(f"class distribution:")
print(data_l["lipophylic_toxins"].value_counts(dropna=False))

X = data_l.drop("lipophylic_toxins", axis=1)
y = data_l["lipophylic_toxins"]

# sklearn lable encoding
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
print(f"class encoding: ['neg','poz'] -> {le.transform(['neg','poz'])}")


# ## Clean instances close to the decision boundary

# Clean the dataset by removing samples close to the decision boundary. Because the dataset is heavily imbalanced in favor of clas 0 (neg) we will remove instances from this class whenever finding samples which do not agree “enough” with their neighboorhood. The EditedNearestNeighbours will be used. One other option is to use Tomek links but it is more conservative and was found to perform slightly worse.

# In[15]:


from collections import Counter
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours

print(f'Original dataset shape: {Counter(y)}')
usmp = EditedNearestNeighbours()
lastMajorityCount = Counter(y)[0]
for i in range(10):
    X_res, y_res = usmp.fit_resample(X, y)
    if Counter(y_res)[0] == lastMajorityCount:
        print('Cannot remove any more samples')
        break
    else:
        print(f'Resampled dataset shape {Counter(y_res)}')
        lastMajorityCount = Counter(y_res)[0]
    X = X_res
    y = y_res


# ## Train-test split

# In[16]:


# All data
import copy

X_all = copy.copy(X)
y_all = copy.copy(y)

# train test split
X, X_eval, y, y_eval = train_test_split(X, y, shuffle=True, stratify=y, test_size=0.30) #random_state=42


# # Correlation analysis on training data

# In[17]:


# Calculate Pearson correlation between numeric features and binary target variable
pearson_correlations = X.corrwith(pd.Series(y), method='pearson')
pearson_correlations.name = 'Pearson'


# Combine the correlation values into a single dataframe
corr_df = pd.concat([pearson_correlations], axis=1)

# Create a new column for absolute Spearman correlation values
corr_df['Pearson_abs'] = corr_df['Pearson'].abs()

# Sort the dataframe by absolute Spearman correlation values
corr_df_sorted = corr_df.sort_values(by=['Pearson_abs'], ascending=False)

# Drop the absolute Spearman correlation column and return the sorted dataframe
corr_df_ranked = corr_df_sorted.drop(columns=['Pearson_abs']).round(2)

corr_df_ranked.to_csv(f"{dir_path}/Pearson_cor_coef-{notebook_name}-{timestamp}.csv", index=True)

corr_df_ranked


# # Model Training and Evaluation

# In[18]:


# Saving models

# Function to save best estimator
def save_best_estimator(grid_search_cv, classifier_name, notebook_name):
    # Get the best estimator from the GridSearchCV object
    best_estimator = grid_search_cv.best_estimator_

    # Construct the file name with the classifier name and notebook name
    pickle_file_name = f"{dir_path}/{classifier_name}-{notebook_name}-{timestamp}.pkl"
    
    # Save the best estimator to the file
    with open(pickle_file_name, 'wb') as f:
        pickle.dump(best_estimator, f)

    print(f"Best estimator saved as: {pickle_file_name}")


# In[19]:


# Function for loading saved models

def load_best_estimator(pickle_file_name):
    with open(pickle_file_name, 'rb') as f:
        best_estimator = pickle.load(f)
    return best_estimator


# ## SVM

# In[20]:


pd.set_option("display.max_rows", None)

pipeline = IMBLpipeline([
    ('smt', SMOTE()), 
    ('under', RandomUnderSampler()), 
    ('clf', SVC())
])

parameters = {
            'clf__C': [0.1, 0.5, 1, 3, 10, 100],
            # 'clf__gamma': ['scale', 'auto'],
            # 'clf__kernel': ['linear', 'rbf', 'poly'],
            'clf__class_weight': ['balanced', None],
            'smt__sampling_strategy': [ 0.2, 0.3, 0.4],
            'under__sampling_strategy': [0.5, 0.6, 0.7],
            'smt__k_neighbors': [1, 3, 5]
             }
nfolds = 5
scores = ['recall', 'precision', 'f1', 'roc_auc', 'recall_weighted']
gscv_svm = GridSearchCV(pipeline, 
                    parameters, 
                    scoring=scores,
                    cv=StratifiedKFold(n_splits=nfolds, shuffle=True),
                    return_train_score=False, 
                    verbose=1, 
                    refit="f1",
                    n_jobs=-1)
resultsGSCV = gscv_svm.fit(X, y)

# Get the classifier name from the pipeline
classifier_name = resultsGSCV.best_estimator_.named_steps['clf'].__class__.__name__
    
# save the best estimator
save_best_estimator(gscv_svm, classifier_name, notebook_name)

results = pd.DataFrame(resultsGSCV.cv_results_)
display(results.sort_values(by=[f'rank_test_f1']).transpose())


# In[21]:


# Evaluation on test data
svm_clf = gscv_svm.best_estimator_.steps[2][1]

# # Load the best estimator from the saved pickle file (replace with acctual file name)
# pickle_file_name = "models/timestamp/classifier_name-notebook_name.pkl"
# svm_clf = load_best_estimator(pickle_file_name).steps[2][1]

# Evaluation on test data
from sklearn.metrics import classification_report
y_pred = svm_clf.predict(X_eval)
SVM_classification_report = classification_report(y_eval, y_pred)

# Create classification report as dictionary
SVM_report_dict = classification_report(y_eval, y_pred, output_dict=True)

print(classification_report(y_eval, y_pred, target_names=["neg", "poz"]))


# ## Decision Tree Model (sklearn)

# In[22]:


pd.set_option("display.max_rows", None)

pipeline = IMBLpipeline([
    ('smt', SMOTE()), 
    ('under', RandomUnderSampler()), 
    ('clf', DecisionTreeClassifier())
])

parameters = {
            'clf__max_depth': [2,3,4],
            'clf__criterion': ['gini', 'entropy', 'log_loss'],
               'clf__class_weight': ['balanced', None],
               'smt__sampling_strategy': [ 0.2, 0.3, 0.4],
               'under__sampling_strategy': [0.5, 0.6, 0.7],
               'smt__k_neighbors': [1, 3, 5]
             }
nfolds = 5
scores = ['recall', 'precision', 'f1', 'roc_auc', 'recall_weighted']
gscv_dt = GridSearchCV(pipeline, 
                    parameters, 
                    scoring=scores,
                    cv=StratifiedKFold(n_splits=nfolds, shuffle=True),
                    return_train_score=False, 
                    verbose=1, 
                    refit="f1",
                    n_jobs=-1)
resultsGSCV = gscv_dt.fit(X, y)

# Get the classifier name from the pipeline
classifier_name = resultsGSCV.best_estimator_.named_steps['clf'].__class__.__name__
    
# save the best estimator
save_best_estimator(gscv_dt, classifier_name, notebook_name)

results = pd.DataFrame(resultsGSCV.cv_results_)
display(results.sort_values(by=[f'rank_test_f1']).transpose())


# In[23]:


# Evaluation on test data
dt_clf = gscv_dt.best_estimator_.steps[2][1]

# # Load the best estimator from the saved pickle file (replace with acctual file name)
# pickle_file_name = "models/12042023_1718/DecisionTreeClassifier-HAB_modelling_5_8-12042023_1718.pkl"
# dt_clf = load_best_estimator(pickle_file_name).steps[2][1]

y_pred = dt_clf.predict(X_eval)
DT_classification_report = classification_report(y_eval, y_pred)

# Create classification report as dictionary
DT_report_dict = classification_report(y_eval, y_pred, output_dict=True)

print(classification_report(y_eval, y_pred, target_names=["neg", "poz"]))


# In[24]:


viz = dtreeviz(dt_clf, X, y,
                target_name="target",
                feature_names=X.columns,
                class_names=["neg", "poz"],
             fancy=False,
               scale=1.5
              )

# Save the visualization as a PNG file
viz_file_name = f"{dir_path}/DT_visualisation.svg"
viz.save(viz_file_name)
print(f"Visualization saved as: {viz_file_name}")

# Display the visualization
viz


# ## Random Forest Model

# #### Model evaluation (Random Forest)

# In[25]:


# Random forest with grid search for parameters, testing on 5-fold CV with shuffling

pipeline = IMBLpipeline([
   ('smt', SMOTE()), 
   ('under', RandomUnderSampler()), 
    ('clf', RandomForestClassifier())
])

parameters = {
              'clf__n_estimators': [100,300,500],
              'clf__criterion': ['gini', 'entropy', 'log_loss'],
              'clf__class_weight': ['balanced', 'balanced_subsample', None],
              'smt__sampling_strategy': [ 0.2, 0.3, 0.4],
              'under__sampling_strategy': [0.5, 0.6, 0.7],
              'smt__k_neighbors': [3, 5]
             }

nfolds = 5
scores = ['recall', 'precision', 'f1', 'roc_auc']
refit_score = 'f1'
gscv_rf = GridSearchCV(pipeline, 
                    parameters, 
                    scoring=scores,
                    cv=StratifiedKFold(n_splits=nfolds, shuffle=True),
                    return_train_score=False, 
                    verbose=1, 
                    refit=refit_score,
                    n_jobs=-1)
resultsGSCV = gscv_rf.fit(X, y)

# Get the classifier name from the pipeline
classifier_name = resultsGSCV.best_estimator_.named_steps['clf'].__class__.__name__
    
# save the best estimator
save_best_estimator(gscv_rf, classifier_name, notebook_name)

results = pd.DataFrame(resultsGSCV.cv_results_)
display(results.sort_values(by=[f'rank_test_f1']).transpose())
pd.set_option("display.max_rows", None)


# In[26]:


# Evaluation on test data
rf_clf = gscv_rf.best_estimator_.steps[2][1]

# # Load the best estimator from the saved pickle file (replace with acctual file name)
# pickle_file_name = "models/timestamp/classifier_name-notebook_name.pkl"
# rf_clf = load_best_estimator(pickle_file_name).steps[2][1]

# Evaluation RF on test set
y_pred = rf_clf.predict(X_eval)
RF_classification_report = classification_report(y_eval, y_pred)

# Create classification report as dictionary
RF_report_dict = classification_report(y_eval, y_pred, output_dict=True)

print(classification_report(y_eval, y_pred, target_names=["neg", "poz"]))


# Plot the mean ROC curve of the algorithm with best performing parameter selection. We will perform CV once again and plot the ROC curve for each fold and compute and plot the mean.

# In[27]:


# Plot the ROC curve for RF

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=3, shuffle=True)
classifier = resultsGSCV.best_estimator_

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10,8))
for i, (train, test) in enumerate(cv.split(X_eval, y_eval)):
    classifier.fit(X_eval.iloc[train], y_eval[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X_eval.iloc[test],
        y_eval[test],
        name="fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Baseline (random prediction)", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

clfname = [str(step[1].__class__.__name__) for step in classifier.steps if step[0]=='clf'][0]
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f'{clfname} evaluation (ROC-AUC, {nfolds}-fold CV)',
)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{dir_path}/RF_ROC_curve-{notebook_name}-{timestamp}.pdf", format="pdf", dpi=300)
plt.show()


# Plot the mean precision-recall curve. The approach is the same as for the mean ROC curve.

# In[28]:


# Plot the mean precision-recall curve. 

from sklearn.metrics import PrecisionRecallDisplay

cv = StratifiedKFold(n_splits=3, shuffle=True)
classifier = resultsGSCV.best_estimator_

prs = []
aucs = []
mean_r = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10,8))
for i, (train, test) in enumerate(cv.split(X_eval, y_eval)):
    classifier.fit(X_eval.iloc[train], y_eval[train])
    viz = PrecisionRecallDisplay.from_estimator(
        classifier,
        X_eval.iloc[test],
        y_eval[test],
        name="fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_pr = np.interp(mean_r, viz.recall[::-1], viz.precision[::-1])
    prs.append(interp_pr)

mean_p = np.mean(prs, axis=0)
ax.plot(
    mean_r,
    mean_p,
    color="b",
    label=f"mean",
    lw=2,
    alpha=0.8,
)
ax.legend(loc="lower left")
clfname = [str(step[1].__class__.__name__) for step in classifier.steps if step[0]=='clf'][0]
ax.set(
    # xlim=[-0.05, 1.05],
    # ylim=[-0.05, 1.05],
    title=f'{clfname} evaluation (precision-recall, {nfolds}-fold CV)')
plt.tight_layout()
plt.savefig(f"{dir_path}/RF_precision-recall_curve-{notebook_name}-{timestamp}.pdf", format="pdf", dpi=300)
plt.show()


# #### Feature importance (Random Forest)

# In[29]:


# Feature importance of model (best RandomForest from gridsearch) with three methods!

fig, (ax2) = plt.subplots(1, 1, figsize=(10,9))
plt.subplots_adjust(wspace=1.1)

# Get feature importance with Permutation Based Feature Importance (randomly shuffles each feature and compute the 
# change in the model’s performance. The features which impact the performance the most are the most important one).
perm_importance = permutation_importance(rf_clf, X, y)
perm_sorted_idx = perm_importance.importances_mean.argsort()
x2 = X.columns[perm_sorted_idx]
y2 = perm_importance.importances_mean[perm_sorted_idx]
ax2.barh(x2, y2)
ax2.set_title("Permutation Importance Random Forest")


# ### SHAP

# In[30]:


# # Get feature importance with SHAP
# explainer = shap.TreeExplainer(rf)
# shap_values = explainer.shap_values(X)
# RF_shap = shap.summary_plot(shap_values, X, plot_type="bar")


# In[31]:


# SHAP summary plot
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_eval)
classid = 1

# Save shap.summary_plot()
fig = shap.summary_plot(shap_values[classid], X_eval, max_display=len(X_eval.columns), class_names=le.classes_, show=False)
plt.savefig(f"{dir_path}/shap_summary_plot-{classifier_name}-{notebook_name}-{timestamp}.pdf", format="pdf", bbox_inches='tight')


# In[32]:


# Try dependence contribution plot
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_eval)
for i in X_eval.columns:
    shap.dependence_plot(i, shap_values[1], X_eval) #interaction_index="salinity"


# ### Exaplain individual prediction with SHAP

# In[33]:


# explain positive example prediction 
correct_indices = np.where((y_eval == 1) & (rf_clf.predict(X_eval) == y_eval))[0]

instanceID = correct_indices[0]
instance = X_eval.iloc[[instanceID]]
display_instance = X_eval.iloc[[instanceID]]

prediction = rf_clf.predict(instance)[0]
prediction_probs = rf_clf.predict_proba(instance)[0]
print(f'real value: {y_eval[instanceID]}, \npredicted: {prediction}, \npredicted probs: {prediction_probs}')
max_p_id = prediction_probs.argmax()  # we will show the explanation of the bigger predicted probability
print(f'Explanation for prediction: class={max_p_id}, p={prediction_probs.max()}')

explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(instance)
shap.force_plot(explainer.expected_value[max_p_id], shap_values[max_p_id], features=display_instance)


# In[34]:


# explain negative example prediction 
correct_indices = np.where((y_eval == 0) & (rf_clf.predict(X_eval) == y_eval))[0]

instanceID = correct_indices[0]
instance = X_eval.iloc[[instanceID]]
display_instance = X_eval.iloc[[instanceID]]

prediction = rf_clf.predict(instance)[0]
prediction_probs = rf_clf.predict_proba(instance)[0]
print(f'real value: {y_eval[instanceID]}, \npredicted: {prediction}, \npredicted probs: {prediction_probs}')
max_p_id = prediction_probs.argmax()  # we will show the explanation of the bigger predicted probability
print(f'Explanation for prediction: class={max_p_id}, p={prediction_probs.max()}')

explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(instance)
shap.force_plot(explainer.expected_value[max_p_id], shap_values[max_p_id], features=display_instance)


# ### Neural Network Model

# #### Model Evaluation (MLP)

# In[35]:


# Preprocessing for NN in scikit_learn

# Model evaluation with the pipeline of SMOTE oversampling and undersampling on the training dataset only (within each cross-validation fold)!

#Remove months from X
Xnn = X.drop(X.columns[0], axis=1)

X_display = Xnn.copy()  # *used for SHAP visualization so we can show unscaled values

# scalling numeric values for NN
scaled_array = StandardScaler().fit_transform(Xnn)
Xsc = pd.DataFrame(scaled_array, columns=Xnn.columns)


# In[36]:


# pd.set_option("display.max_rows", None)


# In[37]:


# MLP with grid search for parameters, testing on 5-fold CV with shuffling

pipeline = IMBLpipeline([
    ('over', SMOTE()),
    ('under', RandomUnderSampler()),
    ('clf', MLPClassifier(solver='lbfgs', max_iter=5000))
])

parameters = {'over__k_neighbors': range(1,7),
              'over__sampling_strategy': [0.5, 0.6, 0.8], # probaj poveča ovresampling do 0.9
              'under__sampling_strategy': [0.6, 0.7, 0.8],
              'clf__hidden_layer_sizes': [(2, ), (2, 2), (3,), (3,3)],
              'clf__solver': ['lbfgs', 'sgd', 'adam']
             }
nfolds = 5
scores = ['recall', "precision", 'f1', 'roc_auc']
gscv_NN = GridSearchCV(pipeline, 
                    parameters, 
                    scoring=scores,
                    cv=StratifiedKFold(n_splits=nfolds, shuffle=True),
                    n_jobs= -1, 
                    return_train_score=False, 
                    verbose=1, 
                    refit= "f1")
resultsGSCV = gscv_NN.fit(Xsc, y)

# Get the classifier name from the pipeline
classifier_name = resultsGSCV.best_estimator_.named_steps['clf'].__class__.__name__
    
# save the best estimator
save_best_estimator(gscv_NN, classifier_name, notebook_name)

results = pd.DataFrame(resultsGSCV.cv_results_)
display(results.sort_values(by=[f'rank_test_f1']).transpose())


# In[38]:


# Evaluation on test data
nn_clf = gscv_NN.best_estimator_.steps[2][1]

# # Load the best estimator from the saved pickle file (replace with acctual file name)
# pickle_file_name = "models/timestamp/classifier_name-notebook_name.pkl"
# nn_clf = load_best_estimator(pickle_file_name).steps[2][1]

# Evaluation of NN on test set
X_eval_nn = X_eval.drop(X_eval.columns[0], axis=1)
X_eval_display = X_eval_nn.copy()

scaler = StandardScaler().fit(Xnn)
X_eval_sc = scaler.transform(X_eval_nn)
X_eval_sc = pd.DataFrame(X_eval_sc, columns=Xnn.columns)
y_pred = nn_clf.predict(X_eval_sc)
NN_classification_report = classification_report(y_eval, y_pred)

# Create classification report as dictionary
NN_report_dict = classification_report(y_eval, y_pred, output_dict=True)

print(classification_report(y_eval, y_pred, target_names=["neg", "poz"]))


# In[39]:


# Plot the ROC curve for MLP (Check code as moved form RF!). 

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=3, shuffle=True)
classifier = resultsGSCV.best_estimator_

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10,8))
for i, (train, test) in enumerate(cv.split(X_eval_sc, y_eval)):
    classifier.fit(X_eval_sc.iloc[train], y_eval[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X_eval_sc.iloc[test],
        y_eval[test],
        name="fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Baseline (random prediction)", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

clfname = [str(step[1].__class__.__name__) for step in classifier.steps if step[0]=='clf'][0]
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title=f'{clfname} evaluation (ROC-AUC, {nfolds}-fold CV)',
)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{dir_path}/MLP_ROC_curve-{notebook_name}-{timestamp}.pdf", format="pdf", dpi=300)
plt.show()


# In[40]:


# Plot the mean precision-recall curve for MLP (Check code as moved form RF!). 

from sklearn.metrics import PrecisionRecallDisplay

cv = StratifiedKFold(n_splits=3, shuffle=True)
classifier = resultsGSCV.best_estimator_

prs = []
aucs = []
mean_r = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10,8))
for i, (train, test) in enumerate(cv.split(X_eval_sc, y_eval)):
    classifier.fit(X_eval_sc.iloc[train], y_eval[train])
    viz = PrecisionRecallDisplay.from_estimator(
        classifier,
        X_eval_sc.iloc[test],
        y_eval[test],
        name="fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_pr = np.interp(mean_r, viz.recall[::-1], viz.precision[::-1])
    prs.append(interp_pr)

mean_p = np.mean(prs, axis=0)
ax.plot(
    mean_r,
    mean_p,
    color="b",
    label=f"mean",
    lw=2,
    alpha=0.8,
)
ax.legend(loc="lower left")
clfname = [str(step[1].__class__.__name__) for step in classifier.steps if step[0]=='clf'][0]
ax.set(
    # xlim=[-0.05, 1.05],
    # ylim=[-0.05, 1.05],
    title=f'{clfname} evaluation (precision-recall, {nfolds}-fold CV)')
plt.tight_layout()
plt.savefig(f"{dir_path}/MLP_precision-recall_curve-{notebook_name}-{timestamp}.pdf", format="pdf", dpi=300)
plt.show()


# #### Feature Importance (MLP)

# #### Feature importance with SHAP

# First, visualize the impact of all features on both classes in one chart. We are using KernelExplainer but simpler general Explainer should be also tested once the SHAP code fixes all bugs.
# 
# **Note: SHAP explanations change between runs because of sampling and probably other random factors!**

# In[41]:


# # explain the model's predictions using SHAP
# import shap
# import warnings
# warnings.filterwarnings("ignore")
# shap.initjs()

# explainer = shap.KernelExplainer(NN_model.predict_proba, shap.sample(X_eval_sc,20))
# shap_values = explainer.shap_values(X_eval_sc, nsamples=50)
# shap.summary_plot(shap_values, X_eval_sc, max_display=len(X.columns), class_names=le.classes_)


# Now for each class separately. We observe the impact of features on the returned model's probability for a given class.

# In[42]:


warnings.filterwarnings("ignore")
shap.initjs()

explainer = shap.KernelExplainer(nn_clf.predict_proba, shap.sample(X_eval_sc, 50))
shap_values = explainer.shap_values(X_eval_sc, nsamples=50)
classid = 1

# Save shap.summary_plot()
fig = shap.summary_plot(shap_values[classid], X_eval_display, max_display=len(X_eval_display.columns), class_names=le.classes_, show=False)
plt.savefig(f"{dir_path}/shap_summary_plot-{classifier_name}-{notebook_name}-{timestamp}.pdf", format="pdf", bbox_inches='tight')


# In[43]:


# Try dependence contribution plot
explainer = shap.KernelExplainer(nn_clf.predict_proba, shap.sample(X_eval_sc, 50))
shap_values = explainer.shap_values(X_eval_sc, nsamples=50)

for i in X_eval_display.columns:
    shap.dependence_plot(i, shap_values[1], X_eval_display,) #interaction_index="salinity"


# Example intepretation: The fact this slopes upward says the higher the soca flow, the higher the model's prediction is for poz/neg. The spread suggests that other features must interact with Soca flow. 
# In general, high Soca flow increases the chance of poz/neg. But if the sea temp is moderate or low, that trend reverses and even high soca flow does not increase preditions of poz/neg as the sea temp is too low.
# https://www.kaggle.com/code/dansbecker/advanced-uses-of-shap-values
# ---- 
# To interpret the dependence contribution plot, look for trends or patterns in the relationship between the 'salinity' feature values and their SHAP values. For example:
# 
# If the points show a clear positive trend, it means that as the 'salinity' value increases, its positive contribution to the model's prediction also increases.
# If the points show a clear negative trend, it means that as the 'salinity' value increases, its negative contribution to the model's prediction increases (i.e., higher 'salinity' values decrease the probability of class 1).
# If there is no clear trend or the points are scattered randomly, it means that there is no strong relationship between the 'salinity' feature values and their SHAP values.
# Additionally, observe the colors of the points in the plot. If there is a clear pattern in the colors, it may indicate that the interaction between 'salinity' and the interaction feature has a significant impact on the model's predictions. This can help you identify interactions between features that the model is capturing.

# Now let's explain the prediction of a single instance. We will show the explanation of the bigger predicted probability to see why the model decided as it did. But in practice we could be interested only in the explanation of the probability of the positive prediction.

# In[44]:


# explain positive example prediction 
correct_indices = np.where((y_eval == 1) & (nn_clf.predict(X_eval_sc) == y_eval))[0]

instanceID = correct_indices[0]
instance = X_eval_sc.iloc[[instanceID]]
display_instance = X_eval_display.iloc[[instanceID]]

prediction = nn_clf.predict(instance)[0]
prediction_probs = nn_clf.predict_proba(instance)[0]
print(f'real value: {y_eval[instanceID]}, \npredicted: {prediction}, \npredicted probs: {prediction_probs}')
max_p_id = prediction_probs.argmax()  # we will show the explanation of the bigger predicted probability
print(f'Explanation for prediction: class={max_p_id}, p={prediction_probs.max()}')

explainer = shap.KernelExplainer(nn_clf.predict_proba, shap.sample(Xsc, 50))
shap_values = explainer.shap_values(instance, nsamples=500)
shap.force_plot(explainer.expected_value[max_p_id], shap_values[max_p_id], features=display_instance)


# In[45]:


# explain negative example prediction 
correct_indices = np.where((y_eval == 0) & (nn_clf.predict(X_eval_sc) == y_eval))[0]

instanceID = correct_indices[0]
instance = X_eval_sc.iloc[[instanceID]]
display_instance = X_eval_display.iloc[[instanceID]]

prediction = nn_clf.predict(instance)[0]
prediction_probs = nn_clf.predict_proba(instance)[0]
print(f'real value: {y_eval[instanceID]}, \npredicted: {prediction}, \npredicted probs: {prediction_probs}')
max_p_id = prediction_probs.argmax()  # we will show the explanation of the bigger predicted probability
print(f'Explanation for prediction: class={max_p_id}, p={prediction_probs.max()}')

explainer = shap.KernelExplainer(nn_clf.predict_proba, shap.sample(Xsc, 50))
shap_values = explainer.shap_values(instance, nsamples=500)
shap.force_plot(explainer.expected_value[max_p_id], shap_values[max_p_id], features=display_instance)


# Interpretation:
# 
# Observe the base value (Expected value) and the prediction line. This gives you an idea of the overall prediction for the instance compared to the average prediction.
# 
# Look at the colored arrows and identify the most important features, which are the ones with the longest arrows. These features have the greatest impact on the prediction.
# 
# Analyze the direction and color of the arrows to understand whether each feature increases or decreases the probability of the positive class (class 1) and whether the feature values are high or low.
# 
# Consider the interactions between the features and their combined impact on the prediction.
# 
# By analyzing the force plot, you can gain insights into the contributions of each feature to the model's prediction for a specific instance, helping you understand the model's decision-making process.

# ### LIME

# In[46]:


import lime
from lime import lime_tabular

# Create a LIME explainer object
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(Xsc),
    feature_names=Xsc.columns,
    class_names=le.classes_,
    mode='classification'
)

# Exaplain an instance
instanceID = 100
lime_exp = lime_explainer.explain_instance(
    data_row=Xsc.iloc[instanceID], 
    predict_fn=nn_clf.predict_proba
)

lime_exp.show_in_notebook(show_table=True)

# Save the LIME explanation as an image file
# fig = lime_exp.as_pyplot_figure(label=1)
# plt.savefig(f"{dir_path}/lime_explanation-{classifier_name}-{notebook_name}-{timestamp}.pdf", format="pdf", bbox_inches='tight')


# ### Conclusion

# In[47]:


# Define a function to extract the metrics from the classification report dictionary
def extract_metrics(report_dict):
    metrics = {}
    for class_label in report_dict:
        if class_label in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        metrics[class_label] = {
            'precision': report_dict[class_label]['precision'],
            'recall': report_dict[class_label]['recall'],
            'f1-score': report_dict[class_label]['f1-score'],
            'support': report_dict[class_label]['support']
        }
    return metrics

# Extract the metrics for each classifier
SVM_metrics = extract_metrics(SVM_report_dict)
DT_metrics = extract_metrics(DT_report_dict)
RF_metrics = extract_metrics(RF_report_dict)
NN_metrics = extract_metrics(NN_report_dict)

# Create a dictionary to store the metrics for each classifier
classifier_metrics = {
    'Support Vector Mashines': SVM_metrics,
    'Decision Tree': DT_metrics,
    'Random Forest': RF_metrics,
    'Neural Network': NN_metrics
}

# Convert the dictionary to a pandas DataFrame
summary_df = pd.concat({k: pd.DataFrame(v).transpose() for k, v in classifier_metrics.items()}, axis=0)
summary_df.reset_index(inplace=True)
summary_df.columns = ['Classifier', 'Class', 'Precision', 'Recall', 'F1-score', 'Support']

# Map the original class labels to the new names
class_name_mapping = {
    '0': 'neg',
    '1': 'poz'
}
summary_df['Class'] = summary_df['Class'].map(class_name_mapping)

# Filter the summary DataFrame to display only the 'poz' class
poz_summary_df = summary_df.query("Class == 'poz'")

# Display the filtered summary DataFrame
poz_summary_df


# In[48]:


# Format table

poz_summary_df = poz_summary_df.round(2)
# poz_summary_df = poz_summary_df.set_index('Classifier')
poz_summary_df = poz_summary_df.drop(['Support', "Class"], axis=1)
# poz_summary_df = poz_summary_df.style.highlight_max(subset=['Precision', 'Recall', 'F1-score'], color='green')
poz_summary_df.to_csv(f"{dir_path}/results_summary_df.csv")
poz_summary_df["Timestamp"] = timestamp
poz_summary_df


# In[49]:


# Add results to results to results df
df_results = pd.read_pickle("data/results/df_results_1_0")
df_results


# In[50]:


df_results = df_results.append(poz_summary_df)
df_results.to_pickle("data/results/df_results_1_0")
df_results

