# Still a bit disorganized
import sys
from pymatgen.io.cif import CifParser, CifFile
from xgboost import XGBRegressor, plot_importance
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen import Lattice, Structure, Molecule
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.ext.matproj import MPRestError
from pymatgen.ext.matproj import MPRester
from sys import stdout
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn import linear_model, decomposition, datasets
from skrvm import RVR
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
from keras.layers import Dense
from keras.models import load_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.callbacks import Callback
from keras.preprocessing import text, sequence
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
import keras
from nltk import pos_tag, word_tokenize
from nltk import WordNetLemmatizer
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from scipy import stats
from sklearn.preprocessing import binarize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn import linear_model
from sklearn import model_selection, preprocessing
import seaborn as sns
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

# import xgboost as xgb
color = sns.color_palette()

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)

warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '4'

############################### Machine Learning ###############################

system = ['entropy', 'C_v', 'eps_total_effective']

sys = system[1]

# Load training/test data

reports_df = pd.DataFrame(
    columns=['Name', 'MARE', 'MSE', 'R2', 'PearsonR', 'SpearmanR'])
reports_df_unlog = pd.DataFrame(
    columns=['Name', 'MSE', 'R2', 'PearsonR', 'SpearmanR'])
X_scalar_id_df = pd.read_csv('X_scalar_id.csv', header=0, low_memory=False)
X_df = pd.read_csv(sys+'X.csv', header=None)
X_data_df = X_df.iloc[:, 0:X_df.shape[1]-1]

scaler = StandardScaler().fit(X_df)

# Scale the train set
X_scaled_df = scaler.transform(X_df)

y = pd.read_csv(sys+"y_"+sys+".csv", header=None)
numRows = y.shape[0]

top_list = []
top_dict = {}
top_intersection_df = pd.DataFrame(
    columns=['id', 'Original', 'Predicted', 'MARE'])


bottom_list = []
bottom_dict = {}
bottom_intersection_df = pd.DataFrame(
    columns=['id', 'Original', 'Predicted', 'MARE'])

y_unlog = y/96.521

y = np.log(y)

for regr_choice in range(6):

    regr_names = ['RF', 'SVM', 'RVM', 'Huber', 'XGBOOST', 'NN', 'RANSAC']

    regr_name = regr_names[regr_choice]

    if 'NN' in regr_name:
        regr = load_model(
            sys+"/SavedModels/random_clustered_dataset_model_random_test_set_"+str(count)+".h5")
    else:
        regr = joblib.load(sys+'/SavedModels/'+regr_name+'_'+str(count)+'.pkl')

    # Now get the dataset from the original files
    if 'XGB' in regr_name:
        X_scaled_df_XGB = pd.DataFrame(X_scaled_df)
        X_scaled_df_XGB = X_scaled_df_XGB.replace(0, 0.00001)
        y_predicted = regr.predict(X_scaled_df_XGB)
    else:
        y_predicted = regr.predict(X_scaled_df)

    y_predicted = np.reshape(y_predicted, (numRows,))

    if sys != 'eps_total_effective':
        y_predicted_unlog = np.exp(y_predicted)/96.521
    else:
        y_predicted_unlog = np.exp(y_predicted)

    MARE_file = open(sys+'FullSetPrediction_MARE_'+regr_name+'.csv', 'w')
    MARE = 0
    for i in range(len(y_predicted)):
        MARE_file.write(str(X_scalar_id_df.id[i])+','+str(y.iloc[i][0])+','+str(
            y_predicted[i])+','+str(abs((y_predicted[i]-y.iloc[i][0])/y.iloc[i][0]*100))+'\n')
        MARE += abs((y_predicted[i]-y.iloc[i][0])/y.iloc[i][0]*100)
    MARE_file.close()

    MARE_file = open(sys+'Unlog_FullSetPrediction_MARE_'+regr_name+'.csv', 'w')
    MARE_unlog = 0
    for i in range(len(y_predicted)):
            MARE_file.write(str(X_scalar_id_df.id[i])+','+str(y_unlog.iloc[i][0])+','+str(
                y_predicted_unlog[i])+','+str(abs((y_predicted_unlog[i]-y_unlog.iloc[i][0])/y_unlog.iloc[i][0]*100))+'\n')
            MARE_unlog += abs((y_predicted_unlog[i] -
                               y_unlog.iloc[i])/y_unlog.iloc[i][0]*100)
    MARE_file.close()

    errors_file = open(sys+'FullSetPrediction_'+regr_name+'_Analysis.txt', 'w')
    errors_file.write('FullSetPrediction_MARE\t' +
                      str(MARE/len(y_predicted))+'\n')
    errors_file.write('FullSetPrediction_MSE\t' +
                      str(np.sqrt(mean_squared_error(y, y_predicted)))+'\n')
    errors_file.write('FullSetPrediction_r2\t' +
                      str(r2_score(y, y_predicted))+'\n')
    errors_file.close()

    errors_file = open(sys+'Unlog_FullSetPrediction_' +
                       regr_name+'_Analysis.txt', 'w')
    errors_file.write('FullSetPrediction_MARE\t' +
                      str(MARE_unlog/len(y_predicted_unlog))+'\n')
    errors_file.write('FullSetPrediction_MSE\t' +
                      str(np.sqrt(mean_squared_error(y_unlog, y_predicted_unlog)))+'\n')
    errors_file.write('FullSetPrediction_r2\t' +
                      str(r2_score(y_unlog, y_predicted_unlog))+'\n')
    errors_file.close()

    reports_df_row = pd.DataFrame(
        columns=['Name', 'MARE', 'MSE', 'R2'])
    reports_df_row.set_value(0, 'Name', regr_name)
    reports_df_row.set_value(0, 'MARE', MARE/len(y_predicted))
    reports_df_row.set_value(0, 'MSE', np.sqrt(
        mean_squared_error(y, y_predicted)))
    reports_df_row.set_value(0, 'R2', r2_score(y, y_predicted))
    reports_df = reports_df.append(reports_df_row)

    reports_df_row = pd.DataFrame(
        columns=['Name', 'MARE', 'MSE', 'R2'])
    reports_df_row.set_value(0, 'Name', regr_name)
    reports_df_row.set_value(0, 'MARE', MARE/len(y_predicted))
    reports_df_row.set_value(0, 'MSE', np.sqrt(
        mean_squared_error(y_unlog, y_predicted_unlog)))
    reports_df_row.set_value(0, 'R2', r2_score(y_unlog, y_predicted_unlog))
    reports_df_unlog = reports_df_unlog.append(reports_df_row)

    # Find best and worst

    analysis_MARE_df = pd.read_csv(
        sys+'Unlog_FullSetPrediction_MARE_'+regr_name+'.csv', header=None)
    analysis_MARE_df.columns = ['id', 'Original', 'Predicted', 'MARE']
    analysis_MARE_df = analysis_MARE_df.replace('[\[\]]', '', regex=True)

    MARE = analysis_MARE_df.sort_values(by='MARE')
    top = MARE.loc[analysis_MARE_df['MARE'] < 0.1]
    # top=MARE.head(30)
    top.to_csv(sys+'Unlog_FullSetPrediction_TopAccuracy_' +
               regr_name+'.csv', header=True, index=False)
    # top.set_index('id')
    top_dict[regr_name] = top
    top_list += [top]

    bottom = MARE.tail(50)
    bottom = MARE.loc[analysis_MARE_df['MARE'] > 10]
    # top=MARE.head(30)
    bottom.to_csv(sys+'Unlog_FullSetPrediction_BottomAccuracy_' +
                  regr_name+'.csv', header=True, index=False)
    # top.set_index('id')
    bottom_dict[regr_name] = bottom
    bottom_list += [bottom]

    if regr_choice == 0:
        top_intersection_df = top
    else:
        top_intersection_df = top_intersection_df.merge(top, on='id')

    if regr_choice == 0:
        bottom_intersection_df = bottom
    else:
        bottom_intersection_df = bottom_intersection_df.merge(top, on='id')

reports_df.to_csv(sys+'/FullSetPrediction_Report_' +
                  sys+'.csv', header=True, index=False)

reports_df_unlog.to_csv(
    sys+'/Unlog_FullSetPrediction_Report_'+sys+'.csv', header=True, index=False)
