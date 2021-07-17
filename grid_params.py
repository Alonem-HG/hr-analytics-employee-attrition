import numpy as np
# Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
# best parameters
from sklearn.model_selection import GridSearchCV
#train
from sklearn.model_selection import train_test_split
#classifatiers
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomTreesEmbedding

def logreg_best_params(X_train, y_train, X_test, y_test):

    lg = LogisticRegression()
    
    # Hiperparámetros 
    #complex
    param_complex_grid = {'penalty': ['l1', 'l2','elasticnet'],#'elasticnet','none'
                  'C': np.logspace(-4, 4, 50), #np.logspace(0, 4, 10),[0.01, 0.1, 1, 2, 10, 100]
                  'class_weight': ['none', 'balanced'],
                  'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'max_iter':[1000]} #1000,5000, 

    #moderate
    param_moderate_grid = {'C':np.logspace(-4, 4, 50), 
                      'penalty': ['l1', 'l2'],
                      'max_iter':[5000,10000]} 

    #Basic
    param_basic_grid={"C": np.logspace(-5, 20, 30), 
                      "penalty":['none', 'l1', 'l2', 'elasticnet'],
                      "max_iter":[30000]}

    lg_cv = GridSearchCV(lg, 
                          param_basic_grid,
                          #verbose=3,
                          scoring = 'accuracy',
                          n_jobs=-1,# usar todo el procesador  
                          cv = 10)


    lg_cv.fit(X_train, y_train)
    print("Accuracy without scaling: ", lg_cv.score(X_test, y_test))

    # StandardScaler 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lg_cv.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    print("Accuracy with scaling StandardScaler ", lg_cv.score(X_test_scaled, y_test))

    # MinMaxScaler
    mms = MinMaxScaler(feature_range = (0,1))
    X_train_mms = mms.fit_transform(X_train)
    lg_cv.fit(X_train_mms, y_train)
    X_test_mms = mms.transform(X_test)
    print("Accuracy with scaling MinMaxScaler ", lg_cv.score(X_test_mms, y_test))

    # Normalize 
    norm= Normalizer()
    normalized_X = norm.fit_transform(X_train)
    lg_cv.fit(normalized_X, y_train)
    X_test_norm = norm.transform(X_test)
    print("Accuracy with scaling Normalizer ", lg_cv.score(X_test_norm, y_test))

    #PowerTransformer
    ss = PowerTransformer()
    X_pt = ss.fit_transform(X_train)
    lg_cv.fit(X_pt, y_train)
    X_test_pt = ss.transform(X_test)
    print("Accuracy with scaling PowerTransformer ", lg_cv.score(X_test_pt, y_test))

    
    besT = print("Best params LogisticRegression",lg_cv.best_params_)
    return besT

def xgbc_best_params(X_train, y_train, X_test, y_test):
    
    xgbc = XGBClassifier()
    
    # Hiperparámetros
    #brute force scan for all parameters, here are the tricks
    #usually max_depth is 6,7,8
    #learning rate is around 0.05, but small changes may make big diff
    #tuning min_child_weight subsample colsample_bytree can have 
    #much fun of fighting against overfit 
    #n_estimators is how many round of boosting
    #finally, ensemble xgboost with multiple seeds may reduce variance
    parameters = {'nthread':[4], #[4]when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [0.01, 0.02, 0.03,0.04,0.05], #so called `eta` value
                  'max_depth': [6],
                  'min_child_weight': [1, 5, 11],
                  'gamma': [0.5, 1, 1.5, 2, 5],
                  'silent': [1],
                  'subsample': [0.6, 0.8, 1.0],
                  'colsample_bytree': [0.4,0.6, 0.7,0.8, 1.0],
                  'n_estimators': [600, 800, 1000]} #number of trees, change it to 1000 for better results
                  #'missing':[-999]}

    xgbc_cv = GridSearchCV(xgbc, 
                          parameters,
                          #verbose=3,
                          scoring = 'accuracy',
                          n_jobs=-1,# usar todo el procesador  
                          cv = 10)
        
    xgbc_cv.fit(X_train, y_train)
    print("accuracy without scaling: ", xgbc_cv.score(X_test, y_test))

    # StandardScaler 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    xgbc_cv.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    print("accuracy with scaling StandardScaler ", xgbc_cv.score(X_test_scaled, y_test))

    # MinMaxScaler
    mms = MinMaxScaler()
    X_train_mms = mms.fit_transform(X_train)
    xgbc_cv.fit(X_train_mms, y_train)
    X_test_mms = mms.transform(X_test)
    print("accuracy with scaling MinMaxScaler ", xgbc_cv.score(X_test_mms, y_test))

    # Normalize 
    norm= Normalizer()
    normalized_X = norm.fit_transform(X_train)
    xgbc_cv.fit(normalized_X, y_train)
    X_test_norm = norm.transform(X_test)
    print("accuracy with scaling Normalizer ", xgbc_cv.score(X_test_norm, y_test))

    #PowerTransformer
    ss = PowerTransformer()
    X_pt = ss.fit_transform(X_train)
    xgbc_cv.fit(X_pt, y_train)
    X_test_pt = ss.transform(X_test)
    print("Accuracy with scaling PowerTransformer ", xgbc_cv.score(X_test_pt, y_test))
    
    besT = xgbc_cv.best_params_
    return besT

def knn_best_params(X_train, y_train, X_test, y_test):
    
    KNN = KNeighborsClassifier()
    
    n_range = np.arange(2,30)
    n_neighbors= n_range.tolist()
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    
    param_grid = {"n_neighbors": n_neighbors, 
                  "weights": weights,
                  "metric": metric,
                  "algorithm":algorithm}
    
    KNN_cv = GridSearchCV(KNN,
                          param_grid,
                         # verbose=3,#details
                          scoring = 'accuracy',  
                          cv = 10)
    
    KNN_cv.fit(X_train, y_train)
    print("accuracy without scaling: ", KNN_cv.score(X_test, y_test))

    # StandardScaler 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    KNN_cv.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    print("Accuracy with scaling StandardScaler ", KNN_cv.score(X_test_scaled, y_test))

    # MinMaxScaler
    mms = MinMaxScaler()
    X_train_mms = mms.fit_transform(X_train)
    KNN_cv.fit(X_train_mms, y_train)
    X_test_mms = mms.transform(X_test)
    print("Accuracy with scaling MinMaxScaler ", KNN_cv.score(X_test_mms, y_test))

    # Normalize 
    norm= Normalizer()
    normalized_X = norm.fit_transform(X_train)
    KNN_cv.fit(normalized_X, y_train)
    X_test_norm = norm.transform(X_test)
    print("Accuracy with scaling Normalizer ", KNN_cv.score(X_test_norm, y_test))

    #PowerTransformer
    ss = PowerTransformer()
    X_pt = ss.fit_transform(X_train)
    KNN_cv.fit(X_pt, y_train)
    X_test_pt = ss.transform(X_test)
    print("Accuracy with scaling PowerTransformer ", KNN_cv.score(X_test_pt, y_test))
    
    besT = KNN_cv.best_params_
    return besT


def svc_best_params(X_train, y_train, X_test, y_test):
    svc = SVC()

    # Hiperparámetros
    #Complex
    param_complex_grid = {'C': [0.1, 1, 10, 100],  
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']}  
    #Basic
    param_basic_grid = {'C': np.logspace(-5, 10, 30), 'gamma':[0.01, 0.1, 1, 10]}

    svc_cv = GridSearchCV(svc, 
                          param_complex_grid,
                          #verbose=3,
                          scoring = 'accuracy',
                          n_jobs=-1,# usar todo el procesador  
                          cv = 10)

    svc_cv.fit(X_train, y_train)
    print("accuracy without scaling:  {}".format(svc_cv.score(X_test, y_test).round(2)))

    # StandardScaler 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    svc_cv.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)
    print("accuracy with scaling StandardScaler ", svc_cv.score(X_test_scaled, y_test))
    besT = svc_cv.best_params_
    return besT