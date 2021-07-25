import numpy as np
# scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
# best parameters
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
#train
from sklearn.model_selection import train_test_split
#classifatiers
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomTreesEmbedding
#suppress warning
import warnings
warnings.filterwarnings('ignore')

def lr_enhance(X_train, y_train, X_test, y_test):
    std_slc = StandardScaler()
    pca = PCA()
    logistic_Reg = LogisticRegression()

    pipe = Pipeline([('std_slc', std_slc),
                            ('pca', pca),
                            ('logistic_Reg', logistic_Reg)])

    #n_components = list(range(1,X.shape[1]+1,1))

    parameters = {'std_slc':[StandardScaler(), MinMaxScaler(),Normalizer(), MaxAbsScaler()],
                'pca__n_components':list(range(1,X_train.shape[1]+1,1)),
                'logistic_Reg__C': np.logspace(-4, 4, 20),
                'logistic_Reg__penalty': ['l1', 'l2'],
                'logistic_Reg__max_iter':[30000]} 

    clf_GS = GridSearchCV(pipe, parameters,
                        scoring = 'accuracy',
                        n_jobs=-1,# usar todo el procesador  
                        cv = 5)
    clf_GS.fit(X_train, y_train)

    print("Accuracy: ", clf_GS.score(X_test, y_test))
    print('Data Normalization: ',clf_GS.best_estimator_['std_slc'])
    print('')
    print(clf_GS.best_estimator_.get_params()['logistic_Reg'])
    print('')
    best= print("Best parameters:",clf_GS.best_params_)
    return best

def xgbc_enhance(X_train, y_train, X_test, y_test):
    std_slc = StandardScaler()
    xgbc = XGBClassifier()
    
    pipe = Pipeline([('std_slc', std_slc),('xgbc', xgbc)])


    # Hiperparámetros
    #brute force scan for all parameters, here are the tricks
    #usually max_depth is 6,7,8
    #learning rate is around 0.05, but small changes may make big diff
    #tuning min_child_weight subsample colsample_bytree can have 
    #much fun of fighting against overfit 
    #n_estimators is how many round of boosting
    #finally, ensemble xgboost with multiple seeds may reduce variance
    parameters = {'std_slc':[StandardScaler(), MinMaxScaler(),Normalizer(), MaxAbsScaler()],
                  'xgbc__use_label_encoder':[False],
                  'xgbc__eval_metric':['mlogloss'],
                  'xgbc__nthread':[4], #[4]when use hyperthread, xgboost may become slower
                  #'xgbc__objective':['binary:logistic'],
                  'xgbc__learning_rate': [0.01, 0.02, 0.05, 0.1, 0.25], #so called `eta` value
                  'xgbc__max_depth': [7],
                  'xgbc__min_child_weight': [1, 5, 7, 10],
                  'xgbc__gamma': [0.5, 1, 1.5, 2, 5],
                  #'silent': [1],
                  'xgbc__subsample': [0.6, 0.8, 1.0],
                  'xgbc__colsample_bytree': [0.6, 0.8, 1.0],
                  'xgbc__n_estimators': [1000]} #number of trees, change it to 1000 for better results
                  #'missing':[-999]}

    # A parameter grid for XGBoost
    params = {
            'xgbc__n_estimators' : [500],
            'xgbc__learning_rate' : [0.01, 0.02, 0.05, 0.1, 0.25],
            'xgbc__min_child_weight': [1, 5, 7, 10],
            'xgbc__gamma': [0.1, 0.5, 1, 1.5, 5],
            'xgbc__subsample': [0.6, 0.8, 1.0],
            'xgbc__colsample_bytree': [0.6, 0.8, 1.0],
            'xgbc__max_depth': [8]
            }

    xgbc_cv = GridSearchCV(pipe, params,
                          #verbose=3,
                          scoring = 'accuracy',
                          n_jobs=-1,# usar todo el procesador  
                          cv = 5)

    xgbc_cv.fit(X_train, y_train)

    print("Accuracy: ", xgbc_cv.score(X_test, y_test))
    print('Data Normalization: ',xgbc_cv.best_estimator_['std_slc'])
    print('')
    print(xgbc_cv.best_estimator_.get_params()['xgbc'])
    print('')
    best= print("Best parameters:", xgbc_cv.best_params_)
    return best

def knn_enhance(X_train, y_train, X_test, y_test):
    SS = StandardScaler()
    KNN = KNeighborsClassifier()
    
    pipe = Pipeline([('ss', SS),('knn', KNN)])

    n_range = np.arange(2,30)
    n_neighbors= n_range.tolist()
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    
    param_grid = {"ss":[StandardScaler(), MinMaxScaler(),Normalizer(), MaxAbsScaler()],
                  "knn__n_neighbors": n_neighbors, 
                  "knn__weights": weights,
                  "knn__metric": metric,
                  "knn__algorithm":algorithm}
    
    KNN_cv = GridSearchCV(pipe,param_grid,
                         # verbose=3,#details
                          scoring = 'accuracy',
                          n_jobs=-1,  
                          cv = 5)
    
    KNN_cv.fit(X_train, y_train)

    print("Accuracy: ", KNN_cv.score(X_test, y_test))
    print('Data Normalization: ',KNN_cv.best_estimator_['ss'])
    print('')
    print(KNN_cv.best_estimator_.get_params()['knn'])
    print('')
    best= KNN_cv.best_params_
    return best


def svc_enhance(X_train, y_train, X_test, y_test):
    SS = StandardScaler()
    SVC = svm.SVC()
    
    pipe = Pipeline([('ss', SS),('svc', SVC)])

    # Hiperparámetros
    #Complex
    param_complex_grid = {'ss':[StandardScaler(), MinMaxScaler(),Normalizer(), MaxAbsScaler()],
                          'svc__C': [0.1, 1, 10, 100],  
                          'svc__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']} 

    #Basic
    param_basic_grid = {'C': np.logspace(-5, 10, 30),
                        'gamma':[0.01, 0.1, 1, 10]}

    svc_cv = GridSearchCV(pipe, param_complex_grid,
                          #verbose=3,
                          scoring = 'accuracy',
                          n_jobs=-1,# usar todo el procesador  
                          cv = 5)

    svc_cv.fit(X_train, y_train)

    print("Accuracy: ", svc_cv.score(X_test, y_test))
    print('Data Normalization: ',svc_cv.best_estimator_['ss'])
    print('')
    print(svc_cv.best_estimator_.get_params()['svc'])
    print('')
    best= print("Best parameters:",svc_cv.best_params_)
    return best