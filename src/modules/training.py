import os
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from .utils import save_model

def fit_model(X_train, y_train, model_name, model,
              gridsearch = False, gridsearch_scoring = ['roc_auc'], gridsearch_params = {},
              train_scores = False, cv = 5, refit_metric = None,
              resample = False, resample_k = 5, random_state = 42,
              save = True, save_path = './models/temp_model.pickle',
              compression = None):
    '''
    Fits an sklearn model on training data with capability for grid searching and SMOTE resampling.
    
    args:
        X_train: pandas DataFrame - dataframe of training features
        y_train: pandas Series - series of target labels
        model_name: str - model name to use in imblearn Pipeline() for access via grid searching
        model: sklearn Model - model to fit
        gridsearch: bool - whether to place model into an sklearn GridSearchCV(), default False
        gridsearch_scoring: str, or list of str - scoring metrics to track while grid searching,
                    default ['roc_auc']
        gridsearch_params: dict - parameters to pass to GridSearchCV() to grid search over
        train_scores: bool - whether to capture training scores during GridSearchCV(), can greatly
                    increase model size according to sklearn documentation, default False
        cv: int - number of cross-validation folds to use while grid searching, default 5
        refit_metric: str or None - scoring metric to use for deciding on the best model found during a grid search;
                    used to refit the best found model, if None, uses the first metric in gridsearch_scoring,
                    default None
        resample: bool - whether to apply SMOTE resampling, default False
        resample_k: int - number of neighbors to pass to SMOTE resampler, default 5
        random_state: int or None - seed to pass to SMOTE resampler for reproducibility, if None,
                    SMOTE resampler creates a RandomState instance via np.random, default 42
        save: bool - whether to save the model after fitting, default True
        save_path: str - path destination for saving model
        compression: None, 'gzip', 'bz2', or 'lzma' - whether to compress the pickled model file using
            the selected compression
        
    returns: full_model - fitted sklearn Model
    '''
    
    # Define the pipeline
    pipeline = Pipeline(steps = [(model_name, model)])
    
    # Insert SMOTE resampling, if specified
    if resample:
        pipeline.steps.insert(0, ('smote', SMOTE(random_state = random_state,
                                                 k_neighbors = resample_k)))
        
    # Wrap pipeline in a gridsearch, if specified
    if gridsearch:
        # Define refit metric
        if refit_metric is None:
            if type(gridsearch_scoring) == str:
                gridsearch_scoring = [gridsearch_scoring]
            refit_metric = gridsearch_scoring[0]
            
        full_model = GridSearchCV(estimator = pipeline,
                                  param_grid = gridsearch_params,
                                  cv = cv, scoring = gridsearch_scoring,
                                  refit = refit_metric,
                                  return_train_score = train_scores,
                                  n_jobs = -1)
    else:
        full_model = pipeline
        
    # Fit model
    full_model.fit(X_train, y_train)
    
    # Save model
    if save:
        save_model(full_model, save_path, compression = compression)
    
    return full_model


def fit_all_models(X_trains, y_train, model_name, model, params, compression = None):
    '''
    Fits all variations using a specific model type.
    Currently fits models using base, feature selected, and pca transformed training data.
    Each training set is fit on a separate model with and witout SMOTE resampling.
    All models except for the base model utilize GridSearchCV() for hyperparameter tuning.
    
    args:
        X_trains: pandas DataFrame - list of dataframes of training features
        y_train: pandas Series - series of target labels
        model_name: str - name to use for grid searching and for defining save paths
        model: sklearn Model - model to use for training
        params: dict - dictionary of grid search parameters to use for grid searching
        compression: None, 'gzip', 'bz2', or 'lzma' - whether to compress the pickled model files using
            the selected compression
    '''
    
    # Define path directory
    path_dir = f'../models/{model_name}'
    
    # Create directory if it does not exist
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    
    # Fit pca and feature-selected models with and without resampling
    for state in [False, True]:
        if state:
            path_ext = '_SMOTE'
        else:
            path_ext = ''
        
        # Fit base model
        fit_model(X_trains[0], y_train, model_name, model, resample = state,
                  save_path = f'{path_dir}/{model_name}_base{path_ext}', compression = compression)
        
        # Gridsearch full model
        fit_model(X_trains[0], y_train, model_name, model, resample = state,
                  gridsearch = True, gridsearch_params = params, gridsearch_scoring = ['roc_auc', 'recall', 'f1'],
                  save_path = f'{path_dir}/{model_name}_grid{path_ext}', compression = compression)
        
        # Gridsearch pca model
        fit_model(X_trains[2], y_train, model_name, model, resample = state,
                  gridsearch = True, gridsearch_params = params, gridsearch_scoring = ['roc_auc', 'recall', 'f1'],
                  save_path = f'{path_dir}/{model_name}_pca{path_ext}', compression = compression)
        
        # Gridsearch feature selected models
        for k in list(X_trains[1].keys()):
            fit_model(X_trains[1][k], y_train, model_name, model, resample = state,
                      gridsearch = True, gridsearch_params = params,
                      gridsearch_scoring = ['roc_auc', 'recall', 'f1'],
                      save_path = f'{path_dir}/{model_name}_top{k:02d}{path_ext}',
                      compression = compression)
            
    return