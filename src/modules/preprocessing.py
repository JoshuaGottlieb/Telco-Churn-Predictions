import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

def encode_data(X_train, X_test):
    '''
    Encodes independent variables using sklearn transformers.
    
    args:
        X_train: pandas DataFrame - training data
        X_test: pandas DataFrame - testing data
        
    returns: X_train_prep, X_test_prep - tuple of processed DataFrames
    '''
    
    # Define categorical columns and preprocessing using OneHotEncoder()
    categorical_columns = ['gender', 'senior_citizen', 'partner',
               'dependents', 'phone_service', 'multiple_lines',
               'internet_service', 'online_security', 'online_backup',
               'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
               'contract', 'paperless_billing', 'payment_method']
    categorical_preprocessor = Pipeline(
        steps = [('ohe', OneHotEncoder(drop = 'first'))]
    )
    
    # Define numerical columns and preprocessing using StandardScaler()
    numerical_columns = ['tenure', 'monthly_charges', 'total_charges']
    numerical_preprocessor = Pipeline(steps = [('ssc', StandardScaler())])

    # Use ColumnTranformer() to put both preprocessing steps into one transformer
    preprocessor = ColumnTransformer(
        transformers = [('cat', categorical_preprocessor, categorical_columns),
                        ('num', numerical_preprocessor, numerical_columns)], n_jobs = -1)
    
    # Fit to training data, transform training and testing data
    X_train_prep = pd.DataFrame(preprocessor.fit_transform(X_train),
                             columns = [x.split('__')[-1] for x in preprocessor.get_feature_names_out()])
    X_test_prep = pd.DataFrame(preprocessor.transform(X_test),
                             columns = [x.split('__')[-1] for x in preprocessor.get_feature_names_out()])
    
    return X_train_prep, X_test_prep


def select_classf_columns(X, y, sig = 0.05):
    '''
    Calculates the ANOVA F-value for each independent variable on the target.
    Filters variables with p-value greater than or equal to the significcance level and sorts the results by F-score.
    
    args:
        X: pandas DataFrame - dataframe of independent variables
        y: pandas Series - Series of dependent variable
        sig: float - significance level to filter by, default 0.05
    
    returns: results, DataFrame of the indepdent features with their F-scores and p-values
    '''
    # Create selector and fit to data
    selector = SelectKBest(f_classif, k = 'all')
    selector.fit(X, y)

    # Extract the features, their scores, and their p-values
    results = pd.DataFrame((selector.feature_names_in_,
                            selector.scores_,
                            selector.pvalues_), index = ['feat', 'scores', 'pvalues']).T
    
    # Filter and sort results
    results = results[results.pvalues < sig].sort_values('scores', ascending = False)
    
    return results


def sklearn_vif(df):
    '''
    Calculates the variance inflation factor (VIF) of each column in a dataframe.
    VIF = 1 / (1 - R^2), where R^2 is the coefficient of determination of a linear regression using
        all other features besides the target feature.
    
    args:
        df: pandas DataFrame - data to use for calculations
        
    returns: df_vif, a DataFrame with the features and their variance inflation factors
    '''
    
    vif_scores = {}
    
    for col in df.columns:
        # Separate into independent and dependent variables
        other_cols = [i for i in df.columns if i != col]
        X = df[other_cols]
        y = df[col]
        
        # Calculate R^2
        r_squared = LinearRegression().fit(X, y).score(X, y)
        
        # If R^2 = 1, use an arbitrarily large number to prevent divison by zero
        if 1 - r_squared == 0:
            vif = 9e10
        else:
            vif = 1 / (1 - r_squared)
            
        vif_scores[col] = vif
    
    # Wrap in dataframe
    df_vif = pd.DataFrame(vif_scores.items(), columns = ['feat', 'VIF'])
    
    return df_vif


def feature_select(X_train, y_train, X_test, N, sig = 0.05, vif_threshold = 10.0):
    '''
    Selects the best N features by their F-scores with the target variable.
    Removes collinear features based off of variance inflation factor (VIF).
    Collects all related columns for one-hot-encoded columns, even if some of the
        one-hot encoded columns are not in the top N features.
    
    args:
        X_train: (pandas DataFrame) - dataframe of training features
        y_train: (pandas Series) - series of target labels
        X_test: (pandas DataFrame) - dataframe of test features
        N: (int) - number of features to select
        sig: (float) - p-value to use for excluding features during ANOVA F-test, default 0.05
        vif_threshold: (float) - threshold to use for determining collinearity using the VIF
    
    returns: X_train_reduced, X_test_reduced - DataFrames with a subset of selected features
    '''
    
    # Extract ANOVA F-scores and filter by p-value
    important_cols = select_classf_columns(X_train, y_train, sig = sig)
    
    # Extract VIF scores for remaining columns and filter by threshold
    df_vif = sklearn_vif(X_train[important_cols.feat.values]).merge(important_cols, on = 'feat')
    df_vif = df_vif[df_vif.VIF < vif_threshold]
    
    # Select the unique column stems to pick up the top N features by column stem
    # Allows for proper extraction of all one-hot-encoded features
    col_stems = [x if len(x.split('_')) == 1 else '_'.join(x.split('_')[:-1]) for x in df_vif.feat.values]
    topN_stems = list(dict.fromkeys(col_stems).keys())[:N]
    
    # Filter on the features selected
    selected_features = [x for x in df_vif.feat.values if x in topN_stems
                             or '_'.join(x.split('_')[:-1]) in topN_stems]
    
    # Subset training and testing data
    X_train_reduced = X_train[selected_features]
    X_test_reduced = X_test[selected_features]
    
    return X_train_reduced, X_test_reduced


def pca_select(X_train, X_test, n_components = 3, explained_ratio = None, component_delta = 0.03):
    '''
    Transforms data using Principal Component Analysis (PCA) and returns a number of components.
    
    args:
        X_train: (pandas DataFrame) - dataframe of training features
        X_test: (pandas DataFrame) - dataframe of test features
        n_components: (int) - number of components to return, value is ignored if explaned_ratio is not None,
                                default 3
        explained_ratio: (float or None) - amount between 0 and 1 of the explained variance to be captured,
                                if not None, returns the first K components where the explained variance ratio
                                of those components is at least the explained ratio or until a component contributes
                                less than the component_delta. Default None
        component_delta: (float) - amount of explained variance ratio for each successive component,
                                function terminates if the next component would contribute less than component_delta
                                to the explained variance ratio. value is ignored if explained_ratio is None,
                                default 0.03
    
    returns:
        X_train_reduced, X_test_reduced - DataFrames with a subset of selected features
        component_exp_variances - array of explained variance ratios of returned components
    '''
    
    # Instantiate and fit a PCA on the training data
    pca = PCA()                   
    pca.fit(X_train)
    
    # If explained_ratio is None, select the first N components
    # Else, extract enough components to reach the desired explained_ratio
    # Then filter away components with too small of a contribution to explained_variance_ratio
    if explained_ratio is None:
        component_exp_variances = pca.explained_variance_ratio_[:n_components]
    else:
        sums = [np.sum(pca.explained_variance_ratio_[:i + 1])
                for i in range(len(pca.explained_variance_ratio_))]
        n = [i + 1 for i, x in enumerate(sums) if x >= explained_ratio][0]
        component_exp_variances = pca.explained_variance_ratio_[:n]
    
    for i, variance in enumerate(component_exp_variances):
        if variance < component_delta:
            component_exp_variances = component_exp_variances[:i]
            break
    
    # Select the appropriate number of components
    selected_features = [f'pc_{i + 1}' for i in range(len(component_exp_variances))]
    
    # Transform and filter train and test sets
    X_train_reduced = pd.DataFrame(pca.transform(X_train),
                                   columns = [f'pc_{i + 1}' for i in range(pca.n_components_)])[selected_features]
    X_test_reduced = pd.DataFrame(pca.transform(X_test),
                                  columns = [f'pc_{i + 1}' for i in range(pca.n_components_)])[selected_features]
    
    return X_train_reduced, X_test_reduced, component_exp_variances