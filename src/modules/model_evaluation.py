import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, PrecisionRecallDisplay
from .utils import set_text_params

def select_X_set(name, Xs):
    '''
    Selects the appropriate X data set based on data set used to fit the model for use in other functions.
    
    args:
        name: str - name of the data set used to fit the model
        Xs: list - data sets to search through, assumes that the 0th entry is the full dataset,
            the 1st entry is a dictionary of feature selected datasets,
            and the 2nd entry is the pca transformed dataset
            
    returns: X - the appropriate data set based on the required criteria
    '''
    
    if 'top' in name:
        X = Xs[1][int(re.search(r'top(\d{1,2})', name)[1])]
    elif 'pca' in name:
        X = Xs[2]
    else:
        X = Xs[0]
        
    return X


def calculate_metrics(model, X, y, proba = 0):
    '''
    Calculates classification metrics using a fitted model and X, y data sets.
    
    args:
        model: fitted sklearn Model - used to create predictions
        X: pandas DataFrame - independent data used to create predictions
        y: pandas Series - dependent data used as ground-truth values
        proba: float - a value between 0 and 1, representing the probability threshold used
                to determine if a prediction belongs to the positive class. If 0, calls model.predict(), otherwise
                calls model.predict_proba() then separately reduces. Default 0.
    
    returns:
        y_pred: array - predicted labels for each data point
        accuracy: float - representing the accuracy of the model using predicted and ground truth labels
        precision: float - representing the precision of the model using predicted and ground truth labels
        recall: float - representing the recall of the model using predicted and ground truth labels
        f1: float - representing the f1-score of the model using predicted and ground truth labels
    '''
    
    # If a probability threshold is given, predict probabilities and reduce to predicted labels
    # Else, predict labels using model.predict()
    if proba != 0:
        probs = model.predict_proba(X)
        y_pred = [1 if k[1] >= proba else 0 for k in probs]
    else:
        y_pred = model.predict(X)
        
    # Calculate classification metrics using ground-truth labels and predicted values
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    return y_pred, accuracy, precision, recall, f1


def metrics_across_models(model_name, models, X_tests, y_test, same_name = True):
    '''
    Calculates classification metrics across all input models.
    Creates a stylized Pandas DataFrame to display all of the calculated data.
    
    args:
        model_name: str - Name to use for labeling the returned data frame
        models: list of tuples of form (str, sklearn Model) - fitted models to use for calculating metrics
        X_tests: list - list of test data to pass to select_X_set() for appropriate calculation of metrics
        y_test: pandas Series - data to use for ground-truth labels
        same_name: bool - whether each model has the same base name for formatting display names of each model,
                default True
                
    returns: metric_frame, a Pandas Styler object containing formatted classification metric data
    '''
    
    # Create a dictionary to hold model metrics
    model_metrics = {}
    
    for name, model in models:
        # Select the appropriate X dataset to use for calculating metrics
        X = select_X_set(name, X_tests)
        
        # Calculate metrics, leaving out y_pred
        metrics = calculate_metrics(model, X, y_test)[1:]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Create pandas column name
        if same_name:
            key = '_'.join(name.split('_')[1:])
        else:
            key = name
        
        # Add model metrics to model metrics dictionary
        model_metrics[key] = {metric_names[i]: np.round(metrics[i], 4) for i in range(len(metric_names))}
    
    # Wrap in a data frame and rename axis
    metric_frame = pd.DataFrame(model_metrics)
    metric_frame = metric_frame.rename_axis(model_name, axis = 1)
    
    # Apply styling
    # Set precision to 3 decimal places
    # Make the columns and index have a black background and cell borders
    # Highlight the cells with maximum accuracy, precision, recall, and f1-score in the dataset
    metric_frame = metric_frame.style.format(precision = 3)\
        .set_properties(**{'width': '12em', 'font-size': '10.5pt'})\
        .set_table_styles([
        {'selector': 'th.index_name',
         'props': 'text-align: center; background-color: black; font-size: 12pt;'
         + 'border-right: 1px solid white; border-bottom: 1px solid white;'},
        {'selector': '.row_heading',
         'props': 'text-align: center; background-color: black; border-right: 1px solid white; font-size: 12pt;'},
        {'selector': 'th.col_heading',
         'props': 'text-align: center; background-color: black; border-bottom: 1px solid white; font-size: 11pt;'},
        {'selector': 'td', 'props': 'text-align: center;'}
    ]).highlight_max(axis = 1, props = 'color: white; background-color: #444444; border: 1px solid white')
    
    return metric_frame


def plot_roc_curves(model_name, models, X_tests, y_test, reduced = False, same_name = True):
    '''
    Plots the Receiver Operator Characteristic (ROC) Curves for each model on the same plot.
    ROC curves are used to visualize the trade-off between recall and false positive rate
        for a model at different decision thresholds for classification.
    
    args:
        model_name: str - Name to use for the title of the returned plot
        models: list of tuples of form (str, sklearn Model) - fitted models to use for calculating metrics
        X_tests: list - list of test data to pass to select_X_set() for appropriate calculation of metrics
        y_test: pandas Series - data to use for ground-truth labels
        reduced: bool - whether to reduce the models chosen for clarity, current formatting selects only
            the base, grid, and pca selected models, along with their SMOTE resampled versions,
            default False
        same_name: bool - whether each model has the same base name for formatting display names of each model,
                default True
                
    returns: ax - Matplotlib Axes object for further display or modification
    '''
    
    # Define figure
    fig, ax = plt.subplots(figsize = (16, 8))
    
    # Reduce the models used, if applicable
    if reduced:
        models = [x for x in models if any([substr in x[0] for substr in ['base', 'grid', 'pca']])]
    
    # Define the colors for the plots
    colors = sns.color_palette('ch:s=0.5, rot=1.5', len(models))
    i = 0
    
    for name, model, in models:
        # Select the appropriate X dataset to use for calculating metrics
        X = select_X_set(name, X_tests)
    
        # Create probability predictions and use sklearn roc_curve and roc_auc_score to acquire data for plotting
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, thresh = roc_curve(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)
        
        # Create label for artist
        if same_name:
            label = f'{"_".join(name.split("_")[1:])} AUC: {roc_auc:.3f}'
        else:
            label = f'{name} AUC: {roc_auc:.3f}'
            
        # Draw ROC curve on axis, using the appropriate color
        sns.lineplot(x = fpr, y = tpr, color = colors[i], label = label, ax = ax);
        i += 1
    
    # Create baseline ROC line to show improvement of models over a random guesser
    ax.axline((0, 0), (1, 1), color = 'k', linestyle = '--', label = 'Random-Weighted Baseline AUC 0.5');
    
    # Set title and axis labels
    set_text_params(ax, title = f'{model_name} ROC Curves',
                    xlabel = 'False Positive Rate (Type 1 Error Rate)',
                    ylabel = 'True Positive Rate (Recall)')    
    
    # Format legend and axis limits
    ax.legend(fontsize = 13, ncols = 2, loc = 4);
    ax.set_xlim([0, 1]);
    ax.set_ylim([0, 1]);
    
    return ax


def plot_prc(model_name, models, X_tests, y_test, reduced = False, same_name = True):
    '''
    Plots the Precision-Recall Curve (PRC) for each model on the same plot.
    Precision-Recall Curves are used to visualize the trade-off between precision and recall
        for a model at different decision thresholds for classification. PRC can often be more useful
        and interpretable for datasets with large class imbalances:
        https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800/
    
    args:
        model_name: str - Name to use for the title of the returned plot
        models: list of tuples of form (str, sklearn Model) - fitted models to use for calculating metrics
        X_tests: list - list of test data to pass to select_X_set() for appropriate calculation of metrics
        y_test: pandas Series - data to use for ground-truth labels
        reduced: bool - whether to reduce the models chosen for clarity, current formatting selects only
            the base, grid, and pca selected models, along with their SMOTE resampled versions,
            default False
        same_name: bool - whether each model has the same base name for formatting display names of each model,
                default True
                
    returns: ax - Matplotlib Axes object for further display or modification
    '''
    
    # Define figure
    fig, ax = plt.subplots(figsize = (16, 8))
    
    # Reduce the models used, if applicable
    if reduced:
        models = [x for x in models if any([substr in x[0] for substr in ['base', 'grid', 'pca']])]
    
    # Define the colors for the plots
    colors = sns.color_palette('ch:s=0.5, rot=1.5', len(models))
    i = 0
    
    for name, model, in models:
        # Select the appropriate X dataset to use for calculating metrics
        X = select_X_set(name, X_tests)
        
        # Create label for artist
        if same_name:
            label = f'{"_".join(name.split("_")[1:])}'
        else:
            label = f'{name}'
        
        # Draw PRC on axis, using the appropriate color
        PrecisionRecallDisplay.from_estimator(model, X, y_test, name = label,
                                              drawstyle = 'default', ax = ax, color = colors[i])
        i += 1
        
    # Calculate the class 1 ratio presence to plot unskilled baseline to show model strength
    # Unskilled baseline for PRC is always predicting the positive class
    # which always results in a precision of class 1 ratio
    class1_ratio = y_test.value_counts(normalize = True).loc[1]
    ax.axhline(y = class1_ratio, linestyle = '--',
               color = 'k', label = f'Unskilled Baseline (AP = {class1_ratio:.02f})');

    # Set title and axis labels
    set_text_params(ax, title = f'Precision-Recall Curves for {model_name}',
                    xlabel = 'Recall', ylabel = 'Precision')
    
    # Format legend and axis limits
    # Reduced the y limits to the floor 0.1 level of the class1_ratio
    ax.legend(fontsize = 11, ncols = 2, loc = 1)
    ax.set_xlim([0, 1]);
    ax.set_ylim([np.trunc(class1_ratio * 10) / 10, 1]);

    return ax



def heatmap_confusion_matrix(model, model_name, X, y, ax = None):
    '''
    Plots a heatmap of the confusion matrix for a model, along with calculated classification metrics.
    
    args:
        model: sklearn Model - fitted model to use for predictions
        model_name: str - name of the model to use for setting the title of the heatmap
        X: pandas DataFrame - data to use for model.predict()
        y: pandas Series - data to use as ground-truth labels for calculating classification metrics
        ax: Matplotlib Axes or None - axis to use for drawing the heatmap, if None is specified, an
            axis is created. Default, None
    
    returns: ax - Matplotlib Axes object for further display or modification
    '''
    
    # Create figure, if necessary
    if ax is None:
        fig, ax = plt.subplots(figsize = (8, 8))
    
    # Calculate classification metrics and confusion matrix
    # Create labels to show confusion matrix counts and percentages
    y_pred, accuracy, precision, recall, f1 = calculate_metrics(model, X, y)
    conf = confusion_matrix(y, y_pred)
    conf_labels = np.asarray([f'{i:0d}\n\n{(i / np.sum(conf)):.2%}' for i in conf.flatten()]).reshape(2, 2)
    
    # Plot heatmap
    sns.heatmap(conf, annot = conf_labels, fmt = '', annot_kws = {'fontsize': 14},
                cbar = False, ax = ax, cmap = 'Blues');
    
    # Set title and axis labels
    set_text_params(ax, title = f'{model_name} Confusion Matrix',
                    xlabel = 'Predicted', ylabel = 'Observed')
    
    # Format legend and axis limits
    ax.set_xticklabels(['Not Churn', 'Churn']);
    ax.set_yticklabels(['Not Churn', 'Churn'], rotation = 'horizontal');
    
    # Create a text box outside of heatmap with classification metric details
    classif_labels = '\n\n'.join(['\n'.join(x) for x in zip(['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                                            [f'{accuracy:.2f}', f'{precision:.2f}',
                                                             f'{recall:.2f}', f'{f1:.2f}'])])
    props = dict(boxstyle = 'round', facecolor = 'grey', alpha = 0.01);
    ax.text(1.1, 0.5, classif_labels, fontsize = 14, transform = ax.transAxes,
            bbox = props, ha = 'center', va = 'center');
    
    return ax



def train_test_performance(model_tup, X_trains, X_tests, y_train, y_test):
    '''
    Shows confusion matrix and classification metrics of a model on training and testing sets.
    Used to investigate possible over- or under-fitting of models to the data.
    
    args:
        model_tup: (str, sklearn Model) - model name and fitted model to use for plotting and calculations
        X_trains: list - list of training data to pass to select_X_set() for appropriate calculation of metrics
        X_tests: list - list of test data to pass to select_X_set() for appropriate calculation of metrics
        y_train: pandas Series - data to use for training ground-truth labels
        y_test: pandas Series - data to use for testing ground-truth labels
                
    returns: ax - Matplotlib Axes object for further display or modification
    '''
    
    # Create figure
    fig, ax = plt.subplots(1, 2, figsize = (16, 8))
    
    # Extract name and model
    name, model = model_tup
    
    # For training and testing, create a heatmap confusion matrix with classification metrics
    for s in ['train', 'test']:
        if s == 'train':
            X = select_X_set(name, X_trains)
            heatmap_confusion_matrix(model, name + ' Train', X, y_train, ax = ax[0])
        else:
            X = select_X_set(name, X_tests)
            heatmap_confusion_matrix(model, name + ' Test', X, y_test, ax = ax[1])
    
    plt.tight_layout()
    
    return ax