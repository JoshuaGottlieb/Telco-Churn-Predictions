import matplotlib.pyplot as plt
import seaborn as sns
from .utils import set_text_params

def categorical_by_categorical(df, categorical, figsize = (8, 8), subset_cond = None,
                               legend = [], xlabel = 'Churn', ylabel = 'Count', title = '',
                               target = 'churn', ax = None):
    '''
    Plots a seaborn countplot of one categorical variable against another categorical variable.
    
    Params:
        df: pandas DataFrame, containing data to use for plotting
        categorical: str, name of the column containing the categorical variable to be used
        figsize: tuple, defines the size of the matplotlib figure in inches
        subset_cond: int, category to exclude when plotting
        legend: list of str, labels to pass to matplotlib for drawing the legend
        xlabel: str, text to use for x-axis label
        ylabel: str, text to use for y-axis label
        title: str, text to use for title
        target: str, name of the column containing the target categorical variable to be used
        ax: matplotlib axis, for drawing on a premade axis, if not passed in, a new matplotlib axis is created
        
    Returns: matplotlib axis with countplot drawn on
    '''
        
    # Create a matplotlib fig, ax subplots object pair if one is not passed in
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    # Create a subset dataframe
    data = df[[target, categorical]].copy()
    
    # If there an additional filtering condition is passed in, slice dataframe
    if subset_cond is not None:
        data = data[data[categorical] != subset_cond]
    
    # Draw seaborn countplot
    sns.countplot(data = data, x = target, hue = categorical, ax = ax);
    
    # Annotate bar values on graph
    # Reorder columns to groupby categorical column first
    values = data[[categorical, target]].value_counts(sort = False)
    index_labels = values.index.levels[0]
    for i, c in enumerate(ax.containers):
        ax.bar_label(container = c, labels = values[index_labels[i]])
    
    # Set labels, ticks, and legend
    ax.legend(labels = legend, fontsize = 14);
    set_text_params(ax, title = title, xlabel = xlabel, ylabel = ylabel)
    
    return ax

def categorical_by_numerical(df, numerical, figsize = (8, 8), height = 6,
                             quartile_label_height = 800, quartile_precision = 2,
                             quartile_xoffset = 1, quartile_yoffset = 0.05,
                             titles = [], xlabel = '', ylabel = 'Count', sharex = True,
                             colors = ['red', 'blue', 'black'], quartiles = True, target = 'churn'):
    '''
    Plots a seaborn FacetGrid of histplots of a categorical variable against a numerical variable.
    
    Params:
        df: pandas DataFrame, containing data to use for plotting
        numerical: str, name of the column containing the numerical variable to be used
        figsize: tuple, defines the size of the matplotlib figure in inches
        height: int, defines the height of the seaborn FacetGrid
        quartile_label_height: int, height to draw labels for quartiles
        quartile_precision: int, precision of quartile labels
        quartile_xoffset: int, x-offset for drawing quartile labels so as not to overlap with plotted lines
        quartile_yoffset: int, y-offset as a decimal percent for drawing consecutive quartile
                            labels so as not to have consecutive labels overlap
        titles: list of str, text to use for titles of FacetGrid
        xlabel: str, text to use for x-axis label
        ylabel: str, text to use for y-axis label
        sharex: bool, whether the FacetGrid should share the x_axis across subplots
        colors: list of str, colors to be used for plotting quartiles, passed to matplotlib
        quartiles: bool, whether to draw 1st, 2nd, and 3rd quartiles on the histplots
        target: str, name of the column containing the target categorical variable to be used
        
    Returns: seaborn FacetGrid with histplots drawn
    '''
    
    # Subset the data
    data = df[[target, numerical]].copy()
    
    # Create the facet grid
    g = sns.FacetGrid(data = data, col = target, height = height, sharex = sharex);
    ax = g.axes[0]
      
    # If quartiles should be drawn, grab the quartiles using pd.DataFrame.describe()
    # Draw using matplotlib ax.axvline and ax.text
    if quartiles:
        # Quartile values
        target_values = sorted(data[target].value_counts().index.tolist())
        quartiles = [data.loc[data[target] == x][numerical].describe().tolist()[-4:-1] for x in target_values]
        for i, outcome in enumerate(quartiles):
            for j, quartile in enumerate(outcome):
                # Create label of percentile
                label = f'{(j + 1) * 25}%'
                # Draw quartile line
                ax[i].axvline(quartile, ls = '--', label = label, c = colors[j])
                # Draw quartile label
                ax[i].text(quartile + quartile_xoffset,
                           quartile_label_height  - (quartile_yoffset * j * quartile_label_height),
                           f'{quartile:.{quartile_precision}f}', transform = ax[i].transData);

    # Draw histplots, add legend, labels, and titles
    g.map(sns.histplot, numerical);
    g.add_legend();
    g.set_xlabels(xlabel, fontsize = 14);
    g.set_ylabels('Count', fontsize = 14);
    for k, a in enumerate(ax):
        a.set_title(titles[k], fontsize = 20);
    g.set_axis_labels(labelsize = 12);

    return g