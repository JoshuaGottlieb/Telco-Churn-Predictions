

def set_text_params(ax, title = '', xlabel = '', ylabel = '',
                    titlesize = 20, labelsize = 16, ticksize = 14, rotation = 0):
    '''
    Sets the title, x- and y-labels, and fontsizes for a Matplotlib Axes object.
    
    params:
        ax: Matplotlib Axes - object to modify
        title: str - to use as title for plot
        xlabel: str - to use as x-label for plot
        ylabel: str - to use as y-label for plot
        titlesize: int - to use as fontsize for title
        labelsize: int - to use as fontsize for x- and y-labels
        ticksize: int - to use as fontsize for tick labels
        rotation: int - to use for rotating x-tick labels
    
    returns:
        ax: Matplotlib Axes - modified object for further modification or display
    '''  
    
    ax.set_title(title, fontsize = titlesize);
    ax.set_xlabel(xlabel, fontsize = labelsize);
    ax.set_ylabel(ylabel, fontsize = labelsize);
    ax.tick_params(axis = 'both', labelsize = ticksize)
    if rotation != 0:
        ax.tick_params(axis = 'x', rotation = rotation)
    
    return

def map_dataframe_column(df, columns, map_dict):
    '''
    '''
    
    if type(columns) == str:
        df[columns] = df[columns].map(map_dict)
    elif type(columns) == list:
        for c in columns:
            df[c] = df[c].map(map_dict)
    else:
        print("Columns must be a str or list of str.")
        
    return
    