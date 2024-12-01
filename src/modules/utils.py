import gzip
import bz2
import lzma
import pickle

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

def save_model(model, path, compression = None):
    '''
    Saves a model to a destination by serializing it as a pickle file.
    
    args:
        model: sklearn Model - fitted sklearn model to save
        path: str - destination path for writing saved model
        compression: None, 'gzip', 'bz2', or 'lzma' - whether to compress the pickled model file using
            the selected compression
    '''
    if compression is not None and compression in ['gzip', 'bz2', 'lzma']:
        if compression == 'gzip':
            ext = '.pickle.gz'
            with gzip.open(path + ext, 'wb') as f:
                pickle.dump(model, f)
        elif compression == 'bz2':
            ext = '.pickle.bz2'
            with bz2.BZ2File(path + ext, 'wb') as f:
                pickle.dump(model, f)
        elif compression == 'lzma':
            ext = '.pickle.xz'
            with lzma.open(path + ext, 'wb') as f:
                pickle.dump(model, f)
    else:
        if compression is not None:
            print('Unknown compression method, defaulting to uncompressed pickle format.')
        
        ext = '.pickle'
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    print(f'Successfully saved model to {path + ext}')
    
    return

def load_model(path, compression = None):
    '''
    Loads a serialized pickle object.
    
    args:
        path: str - path to object to deserialize
        compression: None, 'gzip', 'bz2', or 'lzma' - whether the pickled model is compressed using
            the specified protocol; otherwise, the compression is inferred from the file extension
    
    returns: model - deserialized object respresenting a fitted sklearn Model
    '''
    if compression is None:
        ext = path.split('.')[-1]
    else:
        compression_map = {'gzip': 'gz', 'bz2': 'bz2', 'lzma': 'xz', 'pickle' : 'pickle'}
        ext = compression_map[compression]
    
    if ext not in ['pickle', 'gz', 'bz2', 'xz']:
        print(f'Unknown extension type at {path}, model not loaded.')
        
        return
    
    if ext == 'gz':
        with gzip.open(path, 'rb') as f:
            model = pickle.load(f)
    elif ext == 'bz2':
        with bz2.BZ2File(path, 'rb') as f:
            model = pickle.load(f)
    elif ext == 'xz':
        with lzma.open(path, 'rb') as f:
            model = pickle.load(f)
    else:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
    print(f'Successfully loaded model from {path}')
    
    return model

def load_models_of_type(model_name, ext = '.pickle'):
    '''
    Loads all models given a folder/model name.
    
    args:
        model_name: str - string to use to find where the models are located
        ext: str - extension to use for loading models
        
    returns: models - a list of fitted models from the specified directory
    '''
    
    # Define path stem
    path_stem = f'../models/{model_name}/{model_name}'
    
    models = []
    
    # Load in all models
    for resample in ['', '_SMOTE']:
        models.append((f'{model_name}_base{resample}', load_model(f'{path_stem}_base{resample}{ext}')))
        models.append((f'{model_name}_grid{resample}', load_model(f'{path_stem}_grid{resample}{ext}')))
        models.append((f'{model_name}_pca{resample}', load_model(f'{path_stem}_pca{resample}{ext}')))
        
        for k in [5, 7, 9, 11]:
            models.append((f'{model_name}_top{int(k)}{resample}', load_model(f'{path_stem}_top{k:02d}{resample}{ext}')))
    
    return models
