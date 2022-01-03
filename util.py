import os
import numpy as np
import pandas as pd




def set_options():

    pd.set_option('display.max_columns', None) # display all columns of dataframe
    pd.set_option('display.expand_frame_repr', False) # turn off line wrapping (makes output unreadable imo)



# import the csv file at the end of the passed relative path
def import_data(relative_path):

    working_dir = os.path.dirname(__file__)
    path_to_file = os.path.join(working_dir, relative_path)

    return pd.read_csv(path_to_file)



# returns tuple containing:
#   x_train, y_train
def load_training(datadir):
    xfile = os.path.join(datadir, 'x_train.npy')
    yfile = os.path.join(datadir, 'y_train.npy')

    return np.load(xfile), np.load(yfile)



# returns tuple containing:
#   x_test, y_test
def load_test(datadir):
    xfile = os.path.join(datadir, 'x_test.npy')
    yfile = os.path.join(datadir, 'y_test.npy')

    return np.load(xfile), np.load(yfile)


