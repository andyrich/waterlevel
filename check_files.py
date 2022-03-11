
import os
import pandas as pd

def check_prediction_files_permission(path, foldername):
    '''
    check if netcdf files can be written
    :param path:
    :return:
    '''

    def ch(f):
        if os.path.exists(f):
            try:
                os.rename(f, f)
                print('Access on file "' + f + '" is available!')
            except OSError as e:
                # print('Access-error on file "' + f + '"! \n' + str(e))
                raise OSError('Access-error on file "' + f + '"! \n' + str(e))

    vals = pd.MultiIndex.from_product([['Deep', 'Shallow'], ['Fall', 'Spring']])
    for v, i in vals:
        filename = os.path.join(path, foldername,
                                f'wl_predictions_{v}_{i}.netcdf')
        ch(filename)
        filename = os.path.join(path, foldername,
                                f'wl_change_predictions_{v}_{i}.netcdf')
        ch(filename)