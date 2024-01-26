import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
# https://matplotlib.org/stable/users/explain/customizing.html
def set_pub():
    mpl.rcParams['figure.figsize'] = (8,6)
    mpl.rcParams['figure.constrained_layout.use'] = True
    mpl.rcParams['figure.dpi'] = 250

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)