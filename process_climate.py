from conda_scripts import plot_wet
import pandas as pd


class climate:
    '''
    make an object that can be used to merge climate data with the observations
    '''

    def __init__(self):
        climate = plot_wet.load_precip()
        climate = climate.mean(axis=1).to_frame('Precip')
        self.climate = climate
        self.climate_resampled = None
        self.climate_cols = None

    def get_clim(self, df):
        d = df.resample('12M', closed='left').sum().cumsum()
        d = d.T
        cols = [f'clim{ij}' for ij in range(d.shape[1])]

        d.columns = cols
        return d

    def resample_climate(self, n_months=36):
        xnew = pd.DataFrame()
        for i in range(n_months, self.climate.shape[0]):
            a = self.climate.iloc[i - n_months:i, :]
            a = self.get_clim(a)
            a.index = [self.climate.index[i]]
            xnew = xnew.append(a)

        # make sure timestamp is at beginning of month to match the observed data
        xnew.index = xnew.index + pd.tseries.offsets.MonthBegin()

        self.climate_resampled = xnew
        self.climate_cols = xnew.columns

    def add_climate_to_obs(self, df, column = 'Date'):
        assert column in df.columns, f'{column} columns is not in df\n{df.columns}'

        initial_df = df.copy()
        f = df.shape
        df = pd.merge(df, self.climate_resampled, left_on=column, right_index=True, how = 'left')

        assert df.shape[0] == f[0], 'something went wrong with join of climate data to observations'

        assert initial_df.equals(df.loc[:,initial_df.columns]), print('these are the differences\n')+print(initial_df.compare(df.loc[:,initial_df.columns]))

        return df
