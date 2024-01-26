from conda_scripts import plot_wet
import pandas as pd


class climate:
    '''
    make an object that can be used to merge climate data with the observations
    '''

    def __init__(self, precip = True):

        if precip:
            climate = plot_wet.load_precip()
            climate = climate.mean(axis=1).to_frame('Precip')
            self._col = 'precip'
        else:
            climate = plot_wet.load_climate()
            climate = climate.loc[:, ('SRP', 'tmean')] - climate.loc[:, ('SRP', 'tmean')].mean()
            climate = climate.to_frame('tmean')
            self._col = 'tmean_anom'

        self.climate = climate
        self.climate_resampled = None
        self.climate_cols = None
        self.precip = precip

    def get_clim(self, df):
        d = df.resample('12M', closed='left').sum().cumsum()
        d = d.T
        cols = [f'{self._col}{ij}' for ij in range(d.shape[1])]

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
        self.climate_cols = list(xnew.columns)

    def add_climate_to_obs(self, df, column = 'Date'):
        assert column in df.columns, f'{column} columns is not in df\n{df.columns}'

        #filter observations that are not covered by the climate obs
        df = df.loc[df.loc[:,column] <= self.climate_resampled.index.max(),:]

        assert self.climate_resampled.isnull().sum().sum()==0

        initial_df = df.copy()
        f = df.shape
        df = pd.merge(df, self.climate_resampled, left_on=column, right_index=True, how = 'left')

        newest_ob = df.loc[:,column].max()
        newest_clim = self.climate_resampled.index.max()
        assert newest_ob<=newest_clim, f'climate data needs to be updated.\ngw obs go to:\n{newest_ob}.\nclimate data to:\n{newest_clim}\n'

        assert df.shape[0] == f[0], 'something went wrong with join of climate data to observations'

        assert initial_df.equals(df.loc[:,initial_df.columns]), print('these are the differences\n')+print(initial_df.compare(df.loc[:,initial_df.columns]))

        return df
