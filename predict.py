import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import process_climate as pcr
import conda_scripts.utils.krig_dataset as lgp
import conda_scripts
import pandas as pd
import numpy as np
import os
import pickle
from sklearn import preprocessing
from pykrige.rk import RegressionKriging
import sklearn.ensemble as ens_
import geopandas as gpd
import rasterstats as rs
from conda_scripts.utils.gwl_krig_preprocess import date2year_frac, date2date_frac
import plot_hydros
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

class krig_predict:
    '''
    krig datasets. uses inputs from kriging process
    '''

    def __init__(self, train, option='universal', pred_col=None, add_weights=True, modweight=.5):
        '''
        import krigobject
        :param train: object from krig
        :param option: which type of regression solution
        '''

        if pred_col is None:
            pred_col = ['rasterelevation',
                        'Geol_Krig',
                        'isostatic',
                        'Complete_B',
                        'Simple_Bou',
                        'date_frac',
                        'year_frac',
                        'Shallow',
                        'slope',
                        'Deep',
                        'Other']

        xy_col = ['Easting', 'Northing']
        targ_col = ['Manual Measurement']

        all_col = []
        all_col.extend(pred_col)
        all_col.extend(xy_col)
        all_col.extend(targ_col)

        self.targ_col = targ_col
        self.all_col = all_col
        self.xy_col = xy_col

        self.modelname = option
        self.m_rk = set_model(option=option)

        self.add_weights = add_weights

        self.train = train

        self.load_rmp = load_rmp

        self.modweight = modweight

        if self.train.add_climate:
            allin = all([x in pred_col for x in self.train.climate_cols])
            if allin:
                print('Climate variables already in pred_cols ')
            else:
                print('adding climate variables to prediciton columns b/c add_climate = True')
                pred_col.extend(list(self.train.climate_cols))

        self.pred_col = pred_col

        self.predicted = None
        self.fitted_obs = None

        self.description = f"modelname= {option}\n\
        add_weights= {add_weights}\n\
        mod_weight= {modweight}\n\
        pred_col = {pred_col}\n"

        print(f"\nthe prediction columns are: {pred_col}\n")

    def setup_prediction(self, test_size=False):
        '''
        create inputs for kriging
        :param test_size: test_sizefloat or int, default=None
            If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
            to include in the test split. If int, represents the absolute number of test samples.
        :return: x_train, p_train, target
        '''
        print('\ndoing prediction setup\n')
        x_train, p_train, target, weights = conda_scripts.utils.regression_krig.setup_prediction(self.train.seas_info,
                                                                                                 pred_col=self.pred_col,
                                                                                                 targ_col=self.targ_col,
                                                                                                 modweight=self.modweight)

        np.save(os.path.join('regression_data', 'x_train.npy'), x_train)
        np.save(os.path.join('regression_data', 'p_train.npy'), p_train)
        np.save(os.path.join('regression_data', 'target.npy'), target)

        if test_size > 0:
            x_test, x_train, p_test, p_train, target_test, target, test_weights, weights = train_test_split(
                x_train,
                p_train,
                target,
                weights,
                test_size=test_size,
                random_state=42)

            print(f'using {test_size} of all data for testing purposes' * 3)
        else:
            print('using the ENTIRE dataset for fitting purposes')
            x_test, p_test, target_test, test_weights = None, None, None, None

        print(f"the shape of the inputs is {p_train.shape}\n\n\n")

        self.x_train = x_train
        self.p_train = p_train
        self.target = target
        if self.add_weights:
            self.weights = weights
        else:
            self.weights = None

        self.x_test = x_test
        self.p_test = p_test
        self.target_test = target_test
        self.test_weights = test_weights

        if self.add_weights:
            self.test_weights = test_weights
        else:
            self.test_weights = None

    def run_fit(self):
        print('\n\nrunning predicitons!' * 3)
        print(self.train.hydros_foldname)
        try:
            os.mkdir(os.path.join('GIS', 'hydro_experiment', self.train.hydros_foldname))
        except:
            pass

        if self.train.scale_data:
            self.scaler = preprocessing.StandardScaler().fit(self.p_train)

            self.p_train_scaled = self.scaler.transform(self.p_train.copy())

            self.p_test_scaled = self.scaler.transform(self.p_test.copy())

            self.m_rk.fit(self.p_train_scaled, self.x_train, self.target, weights=self.weights)

            if hasattr(self.m_rk, 'socre'):
                score = self.m_rk.score(self.p_test_scaled, self.x_test, self.target_test)
            else:
                score = 0.0

            [print(f'shape {xi.shape}') for xi in [self.p_train_scaled, self.x_train, self.target, self.weights]]

            print('refitting with all data')
            __allp = np.concatenate([self.p_train_scaled, self.p_test_scaled])
            __allx = np.concatenate([self.x_train, self.x_test])
            __allt = np.concatenate([self.target, self.target_test])

            if self.add_weights:
                __allweights = np.concatenate([self.weights, self.test_weights])
            else:
                __allweights = None

            [print(f'shape {xi.shape}') for xi in [__allp, __allx, __allt, __allweights]]

            self.m_rk.fit(__allp, __allx, __allt, weights=__allweights)


        else:
            self.scaler = None
            # if self.modelname.__contains__('regress_only') and self.add_weights:

            print('adding weights to fitted model')
            self.m_rk.fit(self.p_train, self.x_train, self.target, weights=self.weights)

            score = self.m_rk.score(self.p_test, self.x_test, self.target_test)

        self.score = score
        print(f"score is {score}")

        pickle_file = os.path.join('GIS', 'hydro_experiment', self.train.hydros_foldname,
                                   f'pikckleobj_trained_{self.train.basin}.pickle')
        pick = open(pickle_file, 'wb')
        print('saving pickle object')
        pickle.dump(self.m_rk, pick)
        pick.close()

    def export_predicted(self):
        df = self.train.seas_info.copy()

        x_train, p_train, target, weights = conda_scripts.utils.regression_krig.setup_prediction(self.train.seas_info,
                                                                                                 pred_col=self.pred_col,
                                                                                                 targ_col=self.targ_col,
                                                                                                 modweight=self.modweight)

        p_train = self.scaler.transform(p_train)

        fitted = self.m_rk.predict(p_train, x_train)

        df.loc[:,'predicted'] = fitted

        csv = os.path.join('GIS', 'hydro_experiment', self.train.hydros_foldname,
                                   f'seasinfo_w_predicted_{self.train.basin}.csv')
        print(f'exporting seas_info with predicted values to {csv}')
        df.to_csv(csv)
        print('done\n')

    def run_prediction_for_hydros(self):
        rmp_hydro = predict_rmp_hydros(train=self.train,
                                       pred_col=self.pred_col,
                                       basin=self.train.basin,
                                       model=self.m_rk,
                                       scaler=self.scaler,
                                       slope=self.train.use_slope,
                                       add_climate=self.train.add_climate,
                                       n_months=self.train.nmonths,
                                       dayoffset=self.train.dayoffset,
                                       add_temp=self.train.add_temp)

        self.predicted = rmp_hydro

    def plot_hydros(self, plot_train=False):
        if plot_train:
            observed = self.train.seas_info
        else:
            observed = None

        plot_hydros.plot_rmp_hydro_pred(self.train.hydros_foldname, self.predicted, observed=observed, fancy=True,
                                        errors_keep_going = False)

        print('done!\n' * 3)


def predict_rmp_hydros(train, pred_col, basin='SRP', model=None, scaler=None, slope=False,
                       monthly=True, add_climate=True, n_months=36,
                       dayoffset=30, add_temp=True):
    obs, fnames = load_rmp(train.allinfo)

    newname = fnames['SRP'].replace('.shp', '_402.shp')
    obs = obs[np.isfinite(obs.geometry.x)]
    obs.loc[:, 'Easting'] = obs.geometry.x
    obs.loc[:, 'Northing'] = obs.geometry.y

    unq_sites = obs.Site.unique()
    obs = obs[obs.Site == basin]
    assert obs.shape[0] > 0, f'bad filtering of rmp points with {basin}. options are \n{unq_sites}'

    obs.loc[:, ['Station_Na', 'geometry']].to_file(newname)
    obs.loc[:, 'rasterelevation'] = obs.loc[:, ['TOC_Elevat', 'Well_TOC_E']].mean(axis=1)

    if slope:
        obs_slope = rs.point_query(newname, train.slope_tiff_elev)
        print('getting slope')
    else:
        obs_slope = np.nan

    obs.loc[:, 'slope'] = obs_slope

    #### adding geophysical data here
    obs = lgp.do_krig_all(obs, obs.loc[:, 'Easting'], obs.loc[:, 'Northing'])
    obs = train.geol.add_geol_to_gdf(obs)
    print(f'adjusting datetime by adjusting forward {dayoffset} days')
    base_df = obs.drop_duplicates(['Easting', 'Northing'])
    dfall = pd.DataFrame()
    for year in np.arange(1975, 2024):
        if monthly:
            for month_ in np.arange(1, 13):
                t = base_df.copy()
                t.loc[:, 'ts'] = pd.datetime(year, month_, 1)
                t.loc[:, 'date_frac'] = date2date_frac(t.loc[:, 'ts'], dayoffset=dayoffset)
                t.loc[:, 'year_frac'] = date2year_frac(t.loc[:, 'ts'], dayoffset=dayoffset)
                dfall = dfall.append(t)
            else:
                for month_ in [4, 10]:
                    t = base_df.copy()
                    t.loc[:, 'ts'] = pd.datetime(year, month_, 1)
                    t.loc[:, 'date_frac'] = date2date_frac(t.loc[:, 'ts'], dayoffset=dayoffset)
                    t.loc[:, 'year_frac'] = date2year_frac(t.loc[:, 'ts'], dayoffset=dayoffset)
                    dfall = dfall.append(t)

    if add_climate:
        climate = pcr.climate(precip=True)
        climate.resample_climate(n_months=n_months)
        print('adding climate to prediction locations')
        dfall = climate.add_climate_to_obs(dfall, column='ts')

    if add_temp:
        climate = pcr.climate(precip=False)
        climate.resample_climate(n_months=n_months)
        print(f'adding temperature to observation data. using {n_months}')
        dfall = climate.add_climate_to_obs(dfall, column='ts')

    # p_pred = dfall.loc[:, pred_col].fillna(0.).values
    p_pred = dfall.loc[:, pred_col].fillna(0.).values
    x_pred = dfall.loc[:, ['Easting', 'Northing']].values

    if pd.DataFrame(np.isinf(p_pred), columns=pred_col).sum().sum() > 0:
        raise Exception(f'{pd.DataFrame(np.isinf(p_pred), columns=pred_col).sum()} are inf')

    if pd.DataFrame(p_pred, columns=pred_col).isnull().sum().sum() > 0:
        raise Exception(f'{pd.DataFrame(p_pred, columns=pred_col).isnull()} are null')

    if pd.Series(x_pred.reshape(-1)).isnull().sum():
        raise Exception('nulls in prediction x_pred')

    if model is None:
        print('returning just locations of RMP locations')
        return base_df
    print('doing predictions')

    if scaler is None:
        fitted = model.predict(p_pred, x_pred)
    else:
        print('using scaler in the predictions')
        p_pred_scaled = scaler.transform(p_pred)
        fitted = model.predict(p_pred_scaled, x_pred)

    dfall.loc[:, 'predicted'] = fitted
    # dfall.loc[:, 'datetime'] = pd.to_datetime(dfall.date_frac.apply(conda_scripts.utils.regression_krig.dec2dt))

    dfall.loc[:, 'datetime'] = dfall.loc[:, 'ts']

    print('done w predictions')

    return dfall


def set_model(m=None, option='a'):
    import sklearn.ensemble as ens_
    if m is None:
        m = ens_.ExtraTreesRegressor(verbose=True, random_state=1)  # best

    if option == 'a':
        m_rk = RegressionKriging(regression_model=m,
                                 n_closest_points=10,
                                 variogram_model='linear',
                                 method='ordinary',
                                 weight=True)
    elif option == 'b':
        m_rk = RegressionKriging(regression_model=m,
                                 n_closest_points=10,
                                 variogram_model='linear')
    elif option == 'c':
        m_rk = RegressionKriging(regression_model=m,
                                 n_closest_points=10,
                                 variogram_model='gaussian',
                                 method='ordinary',
                                 weight=True)

    elif option == 'exponential':
        m_rk = RegressionKriging(regression_model=m,
                                 n_closest_points=10,
                                 variogram_model='exponential',
                                 method='ordinary',
                                 weight=True)
    elif option == 'exactfalse':
        m_rk = RegressionKriging(regression_model=m,
                                 n_closest_points=10,
                                 variogram_model='linear',
                                 method='ordinary',
                                 weight=True,
                                 exact_values=False)
    elif option == 'universal':
        m_rk = RegressionKriging(regression_model=m,
                                 n_closest_points=10,
                                 variogram_model='linear',
                                 method='universal',
                                 weight=True,
                                 pseudo_inv_type='pinv',
                                 pseudo_inv=True,
                                 exact_values=False)
    elif option == 'universal_linearregression':
        from sklearn.linear_model import LinearRegression
        m_rk = RegressionKriging(regression_model=LinearRegression(),
                                 n_closest_points=10,
                                 variogram_model='linear',
                                 method='universal',
                                 weight=False)

    elif option == 'SVR':
        from sklearn.svm import SVR
        m_rk = RegressionKriging(regression_model=SVR(),
                                 n_closest_points=10,
                                 variogram_model='gaussian',
                                 method='universal',
                                 weight=False,
                                 verbose=True)
    elif option == 'regress_only':
        m_rk = pure_regress_model()
    elif option == 'adaboost':
        m_rk = adaboost()
    elif option == 'GradientBoostingRegressor':
        m_rk = gradientboost()
    elif option == 'ann':
        m_rk = ann()
    elif option =='svm':
        m_rk = svm()


    else:
        raise AssertionError(f"option {option} not found ")

    return m_rk

class svm:

    def __init__(self):
        self.model = SVR()

    def fit(self, p, x, y, weights=None):
        print('concatenating inputs')
        X = np.hstack([p, x])
        print('fitting inputs')
        self.model.fit(X, y, sample_weight = weights)
        print('done fitting')

    def predict(self, p_pred, xpred):
        print('doing predictions...')
        X_pred = np.hstack([p_pred, xpred])
        pred = self.model.predict(X_pred)
        print('done predicting')
        return pred


class ann:
    '''
    https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html

    '''


    def __init__(self):
        self.model = MLPRegressor(random_state=1)

    def fit(self, p, x, y, weights=None):
        print('concatenating inputs')
        X = np.hstack([p, x])
        print('fitting inputs')
        self.model.fit(X, y)
        print('done fitting')

    def predict(self, p_pred, xpred):
        print('doing predictions...')
        X_pred = np.hstack([p_pred, xpred])
        pred = self.model.predict(X_pred)
        print('done predicting')
        return pred


class gradientboost:
    '''
    Fit the regression method

    Parameters
    ----------
    p: ndarray
    (Ns, d) array of predictor variables (Ns samples, d dimensions)
    for regression
    x: ndarray
    ndarray of (x, y) points. Needs to be a (Ns, 2) array
    corresponding to the lon/lat, for example 2d regression kriging.
    array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
    y: ndarray
    array of targets (Ns, )
    '''

    def __init__(self):
        self.model = ens_.GradientBoostingRegressor(random_state=1)

    def fit(self, p, x, y, weights=None):
        print('concatenating inputs')
        X = np.hstack([p, x])
        print('fitting inputs')
        self.model.fit(X, y, sample_weight=weights)
        print('done fitting')

    def predict(self, p_pred, xpred):
        print('doing predictions...')
        X_pred = np.hstack([p_pred, xpred])
        pred = self.model.predict(X_pred)
        print('done predicting')
        return pred


class adaboost:
    '''
    Fit the regression method

    Parameters
    ----------
    p: ndarray
    (Ns, d) array of predictor variables (Ns samples, d dimensions)
    for regression
    x: ndarray
    ndarray of (x, y) points. Needs to be a (Ns, 2) array
    corresponding to the lon/lat, for example 2d regression kriging.
    array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
    y: ndarray
    array of targets (Ns, )
    '''

    def __init__(self):
        self.model = ens_.AdaBoostRegressor(random_state=1)

    def fit(self, p, x, y, weights=None):
        print('concatenating inputs')
        X = np.hstack([p, x])
        print('fitting inputs')
        self.model.fit(X, y, sample_weight=weights)
        print('done fitting')

    def predict(self, p_pred, xpred):
        print('doing predictions...')
        X_pred = np.hstack([p_pred, xpred])
        pred = self.model.predict(X_pred)
        print('done predicting')
        return pred


def check_inputs(X):
    if (np.isinf(X).any() or not np.isfinite(X).all()):
        infsum = np.isinf(X).sum()
        finsum = ~np.isfinite(X).sum()
        raise ValueError(f"num inf: {infsum}\nnum not finite: {finsum}")


class pure_regress_model:
    '''
    Fit the regression method

    Parameters
    ----------
    p: ndarray
    (Ns, d) array of predictor variables (Ns samples, d dimensions)
    for regression
    x: ndarray
    ndarray of (x, y) points. Needs to be a (Ns, 2) array
    corresponding to the lon/lat, for example 2d regression kriging.
    array of Points, (x, y, z) pairs of shape (N, 3) for 3d kriging
    y: ndarray
    array of targets (Ns, )
    '''

    def __init__(self):
        self.model = ens_.ExtraTreesRegressor(verbose=True, random_state=1)

    def fit(self, p, x, y, weights=None):
        check_inputs(p)
        check_inputs(x)
        check_inputs(y)
        print('concatenating inputs')
        X = np.hstack([p, x])
        print('fitting inputs')
        self.model.fit(X, y, sample_weight=weights)
        print('done fitting')

    def predict(self, p_pred, xpred):
        check_inputs(p_pred)
        check_inputs(xpred)
        print('doing predictions...')
        X_pred = np.hstack([p_pred, xpred])
        pred = self.model.predict(X_pred)
        print('done predicting')
        return pred

    def score(self, p, x, y):
        print('concatenating inputs')
        X = np.hstack([p, x])
        print('fitting inputs')
        score = self.model.score(X, y)
        return score


def categorize_depths_inputs(df, fillvalue='Other'):
    '''
    reclassify well depth to shallow/deep/other
    '''
    if 'Well_Depth_Category_original' in df.columns:
        col = 'Well_Depth_Category_original'
    else:
        col = 'Well_Depth_Category'

    df.loc[:, col] = df.loc[:, col].fillna(fillvalue)

    c = df.loc[:, col].str.contains('Shal')
    df.loc[c, 'Depth_Reclass'] = "Shallow"
    c = df.loc[:, col].str.contains('Medium')
    df.loc[c, 'Depth_Reclass'] = "Deep"
    c = df.loc[:, col].str.contains('Deep')
    df.loc[c, 'Depth_Reclass'] = "Deep"
    c = df.loc[:, col].str.contains('Other')
    df.loc[c, 'Depth_Reclass'] = "Other"
    df.loc[:, 'Depth_Reclass'] = df.loc[:, 'Depth_Reclass'].fillna(fillvalue)
    df.loc[:, 'Well_Depth_Category'] = df.loc[:, 'Depth_Reclass']

    return df.loc[:, col]


def categorize_depths(df, fillvalue='Shallow'):
    '''
    reclassify well depth to shallow/deep/other
    '''
    if 'Well_Depth_Category_original' in df.columns:
        col = 'Well_Depth_Category_original'
    else:
        col = 'Well_Depth_Category'

    df.loc[:, col] = df.loc[:, col].fillna(fillvalue)

    c = df.loc[:, col].str.contains('Shal')
    df.loc[c, 'Depth_Reclass'] = "Shallow"
    c = df.loc[:, col].str.contains('Medium')
    df.loc[c, 'Depth_Reclass'] = "Deep"
    c = df.loc[:, col].str.contains('Deep')
    df.loc[c, 'Depth_Reclass'] = "Deep"
    c = df.loc[:, col].str.contains('Other')
    df.loc[c, 'Depth_Reclass'] = "Other"
    df.loc[:, 'Depth_Reclass'] = df.loc[:, 'Depth_Reclass'].fillna(fillvalue)

    df.loc[:, 'Well_Depth_Category_original'] = df.loc[:, 'Well_Depth_Category']
    df.loc[:, 'Well_Depth_Category'] = df.loc[:, 'Depth_Reclass']
    df = df.drop(columns='Depth_Reclass')

    print('created dummies from predicted locations')
    df = pd.get_dummies(df.loc[:, col])

    return df


def check_depth_dumm(df):
    '''
    make sure all depth types are in dummies
    '''

    for col in ['Shallow', 'Deep', 'Other']:
        if col in df.columns:
            pass
        else:
            df.loc[:, col] = 0
    print('done checking for all depth types')

    return df


def add_dummiestopred(df):
    dum_dep___ = categorize_depths(df)

    assert df.shape[0] == dum_dep___.shape[0], f'something went wrong with dummy\nshape dum_dep___ =\
    {df.shape[0]},\nnew = {dum_dep___.shape[0]}'

    dfout = df.copy().join(dum_dep___)
    assert df.shape[0] == dfout.shape[
        0], f'something went wrong with shape\nshape og = {df.shape[0]},\nnew = {dfout.shape[0]}'
    return dfout


def load_rmp(allinfo):
    fnames = {"SRP": r"C:\GIS\shapefiles\wells\RMP_wells\SRP_S_Shp\SRP_RMPs_Shallow.shp",
              "SV": r"C:\GIS\shapefiles\wells\RMP_wells\SV_S_Shp\SV_RMPs_Shallow.shp",
              "PV": r"C:\GIS\shapefiles\wells\RMP_wells\Pet_shp\PET_RMPs.shp",
              "SWD": r"C:\GIS\shapefiles\wells\RMP_wells\SWD\TSS\SRP_TSSWells_Surveyed.shp",
              "TSS": r"C:\GIS\shapefiles\wells\RMP_wells\TSS\SRP_TSSWells_Surveyed.shp"}
    obs = gpd.read_file(fnames['SRP'])
    obs = obs.to_crs(2226)
    sv = gpd.read_file(fnames["SV"])
    sv = sv.to_crs(2226)
    pv = gpd.read_file(fnames["PV"])
    pv = pv.to_crs(2226)
    swd = gpd.read_file(fnames["SWD"])
    swd = swd.to_crs(2226)
    tss = gpd.read_file(fnames["TSS"])
    tss = tss.to_crs(2226)

    obs = obs.assign(welltype='SRP RMP').assign(Site='SRP')
    sv = sv.assign(welltype='SV RMP').assign(Site='SON')
    pv = pv.assign(welltype='PET RMP').assign(Site='PET')
    swd = swd.assign(welltype='SWD RMP').assign(Site='SRP')
    tss = tss.assign(welltype='TSS RMP').assign(Site='SRP')

    obs = obs.append(sv).append(pv).append(swd).append(tss)

    obs = obs.drop_duplicates('Station_Na')
    #     print('---'*3)
    #     print(obs[obs.Station_Na=='SRP0714'].loc[:,'Well_Dep_1'])
    # replace the depth categories from those in the allinfo df
    obs.loc[:, 'Well_Depth_Category'] = np.nan

    __f = allinfo[allinfo.index.isin(obs.Station_Na)].loc[:, ['Well_Depth_Category']]

    obs = obs.set_index('Station_Na').combine_first(__f)
    obs.index = obs.index.set_names(['Station_Na'])
    obs = obs.reset_index()

    obs = add_dummiestopred(obs)
    obs = check_depth_dumm(obs)

    return obs, fnames
