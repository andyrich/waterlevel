import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import conda_scripts.make_map as mp
import conda_scripts.rich_gis as rich_gis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import calendar
import cartopy.crs as ccrs
import geopandas as gpd
from conda_scripts.utils.gwl_krig_preprocess import date2year_frac, date2date_frac
import textwrap
from sklearn.metrics import mean_squared_error
import process_climate as pcr

import xarray as xr


def months(season):
    m = {'Spring': 4, 'Fall': 10, 'Fall Early': 9}
    return m[season]


class MapGW:
    def __init__(self, train_pred, train_input, smooth=False, smooth_value=1.0):
        self.train_pred = train_pred
        self.train_input = train_input
        self.smooth = smooth

        if smooth == False:
            smooth_value = -999
        self.smooth_value = smooth_value

        more_info = f"smooth = {smooth}\nsmooth_value = {smooth_value:.2f}"
        self.description = train_input.description + more_info

        self.path = os.path.join('GIS', 'hydro_experiment')

        print(self.description)

    def plotmap(self, yearstep=3, seasons=None, plot_residuals=False, netcdf_only=False):
        '''
        perform map projections for years/seasons noted
        :param yearstep: can be step (int) or range of years to plot
        :param seasons: ['Deep', 'Shallow']
        :param plot_residuals: plot residuals on map
        :param netcdf_only: skip map plotting and save only netcdf
        :return:
        '''

        if seasons is None:
            seasons = ['Spring', 'Fall']

        print(f"saving maps to {os.path.join(self.path, self.train_input.map_foldername)}")
        try:
            os.mkdir(os.path.join(self.path, self.train_input.map_foldername))

        except Exception as e:
            print('failed to make directory')
            print(e)
            pass

        print(f"scaling input data:\t{self.train_input.scale_data}")
        # basins = ['SRP', 'SON', 'PET']

        if isinstance(yearstep, int):
            years = np.arange(2010, 2022, yearstep)
        else:
            years = yearstep

        basins = [self.train_input.basin]
        contours = np.arange(-20, 260, 20)
        first = PlotContour(self.train_pred.m_rk,
                            foldername=self.train_input.map_foldername)

        for season in seasons:
            for deep in ['Deep', 'Shallow', ]:
                for bas in basins:
                    for year in years:
                        first.predict(elev=self.train_input.elev,
                                      pred_col=self.train_pred.pred_col,
                                      year_predict=year,
                                      season=season,
                                      scale_data=self.train_input.scale_data,
                                      scaler=self.train_pred.scaler,
                                      depth_type=deep,
                                      smooth=self.smooth,
                                      dayoffset=self.train_input.dayoffset,
                                      add_climate=self.train_input.add_climate,
                                      n_months=self.train_input.nmonths,
                                      add_temp=self.train_input.add_temp)

                        if netcdf_only:
                            first.save_prediction(first.grid_z2, year, deep, season)
                            print('skipping plotting and only saving netcdf outputs')
                            continue

                        ax = first.map_it(calc=True,
                                          plot_points=True,
                                          contours=contours,
                                          crs=ccrs.epsg(2226),
                                          label_contour=True,
                                          locname=bas + "_MOD")

                        # plot points
                        locs = self.train_input.seas_info.loc[~self.train_input.seas_info.index.duplicated()].loc[:,
                               ['Easting', "Northing", 'rasterelevation', 'Well_Depth_Category']].copy()

                        seas_gdf = get_seas_values(self.train_input.seas_info,
                                                   year, season, locs,
                                                   depth_type=deep)

                        filename = os.path.join(self.path, self.train_input.map_foldername,
                                                f'{bas}_{year}_{first.season}_{deep}.png')

                        if seas_gdf is None:
                            print(f"{year}-{season}--{deep} is not in the columns... ie is not covered by observations\n\n")

                        else:
                            seas_gdf = seas_gdf.to_crs(2226)
                            # modeled points
                            # mod_gdf = seas_gdf[(seas_gdf.index.str.contains('mod'))]
                            # observed points
                            seas_gdf = seas_gdf[~(seas_gdf.index.str.contains('mod'))]

                            if seas_gdf.shape[0] == 0:
                                seas_gdf = None

                        if seas_gdf is None:
                            print(f'there are no observation points for {year} {season} {deep}')
                            pass  # if there are no points to add to map, continue
                        else:
                            # add points to map
                            print(f'adding observation points for {year} {season} {deep}')
                            print('removing duplicated points at observation locations.... kinda crudely')
                            seas_gdf = limit_duplicates(seas_gdf)

                            seas_gdf.loc[:, 'predicted'] = interp_at_obs(xgrid=first.x_stp,
                                                                         ygrid=first.y_stp,
                                                                         z_predicted=first.grid_z2,
                                                                         xout=seas_gdf.geometry.x,
                                                                         yout=seas_gdf.geometry.y)

                            seas_gdf.loc[:, 'Residual'] = seas_gdf.loc[:, 'Observations'] - seas_gdf.loc[:, 'predicted']

                            seas_gdf.loc[:, 'label'] = seas_gdf.apply(
                                lambda x: "{:.0f} ({:+.0f})".format(x['Observations'],
                                                                    x['Residual']), axis=1)

                            seas_gdf.plot(ax=ax, color='k', markersize=5)

                            if plot_residuals:
                                col2plot = 'label'
                                already_str = True
                            else:
                                col2plot = 'Observations'
                                already_str = False

                            label_points(ax, seas_gdf, col2plot,
                                         basin_name=[bas.upper()],
                                         buffer=2000, limit=5, already_str=already_str)

                            seas_gdf.to_file(filename.replace('.png', '.shp'))

                            mean_residual = seas_gdf.loc[:, 'Residual'].mean()
                            rmse = mean_squared_error(seas_gdf.loc[:, 'Observations'], seas_gdf.loc[:, 'predicted'])
                            props = dict(boxstyle='round', facecolor='yellow', alpha=0.5)
                            ax.text(1, 1, f"RMSE = {rmse:.2f}\nMean Residual = {mean_residual:.2f}",
                                    ha='right', va='top', bbox=props, transform=ax.transAxes)

                        # add annotation to outside of graph
                        def twrap(li):
                            f = '\n'.join(textwrap.wrap(li, 30))
                            return f

                        props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
                        ax.text(1, 0, twrap(self.description),
                                ha='right', va='top', bbox=props, transform=ax.transAxes)
                        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
                        ax.text(0, 0, twrap(self.train_pred.description),
                                ha='left', va='top', bbox=props, transform=ax.transAxes)

                        # export gis
                        print(os.path.abspath(filename))

                        plt.savefig(filename, dpi=250, bbox_inches='tight')

                        rich_gis.write_acsii(np.flipud(first.grid_z2),
                                             first.x_stp, first.y_stp, -999,
                                             filename.replace('.png', '.tif'))

                        cntrs_filled_gdf = rich_gis.contours2shp(first.mpl_contours_filled)
                        cntrs_filled_gdf.to_file(filename.replace('.png', '_fill_contours_filled.shp'))

                        rich_gis.raster2contour(filename.replace('.png', '.tif'),
                                                filename.replace('.png', '_cnt20ft.shp'),
                                                20)

                        rich_gis.raster2contour(filename.replace('.png', '.tif'),
                                                filename.replace('.png', '_cnt100ft.shp'),
                                                100)

                        first.save_prediction(first.grid_z2, year, deep, season)

        first.export_prediction(self.path)

        return first


class PlotContour(object):
    def __init__(self, m_rk, foldername):

        self.m_rk = m_rk
        self.predictions = None
        self.foldername = foldername

    def predict(self, elev, pred_col, year_predict, season, scale_data=False, scaler=None, depth_type=None,
                smooth=False, dayoffset=92, add_climate=False, add_temp=True, n_months=36):
        '''

        :param elev:
        :param pred_col:
        :param year_predict:
        :param season:
        :param scale_data:
        :param scaler:
        :param depth_type:
        :param smooth:
        :param dayoffset:
        :param add_climate:
        :param n_months:
        :return:
        '''
        # setup points for running prediction
        df_elev = elev.copy()

        self.year_frac = months(season)
        self.year_predict = year_predict
        self.season = season
        self.plot_title = f"{depth_type} Aquifer" + '\n' + f"{self.season} - {self.year_predict:.0f}"

        # add columns to elev
        out_predict = addcol2elev(df_elev, pred_col, year=year_predict, month=months(season), shallow=depth_type,
                                  dayoffset=dayoffset, add_climate=add_climate, add_temp=add_temp, n_months=n_months)
        out_predict = out_predict.loc[:, pred_col]
        print(out_predict.head(1))

        out_latlon = df_elev.loc[:, 'Easting':'Northing'].values
        print(f"these are the predictors {pred_col}")

        if scale_data:
            out_predict = scaler.transform(out_predict)
        else:
            pass

        df_elev['predicted'] = self.m_rk.predict(out_predict, out_latlon)
        dfgrid = df_elev.pivot_table(values='predicted', index='Northing', columns='Easting')

        if smooth:
            print('smoothing head values ')
            smoothed_head = smooth_head(dfgrid.values)
            self.grid_z2 = smoothed_head
        else:
            self.grid_z2 = dfgrid.values

        self.raw_head = dfgrid.values

        self.x_stp, self.y_stp = dfgrid.columns.values, dfgrid.index.values

    def map_it(self, calc=True, plot_points=True,
               contours=(0, 10, 20, 30, 40), locname="PV",
               label_contour=True, crs=ccrs.epsg(3857),
               maptype='ctx.OpenStreetMap.Mapnik'):

        m = mp.make_map(self.plot_title, projection=crs)
        fig = plt.figure(figsize=(10, 10))
        ax = m.plotloc(fig, locname=locname, maptype=maptype)
        mpl_contours_filled = ax.contourf(self.x_stp, self.y_stp, self.grid_z2, levels=contours, alpha=0.5,
                                          transform=ccrs.epsg(2226),
                                          cmap='nipy_spectral')

        mpl_contour_lines = ax.contour(self.x_stp, self.y_stp, self.grid_z2, levels=contours, alpha=.9,
                                       transform=ccrs.epsg(2226), colors='k')

        if label_contour:
            fmt = '%.0f'

            # class nf(float):
            #     def __repr__(self):
            #         s = f'{self:.1f}'
            #         return f'{self:.0f}' if s[-1] == '0' else s
            #
            # # Recast levels to new class
            # CS.levels = [nf(val) for val in CS.levels]
            ax.clabel(mpl_contour_lines, mpl_contour_lines.levels, inline=True, fmt=fmt, fontsize=10)

        cbar = fig.colorbar(mpl_contours_filled)
        self.ax = ax
        self.mpl_contours_filled = mpl_contours_filled
        self.mpl_contour_lines = mpl_contour_lines
        return ax

    def save_prediction(self, array, year, depth, season, ):
        '''

        :param year:
        :param season:
        :return:
        '''
        # easting, northing = np.meshgrid(self.x_stp, self.y_stp)
        easting, northing = self.x_stp, self.y_stp
        array = np.expand_dims(array, 2)

        ds = make_xr(array, year, depth, season, easting, northing)
        if self.predictions is None:
            print('creating new prediction attr')
            self.predictions = ds
        else:
            print('concatenating predictions')
            self.predictions = xr.concat([self.predictions, ds], dim='time')

    def export_prediction(self, path):
        vals = pd.MultiIndex.from_product([['Deep', 'Shallow'], ['Fall', 'Spring']])
        for v, i in vals:
            print(v, i)

            b = self.predictions.sel(dict(depth=v, season=i))
            b = b.sortby('year')
            filename = os.path.join(path, self.foldername,
                                    f'wl_predictions_{v}_{i}.netcdf')
            print(f'saving simulated heads netcdf to {filename}')
            b.to_netcdf(filename)

            b = b.diff(dim='year')
            filename = os.path.join(path, self.foldername,
                                    f'wl_change_predictions_{v}_{i}.netcdf')
            print(f"saving WL changes netcdf to {filename}")
            print(b)
            b.to_netcdf(filename)


def make_xr(array, year, depth, season, easting, northing):
    ind = pd.MultiIndex.from_arrays([[year], [depth], [season]], names=['year', 'depth', 'season'])
    ds = xr.Dataset(
        {"Head": (["northing", "easting", "time"], array)},
        coords={"northing": (["northing"], northing),
                "easting": (["easting"], easting),
                "time": ind})

    return ds

    # def make_xr(array, year, depth, season, easting, northing):
    #     ind = pd.MultiIndex.from_arrays([[year], [depth], [season]], names=['year', 'depth', 'season'])
    #     ds = xr.Dataset(
    #         {"Head": (["x", "y", "time"], array)},
    #         coords={"northing": (["x", "y"], northing),
    #                 "easting": (["x", "y"], easting),
    #                 "time": ind})
    #
    #     return ds


def addcol2elev(elev, pred_col, year, month, shallow='Shallow', dayoffset=92, add_climate=False, add_temp=False,
                n_months=36):
    #     year_frac = np.float(year_frac)
    df = elev.copy()
    assert month > 0 and month < 13, 'year_frac must be between 1 and 12'
    df.loc[:, 'Date'] = pd.datetime(year, month, 1)
    df.loc[:, 'date_frac'] = date2date_frac(df.loc[:, 'Date'], dayoffset=dayoffset)
    df.loc[:, 'year_frac'] = date2year_frac(df.loc[:, 'Date'], dayoffset=dayoffset)
    df.loc[:, ['Shallow', 'Deep', 'Other']] = 0
    if shallow == 'Shallow':
        df.loc[:, 'Shallow'] = 1
    else:
        df.loc[:, 'Deep'] = 1

    if add_climate:
        climate = pcr.climate()
        climate.resample_climate(n_months=n_months)
        print('adding climate to prediction locations')
        df = climate.add_climate_to_obs(df)

    if add_temp:
        climate = pcr.climate(precip=False)
        climate.resample_climate(n_months=n_months)
        print(f'adding temperature to observation data. using {n_months}')
        df = climate.add_climate_to_obs(df)

    df = df.rename(columns={'RASTERVALU': "rasterelevation"})
    assert df.rasterelevation.mean() > 0
    return df.loc[:, pred_col]


def label_points(ax, gdf, colname, basin_name=None, buffer=None, limit=20, already_str=False):
    if basin_name is None:
        basin_name = ["SRP"]
    from adjustText import adjust_text
    bas = rich_gis.get_active_subbasins()

    bas.index = ['PET', 'SRP', "SON"]

    if buffer is None:
        pass
    else:
        bas.geometry = bas.buffer(buffer)

    gdf = gpd.overlay(gdf, bas.loc[basin_name, :])
    x__ = gdf.geometry.x
    y__ = gdf.geometry.y
    if already_str:
        val__ = gdf.loc[:, colname].values
    else:
        val__ = [f"{xi:.0f}" for xi in gdf.loc[:, colname]]

    texts = [ax.text(x__[i], y__[i], val__[i],
                     transform=ccrs.epsg(2226), color='r')
             for i in range(len(x__))]
    print('labeling points...')
    adjust_text(texts, lim=limit, arrowprops=dict(arrowstyle='-', color='g'))


def spring_fall(df):
    '''
    retun two numpy arrays of true/false for spring fall.
    df must have datetime index
    Args:
        df: dataframe with datetimeindex

    Returns:
    spring/fall numpy arrays
    '''

    spring = np.logical_and((df.index.month >= 3), (df.index.month <= 5))
    fall = np.logical_and((df.index.month >= 10), (df.index.month <= 12))

    return spring, fall


def calc_seasonal(df, station='station name', obs_column="Manual Measurement"):
    '''

    Calculate seasonal median values from oberved data

    Args:
        df: df with a datetimeindex with a "station" column and a measurement column
        station: name of column to get median
        obs_column:

    Returns:

    '''

    spring, fall = spring_fall(df)
    filt_seas = np.logical_or(spring, fall)
    maindf_long = df.loc[filt_seas, :]

    df_out = maindf_long.loc[:, [station, obs_column]]. \
        groupby([station, pd.Grouper(freq='2QS')]).median().unstack()

    df_out.columns = df_out.columns.droplevel()

    df_out.columns = [pd.to_datetime(x).strftime('%Y-%b') for x in df_out.columns.values]

    cols_f = df_out.columns

    for m in np.arange(1, 7):
        month = calendar.month_name[m][0:3]
        cols_f = [x.replace(f'-{month}', '-Spring') for x in cols_f]

    for m in np.arange(7, 13):
        month = calendar.month_name[m][0:3]
        cols_f = [x.replace(f'-{month}', '-Fall') for x in cols_f]

    df_out.columns = cols_f

    return df_out


def get_seas_values(df, year, season, locs, depth_type=None):
    '''
    get seasonal values from observed data
    '''
    seas_fiol = calc_seasonal(df.reset_index().set_index('Timestamp'), "Station Name")

    # print('seas_fiol:')
    # print(seas_fiol.head(5))

    if f'{year}-{season}' in seas_fiol.columns:

        seas_fiol = seas_fiol.loc[:, f'{year}-{season}'].dropna().to_frame("Observations")

        if isinstance(depth_type, str):
            locs = locs[locs.Well_Depth_Category.str.contains(depth_type)]

        seas_fiol = seas_fiol.join(locs)
        seas_fiol = seas_fiol.dropna()
        seas_gdf = gpd.GeoDataFrame(seas_fiol, geometry=gpd.points_from_xy(seas_fiol.Easting, seas_fiol.Northing),
                                    crs=2226)

        return seas_gdf

    else:
        return None


# these are for plotting monthly points
def df_resampleMonth(seas_info):
    df = seas_info.set_index('Timestamp', append=True).loc[:, 'Manual Measurement'].unstack('Timestamp')
    return df


def get_month_values(df, year, month, locs):
    seas_fiol = df_resampleMonth(df)
    seas_fiol = seas_fiol.loc[:, f'{year}-{month}-01'].dropna().to_frame("Observations")
    seas_fiol = seas_fiol.join(locs)
    seas_gdf = gpd.GeoDataFrame(seas_fiol, geometry=gpd.points_from_xy(seas_fiol.Easting, seas_fiol.Northing), crs=2226)

    return seas_gdf


def smooth_head(array, sigma=1):
    from scipy import ndimage

    # array = obj.grid_z2
    # fill in missing values
    col_mean = np.nanmean(array, axis=0)
    # Find indices that you need to replace
    missing = np.isnan(array)
    inds = np.where(missing)
    # Place column means in the indices. Align the arrays using take
    array[inds] = np.take(col_mean, inds[1])

    img = ndimage.gaussian_filter(array, sigma=sigma)
    img[missing] = np.nan

    return img


def export_gis(filename, wlmap):
    # filename = filename + colname + ' ' + wlmap.modelname

    # file_path_raster = os.path.join('GIS', 'hydro_experiment',folder, filename)
    # file_path_csv = file_path_raster.replace('.tif', '.csv')

    rich_gis.write_acsii(np.flipud(wlmap.grid_z2),
                         wlmap.x_stp, wlmap.y_stp, -999,
                         filename)

    # write_csv_gis(wlmap.df, file_path_csv, colname, easting='Easting', northing='Northing')


def write_csv_gis(df, filename, columns, easting='Easting', northing='Northing', dropna=True):
    '''
    create a file with:
    easting, northing, value.
    df is dataframe
    filename is name of file to be created
    columns is the name of the column to be exported
    easting/northing are names of columns to be named easting and northing.
    dropna will remove rows with nan
    '''
    cols = pd.Index([easting, northing]).append(pd.Index([columns]))

    df = df.loc[:, cols]

    df = df.rename(columns={columns: 'Value', easting: 'Easting', northing: 'Northing'})

    if dropna:
        'drop rows with nan'
        df = df.dropna(axis=0)

    df.to_csv(filename, index=False)


def limit_duplicates(gdf):
    gdf.loc[:, 'Easting_rrr'] = gdf.loc[:, 'Easting'].apply(lambda x: round(x, -1)).astype(int)
    gdf.loc[:, 'Northing_rrr'] = gdf.loc[:, 'Northing'].apply(lambda x: round(x, -1)).astype(int)

    d = gdf.duplicated(['Easting_rrr', 'Northing_rrr'])

    gdf = gdf.loc[~d].drop(columns=['Easting_rrr', 'Northing_rrr'])

    return gdf


def interp_at_obs(xgrid, ygrid, z_predicted, xout, yout):
    '''
    predict head at observation locations using 2-d surface of heads
    :param xgrid: array from dfgrid.x_stp
    :param ygrid: array from dfgrid.y_stp
    :param z_predicted: array from dfgrid.grid_z2
    :param xout: Input coordinates
    :param yout: Input coordinates
    :return:
    '''
    # from scipy.interpolate import interp2d
    from scipy.interpolate import RegularGridInterpolator

    p = RegularGridInterpolator((ygrid, xgrid), z_predicted)

    pts = list(zip(yout, xout))
    predicted = p(pts)

    # print([f"shape = {x.shape}\n" for x in [xgrid, ygrid, z_predicted]])

    assert predicted.size == xout.size, f"{predicted.size} = predicted.size\n{xout.size} = xout.size"

    return predicted
