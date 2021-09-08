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

def months(season):
    m = {'Spring': 4, 'Fall': 10, 'Fall Early': 9}
    return m[season]



class MapGW:

    def __init__(self, train_pred, train_input,smooth = False):
        self.train_pred = train_pred
        self.train_input = train_input
        self.smooth = smooth

    def plotmap(self, yearstep=3, seasons=['Spring', 'Fall']):
        path = os.path.join('GIS', 'hydro_experiment')

        print(f"saving maps to {self.train_input.map_foldername}")
        try:
            os.mkdir(os.path.join(path, self.train_input.map_foldername))

        except Exception as e:
            print('failed to make directory')
            print(e)
            pass

        print(f"scaling input data:\t{self.train_input.scale_data}")
        # basins = ['SRP', 'SON', 'PET']

        if isinstance(yearstep, int):
            years = np.arange(2010, 2021, yearstep)
        else:
            years = yearstep

        basins = [self.train_input.basin]

        for season in seasons:
            for deep in ['Deep', 'Shallow', ]:
                for bas in basins:
                    for year in years:
                        first = plot_cntr(self.train_pred.m_rk, self.train_input.seas_info,
                                          f'{deep} Aquifer Co-Krig with ')

                        first.predict(elev=self.train_input.elev,
                                      pred_col=self.train_pred.pred_col,
                                      year_predict=year,
                                      season=season,
                                      scale_data=self.train_input.scale_data,
                                      scaler=self.train_pred.scaler,
                                      depth_type=deep,
                                      smooth=self.smooth,
                                      dayoffset=self.train_input.dayoffset)

                        ax = first.map_it(calc=True,
                                          plot_points=True,
                                          contours=np.arange(-20, 260, 20),
                                          crs=ccrs.epsg(2226),
                                          label_contour=True,
                                          locname=bas + "_MOD")

                        # plot points
                        locs = self.train_input.seas_info.loc[~self.train_input.seas_info.index.duplicated()].loc[:,
                               ['Easting', "Northing", 'rasterelevation', 'Well_Depth_Category']].copy()

                        seas_gdf = get_seas_values(self.train_input.seas_info,
                                                   year, season, locs,
                                                   depth_type=deep)

                        filename = os.path.join(path, self.train_input.map_foldername,
                                                f'{bas}_{year}_{first.season}_{deep}.png')

                        if seas_gdf is None:
                            print(f"{year}-{season}  is not in the columns... ie is not covered by observations\n\n")

                        else:

                            seas_gdf = seas_gdf.to_crs(2226)
                            seas_gdf.plot(ax=ax, color='k', markersize=5)

                            label_points(ax, seas_gdf[~(seas_gdf.index.str.contains('mod'))], 'Observations', basin_name=[bas.upper()],
                                         buffer=2000, limit=5)

                            seas_gdf.to_file(filename.replace('.png', '.shp'))

                        print(filename)

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

                        # cntrs_ = rich_gis.get_contours_gdf(first.mpl_contour_lines)
                        # cntrs_.to_file(filename.replace('.png', '_fill_contours_lines.shp'))


class plot_cntr(object):
    def __init__(self, m_rk, seas_info, plot_title):
        # self.modelname = m.__class__.__name__
        self.plot_title = plot_title
        self.m_rk = m_rk
        # print('regression model:', self.modelname)

    def predict(self, elev, pred_col, year_predict, season, scale_data=False, scaler=None, depth_type=None,
                smooth=False, dayoffset = 92):
        # setup points for running prediction
        df_elev = elev.copy()

        # m =
        self.year_frac = months(season)
        self.year_predict = year_predict
        self.season = season

        self.plot_title = self.plot_title + '\n' + f"{self.season} - {self.year_predict:.0f}"

        # add columns to elev
        out_predict = addcol2elev(df_elev, pred_col, year=year_predict, month=months(season), shallow=depth_type,
                                  dayoffset=dayoffset)
        out_predict = out_predict.loc[:, pred_col]
        print(out_predict.head(1))

        out_latlon = df_elev.loc[:, 'Easting':'Northing'].values
        print(f"these are the predictors {pred_col}")

        if scale_data:
            print(pd.DataFrame(out_predict).describe())
            out_predict = scaler.transform(out_predict)
            print('scaling inputs using scaler')
            print(pd.DataFrame(out_predict).describe())
        else:
            pass

        df_elev['predicted'] = self.m_rk.predict(out_predict, out_latlon)
        dfgrid = df_elev.pivot_table(values='predicted', index='Northing', columns='Easting')

        if smooth:
            print('smoothing head values ' * 3)
            smoothed_head = smooth_head(dfgrid.values)
            self.grid_z2 = smoothed_head
        else:
            self.grid_z2 = dfgrid.values

        # self.dfgrid = dfgrid

        self.x_stp, self.y_stp = dfgrid.columns.values, dfgrid.index.values

    def map_it(self, calc=True, plot_points=True,
               contours=20, locname="PV",
               label_contour=True, crs=ccrs.epsg(3857),
               maptype='ctx.OpenStreetMap.Mapnik'):

        m = mp.make_map(self.plot_title, projection=crs)
        fig = plt.figure(figsize=(10, 10))
        ax = m.plotloc(fig, locname=locname, maptype=maptype)
        mpl_contours_filled = ax.contourf(self.x_stp, self.y_stp, self.grid_z2, levels=contours, alpha=0.5,
                                          transform=ccrs.epsg(2226),
                                          cmap='nipy_spectral')

        mpl_contour_lines = ax.contour(self.x_stp, self.y_stp, self.grid_z2, levels=contours, alpha=.9,
                                       transform=ccrs.epsg(2226), color='k')

        if label_contour:
            print('labeling contours')
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


def addcol2elev(elev, pred_col, year, month, shallow='Shallow', dayoffset = 92):
    #     year_frac = np.float(year_frac)
    df = elev.copy()
    assert month > 0 and month < 13, 'year_frac must be between 1 and 12'
    df.loc[:, 'ts'] = pd.datetime(year, month, 1)
    df.loc[:, 'date_frac'] = date2date_frac(df.loc[:, 'ts'], dayoffset=dayoffset)
    df.loc[:, 'year_frac'] = date2year_frac(df.loc[:, 'ts'], dayoffset=dayoffset)
    df.loc[:, ['Shallow', 'Deep', 'Other']] = 0
    if shallow == 'Shallow':
        df.loc[:, 'Shallow'] = 1
    else:
        df.loc[:, 'Deep'] = 1

    df = df.rename(columns={'RASTERVALU': "rasterelevation"})
    assert df.rasterelevation.mean() > 0
    return df.loc[:, pred_col]


def label_points(ax, gdf, colname, basin_name=["SRP"], kwds_dict={}, buffer=None, limit=20):
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
    val__ = gdf.loc[:, colname]
    texts = [ax.text(x__[i], y__[i], f"{val__[i]:.0f}",
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

    print('seas_fiol:')
    print(seas_fiol.head(5))

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
