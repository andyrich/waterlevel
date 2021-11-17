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

import xarray as xr




class MapHeadDiff:
    def __init__(self, folder):

        self.path = os.path.join('GIS', 'hydro_experiment', folder)
        self.filename = lambda x: 'wl_change_predictions_{:}_{:}.netcdf'.format(x[0], x[1])
        self.nc = {}

    def load_data(self):
        vals = pd.MultiIndex.from_product([['Deep', 'Shallow'], ['Fall', 'Spring']])
        for v, i in vals:
            print(v, i)
            key = self.filename(v,i)
            name = key.replace('wl_change_predictions_','').replace('.netcdf','')
            print(key)
            f = os.path.join(self.path,key)
            rio = xr.open_dataset(f)
            self.nc[name] = rio


    def plotmap(self):

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
                                      n_months=self.train_input.nmonths)

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

                        # add points to map
                        if seas_gdf is None:
                            print(f"{year}-{season}  is not in the columns... ie is not covered by observations\n\n")

                        else:

                            seas_gdf = seas_gdf.to_crs(2226)
                            # modeled points
                            mod_gdf = seas_gdf[(seas_gdf.index.str.contains('mod'))]
                            # observed points
                            seas_gdf = seas_gdf[~(seas_gdf.index.str.contains('mod'))]
                            print('removing duplicated points at observation locations.... kinda crudely')
                            seas_gdf = limit_duplicates(seas_gdf)

                            seas_gdf.loc[:, 'predicted'] = interp_at_obs(xgrid=first.x_stp,
                                                                         ygrid=first.y_stp,
                                                                         z_predicted=first.grid_z2,
                                                                         xout=seas_gdf.geometry.x,
                                                                         yout=seas_gdf.geometry.y)

                            seas_gdf.loc[:, 'Residual'] = seas_gdf.loc[:, 'Observations'] - seas_gdf.loc[:, 'predicted']

                            seas_gdf.loc[:, 'label'] = seas_gdf.apply(
                                lambda x: "{:.0f} ({:+.0f})".format(x['Observations'], x['Residual']), axis=1)

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
    def __init__(self, m_rk,  foldername):

        self.m_rk = m_rk
        self.predictions = None
        self.foldername = foldername


    def predict(self, elev, pred_col, year_predict, season, scale_data=False, scaler=None, depth_type=None,
                smooth=False, dayoffset=92, add_climate=False, n_months=36 ):
        # setup points for running prediction
        df_elev = elev.copy()


        self.year_frac = months(season)
        self.year_predict = year_predict
        self.season = season
        self.plot_title = f"{depth_type} Aquifer" + '\n' + f"{self.season} - {self.year_predict:.0f}"

        # add columns to elev
        out_predict = addcol2elev(df_elev, pred_col, year=year_predict, month=months(season), shallow=depth_type,
                                  dayoffset=dayoffset, add_climate=add_climate, n_months=n_months)
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
               contours=(0,10,20,30,40), locname="PV",
               label_contour=True, crs=ccrs.epsg(2226),
               maptype='ctx.OpenStreetMap.Mapnik'):

        m = mp.make_map(self.plot_title, projection=crs)
        fig = plt.figure(figsize=(10, 10))
        ax = m.plotloc(fig, locname=locname, maptype=maptype)
        ax.contourf(self.x_stp, self.y_stp, self.grid_z2, levels=contours, alpha=0.5,
                                          transform=ccrs.epsg(2226),
                                          cmap='nipy_spectral')

        ax.contour(self.x_stp, self.y_stp, self.grid_z2, levels=contours, alpha=.9,
                                       transform=ccrs.epsg(2226), color='k')


        return ax




