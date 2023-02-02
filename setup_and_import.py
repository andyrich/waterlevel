import warnings

# import process_climate

warnings.simplefilter(action='ignore', category=FutureWarning)

import conda_scripts.make_map as mp

from conda_scripts.utils import load_all_gw_wiski
import conda_scripts.utils.krig_dataset as lgp
import conda_scripts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import seaborn as sns

import geopandas as gpd
# import pykrige

import contextily as ctx


class Krig:

    def __init__(self, add_modeled = False,
                monthlytimestep = 18,
                 scale_data=False,
                 filename_base = 'foldername',
                 load_old_elev = True,
                 deep_categories = True,
                 slope_tiff_elev = r'C:\GIS\raster\DEM\sgdem_slope_402.tif',
                 elev_tiff_elev = r'C:\GIS\raster\DEM\sgmdemuncl_test1.tif',
                 outfolder_data = 'temp',
                 add_geology = True,
                 add_geophys = True,
                 plot_all = False,
                 basin="SRP",
                 dayoffset = 92,
                 deeplayer = 3,
                 xysteps = 12,
                 add_climate = True,
                 add_temp = True,
                 filter_manual = False,
                 nmonths = 36,
                 obs_filename = 'all_gw_for_surf_2022.csv'):

        self.add_modeled = add_modeled
        self.monthlytimestep = monthlytimestep #of the SRPHM/SVIGM etc
        self.scale_data = scale_data
        self.plot_all = plot_all
        self.outfolder_data = outfolder_data

        self.load_old_elev = load_old_elev
        self.deep_categories = deep_categories

        self.add_geology = add_geology
        self.add_geophys = add_geophys

        self.deeplayer = deeplayer
        self.xysteps = xysteps

        self.add_climate = add_climate
        self.climate_cols = None
        self.add_temp = add_temp
        self.nmonths = nmonths
        self.filter_manual = filter_manual
        self.obs_filename = obs_filename

        assert basin in ['SRP', 'SON', 'PET', 'ALL_BASIN'], 'wrong option for basin name, should be SRP, SON, or PET'
        self.basin = basin

        self.dayoffset = dayoffset

        # day = pd.to_datetime('now').strftime('%Y%m%d')
        # base = day + filename_base

        self.map_foldername = 'maps_' + filename_base
        self.hydros_foldname = 'hydros_' + filename_base
        print(self.map_foldername)

        if slope_tiff_elev is None:
            self.use_slope = False
        else:
            self.use_slope = True

        self.slope_tiff_elev = slope_tiff_elev
        self.elev_tiff_elev = elev_tiff_elev

        if self.add_modeled:
            self.modheads_allinfo = None
            self.modheads_stat_info_wisk = None
            self.modheads_stations = None
            self.modheads_all_obs = None


        self.plot_all = plot_all

        self.description = f"\
        add_modeled = {add_modeled}\n\
        monthlytimestep(model) = {monthlytimestep}\n\
        scale_data = {scale_data}\n\
        use_slope = {self.use_slope}\n\
        slope_tiff_elev = {slope_tiff_elev}\n\
        elev_tiff_elev = {elev_tiff_elev}\n\
        dayoffset = {dayoffset}\n\
        deeplayer[actual] = {deeplayer+1}\n\
        deep_categories = {deep_categories}\n\
        add_geology = {add_geology}\n\
        add_geophys = {add_geophys}\n\
        climate = {add_climate}\n\
        add_temp = {add_temp}\n\
        nmonths = {nmonths}\n\
        obs_filename = {obs_filename}\n\
        filter_manual = {filter_manual}\n"

        assert os.path.exists('T:\\'), 'need to connect to SCWA computers'
        assert os.path.exists('S:\\'), 'need to connect to SCWA computers'

        print('done setting up.')

    def load_obs(self):
        '''
        # Use this to map gw levels for gw basins.

        this heavily relies on the following scripts, both of which are located in the conda_scripts folder:
        - plot_wet
        - and gwplot_wiski

        This script uses the above scripts to download all of the data for each station.

        The station data is downloaded direclty from wiski using their web service.

        :return:
        '''

        # TODO - remove filter of observations 50ft bgs
        print('\n\nloading data')
        # # collect all of the station info here:
        allinfo = conda_scripts.utils.gwl_krig_preprocess.collect_station_info()

        allinfo.to_excel(os.path.join(r"C:\GSP\waterlevel\temp",'info.xlsx'))


        #check for missing stations
        # missing = pd.Series(['SRP0129' ,'SRP0140','SRP0148', 'SRP0154' ,'SRP0157' ,'SRP0164', 'SRP0166',
        # 'SRP0171' ,'SRP0178' ,'SRP0187' ,'SRP0226', 'SRP0723'])
        # assert missing.isin(allinfo.loc[:,'Station Name']).all(), missing[~missing.isin(allinfo)]

        allinfo.loc[:, 'Site Number'] = allinfo.loc[:,'Site Number'].fillna('ss').str.upper()

        # TODO - remove basin filter
        allinfo = allinfo[allinfo.loc[:,'Site Number'].isin(pd.Series(self.basin))]

        # assert allinfo.shape[0]>0, 'allinfo shape is zero after filtering for basin'


        # # add location data to wells where it is missing. use lat/lon to give easting/northing and vice versa
        allinfo = conda_scripts.utils.gwl_krig_preprocess.add_stat_loc(allinfo)

        # # check for missing stations
        # assert missing.isin(allinfo.loc[:,'Station Name']).all(), missing[~missing.isin(allinfo)]

        # # load all gw level data.
        # #### if this hasn't been done recently, do this. otherwise can be skipped and just import data from csv instead
        outfolder = 'regression_data'
        maindf = load_all_gw_wiski.load_all_gw(download=False, outfolder=outfolder,
                                               filter_manual = self.filter_manual,
                                               outfile = self.obs_filename)

        # print(maindf.filter(like='Site').head())
        # TODO fix model so that 'ALL_BASIN' filter works
        if self.basin == 'ALL_BASIN':
            print('Not filter by basin because basin==ALL_BASIN')
            pass
        else:
            maindf = maindf[maindf.Site == self.basin]

        assert maindf.shape[0] > 0, 'maindf shape is zero after filtering for basin'

        # if self.plot_all:
        #     temp = maindf.copy()
        #     # temp = temp.set_index('Timestamp')
        #     temp.loc[:, 'year'] = temp.index.year
        #     for name, g in temp.query("year>2010").groupby('year'):
        #         plt.figure()
        #         ax = g.groupby('station_name').min().filter(like='Manual Measurement').plot(label=name)
        #         ax.set_title(name)
        #         plt.savefig(os.path.join('temp plots', f'year{name}'+'.png'))
        #         # plt.close(fig)
        #         plt.close('all')


        maindf.loc[:, 'Manual Measurement'] = pd.to_numeric(maindf.loc[:, 'Manual Measurement'], errors = 'coerce')
        maindf = maindf[maindf['Manual Measurement'].notnull()]
        maindf = maindf[~np.isinf(maindf['Manual Measurement'])]
        maindf = maindf[np.isfinite(maindf['Manual Measurement'])]
        maindf = maindf[maindf['Manual Measurement'].abs() < 2000]


        print(maindf['Manual Measurement'].describe())

        print(f"these are the sites {maindf.Site.unique()}")
        print(f"these are the ts_shortname {maindf.ts_shortname.unique()}")
        print(f"these are the ts_type_name {maindf.ts_type_name.unique()}")
        print(f"these are the ts_type_name {maindf.ts_type_name.fillna('bad').value_counts()}")


        all_obs = conda_scripts.utils.gwl_krig_preprocess.process_timeseries(maindf, dayoffset=self.dayoffset, freq='MS')

        if self.plot_all:
            try:
                os.mkdir('temp plots')
            except:
                pass
            for g, group in all_obs.groupby('Station Name'):
                fig = plt.figure()
                ax = group.plot(x = 'Timestamp',y = 'Manual Measurement')
                ax.set_title(g)
                plt.savefig(os.path.join('temp plots', g+'.png'))
                # plt.close(fig)
                plt.close('all')
        # temp = maindf.groupby(['Site', "station_name", pd.Grouper(level=0, freq='4QS')]).mean().reset_index()

        if self.add_modeled:
            print('\n\nadding modeled data to allinfo')

            modheads_allinfo, modheads_stat_info_wisk, \
            modheads_stations, modheads_all_obs = self.__extract_model_data(deeplayer = self.deeplayer,
                                                                            monthlytimestep = self.monthlytimestep,
                                                                            xysteps = self.xysteps)

            self.modheads_allinfo = modheads_allinfo
            self.modheads_stat_info_wisk = modheads_stat_info_wisk
            self.modheads_stations = modheads_stations
            self.modheads_all_obs = modheads_all_obs

            filemod = os.path.join(outfolder,f'model_heads_{self.basin}.csv')
            print(f'exporting modeled heads to {filemod}')
            modheads_all_obs.to_csv(filemod)

            print(f'shape before:{allinfo.shape}')
            allinfo = allinfo.append(modheads_allinfo)
            print(f'shape after:{allinfo.shape}')

            all_obs = all_obs.append(modheads_all_obs)


        self.allinfo = allinfo
        self.maindf = maindf
        self.all_obs = all_obs


    def process_climate(self):
        if self.add_climate:
            import process_climate as pcr
            climate = pcr.climate(precip=True)
            climate.resample_climate(n_months = self.nmonths)
            print(f'adding precip to observation data. using {self.nmonths}')
            self.all_obs = climate.add_climate_to_obs(self.all_obs, column = 'Timestamp')
            self.climate_cols = climate.climate_cols
        else:
            pass

        if self.add_temp:
            import process_climate as pcr
            climate = pcr.climate(precip=False)
            climate.resample_climate(n_months = self.nmonths)
            print(f'adding temperature to observation data. using {self.nmonths}')
            self.all_obs = climate.add_climate_to_obs(self.all_obs, column = 'Timestamp')
            self.climate_cols.extend( climate.climate_cols)

    def export_processed_ts(self):
        self.all_obs.to_csv(os.path.join('regression_data',f'all_gw_for_surf_processed_{self.basin}.csv'))


    def filter_and_export_shp(self, filt_str=None, allow_missing = True):
        if filt_str is None:
            filt_str = ['Santa', 'Sonoma', 'Peta']

        print('\n\nfiltering stations and exporting shapefile')
        filename = 'wiski_wells_v3.shp'
        wells_shp = os.path.join(self.outfolder_data, filename)
        self.wells_shp = wells_shp
        stat_info_wisk = conda_scripts.utils.gwl_krig_preprocess.filter_station(self.allinfo,
                                                                                outfolder=self.outfolder_data,
                                                                                filename=filename,
                                                                                filt_str=filt_str)

        self.stat_info_wisk = stat_info_wisk
        print('stat_info_wisk head:')
        print(stat_info_wisk.head(1))

        c = ~(self.all_obs.loc[:,'Station Name'].isin(stat_info_wisk.loc[:,'Station Name']))

        if c.sum()>0:
            missing = self.all_obs.loc[c,:]
            missing = missing.drop_duplicates(subset=['Station Name'])


            if allow_missing:
                import warnings
                warnings.warn('There are missing stations\n')
                miss = ', '.join(missing.loc[:,'Station Name'].unique())+' are missing'
                warnings.warn(miss)
                warnings.warn(f'size of missing stations in stat_info_wisk:\t{missing.shape}')
            else:
                raise AssertionError("the following stations are missing:\n{:}".format( missing.loc[:,'Station Name'].unique()))



    def __extract_model_data(self, deeplayer, monthlytimestep, xysteps):
        if self.basin == 'SRP':
            modheads_allinfo, modheads_stat_info_wisk, \
                modheads_stations, modheads_all_obs = load_srp_mod(dayoffset=self.dayoffset,
                                                                      deeplayer = deeplayer,
                                                                      monthlytimestep = monthlytimestep,
                                                                      xysteps = xysteps)
        elif self.basin.upper() == 'PET':
            modheads_allinfo, modheads_stat_info_wisk,\
                modheads_stations, modheads_all_obs = load_pet_mod(dayoffset=self.dayoffset,
                                                                      deeplayer=deeplayer,
                                                                      monthlytimestep=monthlytimestep,
                                                                      xysteps=xysteps)
        elif self.basin.upper() == 'SON':
            modheads_allinfo, modheads_stat_info_wisk,\
                modheads_stations, modheads_all_obs = load_son_mod(dayoffset=self.dayoffset,
                                                                      deeplayer=deeplayer,
                                                                      monthlytimestep=monthlytimestep,
                                                                      xysteps=xysteps)
        elif self.basin.upper() == 'ALL_BASIN':

            modheads_allinfo_0, modheads_stat_info_wisk_0, \
                modheads_stations_0, modheads_all_obs_0 = load_srp_mod(dayoffset=self.dayoffset,
                                                                      deeplayer = deeplayer,
                                                                      monthlytimestep = monthlytimestep,
                                                                      xysteps = xysteps)
            modheads_allinfo_1, modheads_stat_info_wisk_1,\
                modheads_stations_1, modheads_all_obs_1 = load_son_mod(dayoffset=self.dayoffset,
                                                                      deeplayer=deeplayer,
                                                                      monthlytimestep=monthlytimestep,
                                                                      xysteps=xysteps)

            modheads_allinfo, modheads_stat_info_wisk,\
                modheads_stations, modheads_all_obs = load_pet_mod(dayoffset=self.dayoffset,
                                                                      deeplayer=deeplayer,
                                                                      monthlytimestep=monthlytimestep,
                                                                      xysteps=xysteps)

            modheads_allinfo = modheads_allinfo.append(modheads_allinfo_0).append(modheads_allinfo_1)
            modheads_stat_info_wisk = modheads_allinfo.append(modheads_stat_info_wisk_0).append(modheads_stat_info_wisk_1)
            modheads_stations = modheads_allinfo.append(modheads_stations_0).append(modheads_stations_1)
            modheads_all_obs = modheads_allinfo.append(modheads_all_obs_0).append(modheads_all_obs_1)

        else:
            modheads_allinfo, modheads_stat_info_wisk, modheads_stations, modheads_all_obs = None, None, None, None

        if self.plot_all:
            fig = plt.figure(figsize=(5, 5), dpi=220)
            mm = conda_scripts.make_map.make_map('test station')
            ax = mm.plotloc(fig, locname='SRP_MOD')
            modheads_stations.plot(ax=ax, marker='o')

        return modheads_allinfo, modheads_stat_info_wisk, modheads_stations, modheads_all_obs



    def add_krig_values(self):
        # keep observed elevations of observations (do not use DEM values)

        self.stat_info_wisk = conda_scripts.utils.gwl_krig_preprocess.add_krig_values(self.stat_info_wisk,
                                                                                 self.wells_shp,
                                                                                 elev_tiff_elev=self.elev_tiff_elev,
                                                                                 slope_tiff_elev=self.slope_tiff_elev
                                                                                 )

        if self.plot_all:
            ax = self.stat_info_wisk.plot('rasterelevation', legend=True, figsize=(10, 10))
            ctx.add_basemap(ax, crs=2226)

        if 'slope' in self.stat_info_wisk.columns:
            pass
        else:
            print('\n\nadding slope = nan to stat_info_wiski')
            self.stat_info_wisk.loc[:, 'slope'] = np.nan

        # if self.add_modeled:
        #     self.stat_info_wisk = self.stat_info_wisk.append(self.modheads_stat_info_wisk)

    # def add_model_head(self):
    #
    #     if self.add_modeled:
    #         print('adding modeled data to stat_info_wisk')
    #         print(f'shape before:{stat_info_wisk.shape}')
    #         stat_info_wisk = stat_info_wisk.append(modheads_stat_info_wisk)
    #         print(f'shape after:{stat_info_wisk.shape}')
    #         stat_info_wisk = stat_info_wisk.reset_index(drop = True)


    def join_well_data(self,depth_filter=None):
        print('\n\njoining well data')
        seas_info = conda_scripts.utils.gwl_krig_preprocess.join_well_data(
            self.allinfo,
            self.all_obs,
            self.stat_info_wisk,
            depth_filter=depth_filter,
            site_filter=None)


        assert seas_info.shape[0]>0,'Seas_info shape ==0 after join_well_data'

        c = (seas_info.loc[:,'Manual Measurement'] - seas_info.loc[:,'rasterelevation']) > 50
        print(f"removing {c.sum()} measurements that are 50 feet greater than surface elevation")
        seas_info = seas_info.loc[~c,:]

        self.seas_info = seas_info

        print('heres the seas_info')
        print(seas_info.tail(2))
        print('\n\n')

    def do_geophys(self):
        print('\nloading geophysics data\n')
        geophys = lgp.load_data()
        geop_pred = lgp.do_krig_all()

        fff = pd.DataFrame(geophys.drop(columns=['station_id', 'latitude', 'longitude', 'geometry']))
        sns.pairplot(fff)

        self.geophys = geophys
        self.geop_pred = geop_pred

        if self.plot_all:
            for col in geop_pred.loc[:, 'Simple_Bou':].select_dtypes(float).columns:
                #     col = 'Simple_Bou'

                fig = plt.figure(figsize=(5, 5), dpi=100)
                mmmm = mp.make_map(col)

                ax = mmmm.plotloc(fig, locname='all_mod', maptype='ctx.Esri.WorldStreetMap')
                #     ax = mmmm.plotloc(fig, locname = 'all_mod',maptype = 'None')

                self.stat_info_wisk.plot(legend=False, ax=ax, markersize=3, marker='o', edgecolor='b', facecolor="None",
                                    linewidth=.2)

                geophys.plot(col, ax=ax,
                             cmap='jet', legend=True, edgecolor='k', linewidth=.2, markersize=1)


    def _do_geology(self):
        if self.add_geology:
            import get_geology
            geol = get_geology.geology()
            geol.get_simple_geol()
            self.geol = geol


    def load_elev_pred(self):
        print('\n\ngetting elevation data')
        outfolder = 'temp'
        file_points_name = os.path.join(outfolder, 'pred_points.shp')
        self.file_points_name = file_points_name


        if self.load_old_elev:
            print('reading elevation file.')
            print(f"filename is {self.file_points_name.replace('.shp', '.json')}")
            elev = gpd.read_file(self.file_points_name.replace('.shp', '.json'), )
        else:
            elev = conda_scripts.utils.regression_krig.make_grid(self.file_points_name,
                                                                 elev_tiff_elev=self.elev_tiff_elev,
                                                                 slope_tiff_elev=self.slope_tiff_elev,
                                                                 shpfor_fishnet=None,
                                                                 )

            elev = lgp.do_krig_all(elev, elev.loc[:, 'Easting'], elev.loc[:, 'Northing'])
            if self.add_geology:
                elev = self.geol.add_geol_to_gdf(elev)
            elev.to_file(file_points_name.replace('.shp', '.json'), driver="GeoJSON")

        self.elev = elev

        if self.plot_all:
            for col in self.geop_pred.loc[:, 'Simple_Bou':].select_dtypes(float).columns:

                fig = plt.figure(figsize=(5, 5), dpi=100)
                mmmm = mp.make_map(col)

                ax = mmmm.plotloc(fig, locname='SRP_MOD', maptype='ctx.Esri.WorldStreetMap')
                #     ax = mmmm.plotloc(fig, locname = 'SRP_MOD',maptype = None)

                self.stat_info_wisk.plot(legend=False, ax=ax, markersize=3, marker='o', edgecolor='b', facecolor="None",
                                    linewidth=.2)

                elev.plot(col, ax=ax, legend=True, markersize=1, cmap='jet')
                self.geophys.plot(col, ax=ax,
                             cmap='jet', legend=True, edgecolor='k', linewidth=.2, markersize=3)

    def add_geophys_geol(self, pre_export = False):
        print('\n\nadding krig data to seas_info')
        seas_info = self.seas_info

        if pre_export:
            file_ = "seas_info_erase.csv"
            print(f'exporting {file_}')
            seas_info.to_csv(os.path.join('temp', file_))


        if self.add_geophys:
            print('adding geophysics')
            __statlocs = seas_info.loc[~seas_info.index.duplicated(),['Easting','Northing']]

            __statlocs = lgp.do_krig_all(__statlocs, __statlocs.loc[:,'Easting'], __statlocs.loc[:,'Northing'])

            __statlocs = __statlocs.drop(columns = ['Easting', 'Northing'])
            sh = seas_info.shape
            seas_info = pd.merge(seas_info, __statlocs, left_index=True, right_index=True)
            print(f"seasinfo before join:{sh}\n and after {seas_info.shape}")

            seas_info = gpd.GeoDataFrame(seas_info,
                                        geometry = gpd.points_from_xy(seas_info.loc[:,'Easting'],
                                        seas_info.loc[:,'Northing']),
                                        crs = 2226)
            # seas_info = lgp.do_krig_all(seas_info, seas_info.loc[:,'Easting'],seas_info.loc[:,'Northing'] )
            # seas_info = gpd.GeoDataFrame(seas_info,
            #                      geometry =
            #                         gpd.points_from_xy(seas_info.loc[:,'Easting'],
            #                                            seas_info.loc[:,'Northing']),
            #                         crs = 2226)
            check_bad(seas_info,'Simple_Bou')
            # if seas_info.Simple_Bou.isnull().sum() > 0:
            #     print(seas_info[seas_info.Simple_Bou.notnull()])
            #     ax = seas_info[seas_info.Simple_Bou.notnull()].plot()
            #     ctx.add_basemap(ax, crs = 2226)
            #     raise ValueError('bad geophys prediction')

        if pre_export:
            file_ = "seas_info_erasev2.csv"
            print(f'exporting {file_}')
            seas_info.head(10000).to_csv(os.path.join('temp', file_))

        print('adding krig data to seas_info')
        if self.add_geology:
            self._do_geology()
            print('adding geology')
            seas_info = self.geol.add_geol_to_gdf(seas_info)

            seas_info = pd.DataFrame(seas_info.drop(columns = 'geometry'))

            check_bad(seas_info, 'Geol_Krig')

        check_bad(seas_info, 'slope')

        self.seas_info = seas_info


    def categorize_depths(self):
        self.seas_info.loc[:, 'Well_Depth_Category'] = categorize_depths_inputs(self.seas_info, 'Other')

    def export_seas_info(self):
        self.seas_info.to_csv(os.path.join('regression_data',f"{self.basin}_seas_info.csv"))

def check_bad(gdf, col):
    df = gdf[gdf.loc[:,col].isnull()]
    if df.shape[0]>0:
        print(df.shape)
        print(df.drop(columns = 'geometry'))
        print(df.filter(like = 'ation'))

        ax = df.plot()
        ctx.add_basemap(ax, crs=2226)
        raise ValueError(f'bad {col} prediction')

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


def load_srp_mod(dayoffset, deeplayer, monthlytimestep, xysteps):
    from conda_scripts.utils import extract_model_heads

    modgeoms, dts, workspace, mg = extract_model_heads.get_model_info()
    filename = extract_model_heads.get_file(workspace)
    head, trefall = extract_model_heads.get_head(filename, mg, layers=[0, deeplayer], step=monthlytimestep, basin = 'srp')

    headgdf = extract_model_heads.head_array_to_gdf(head, trefall, modgeoms, xysteps=xysteps)
    headgdf.loc[:, 'geometry'] = headgdf.geometry.centroid

    modheads_all_obs = extract_model_heads.format_heads_for_krig(headgdf, dayoffset = dayoffset)

    modheads_stations = extract_model_heads.get_mod_stations(headgdf)
    modheads_stations = modheads_stations.reset_index(drop=True)

    modheads_allinfo = modheads_stations.loc[:, ['Station Name', 'Easting', 'Northing', 'Latitude',
                                                 "Longitude", 'Well_Depth_Category']]
    modheads_allinfo = modheads_allinfo.set_index(modheads_allinfo.loc[:, 'Station Name'])
    modheads_allinfo.loc[:, ['Site']] = 'Santa Rosa Plain'

    modheads_stat_info_wisk = modheads_stations.loc[:, ['Station Name', 'Easting', 'Northing', 'geometry']]
    modheads_stat_info_wisk.loc[:, ['Reference Elevation']] = np.nan
    modheads_stat_info_wisk.loc[:, ['Site']] = 'Santa Rosa Plain'

    return modheads_allinfo, modheads_stat_info_wisk, modheads_stations, modheads_all_obs

def load_son_mod(dayoffset, deeplayer,  monthlytimestep, xysteps):
    from conda_scripts.utils import extract_model_heads

    modgeoms, dts, workspace, mg = extract_model_heads.get_model_info(basin = 'son')
    filename = extract_model_heads.get_file(workspace, "sv_model_grid_6layers.hds")
    head, trefall = extract_model_heads.get_head(filename, mg, layers=[0, deeplayer], step=monthlytimestep,
                                                 basin = 'son', formatted=False)

    headgdf = extract_model_heads.head_array_to_gdf(head, trefall, modgeoms, xysteps=xysteps)
    headgdf.loc[:, 'geometry'] = headgdf.geometry.centroid

    modheads_all_obs = extract_model_heads.format_heads_for_krig(headgdf, dayoffset = dayoffset, basin='SON')

    modheads_stations = extract_model_heads.get_mod_stations(headgdf)
    modheads_stations = modheads_stations.reset_index(drop=True)

    modheads_allinfo = modheads_stations.loc[:, ['Station Name', 'Easting', 'Northing', 'Latitude',
                                                 "Longitude", 'Well_Depth_Category']]
    modheads_allinfo = modheads_allinfo.set_index(modheads_allinfo.loc[:, 'Station Name'])
    modheads_allinfo.loc[:, ['Site']] = 'Sonoma Valley'

    modheads_stat_info_wisk = modheads_stations.loc[:, ['Station Name', 'Easting', 'Northing', 'geometry']]
    modheads_stat_info_wisk.loc[:, ['Reference Elevation']] = np.nan
    modheads_stat_info_wisk.loc[:, ['Site']] = 'Sonoma Valley'

    return modheads_allinfo, modheads_stat_info_wisk, modheads_stations, modheads_all_obs

def load_pet_mod(dayoffset, deeplayer,  monthlytimestep, xysteps):
    # TODO remove model head where values don't fall on first of month DONE
    from conda_scripts.utils import extract_model_heads
    basin = 'PET'
    modgeoms, dts, workspace, mg = extract_model_heads.get_model_info(basin = basin)
    # filename = extract_model_heads.get_file(workspace, "Head_save.fhd")
    filename = os.path.join(workspace, "Output", "Head_save.fhd")
    head, trefall = extract_model_heads.get_head(filename, mg, layers=[0, deeplayer], step=monthlytimestep,
                                                 basin = 'pet', formatted=True)

    headgdf = extract_model_heads.head_array_to_gdf(head, trefall, modgeoms, xysteps=xysteps)
    headgdf.loc[:, 'geometry'] = headgdf.geometry.centroid

    modheads_all_obs = extract_model_heads.format_heads_for_krig(headgdf, dayoffset = dayoffset, basin=basin)

    modheads_all_obs.loc[:,'Manual Measurement'] = modheads_all_obs.loc[:,'Manual Measurement']*3.28084

    modheads_stations = extract_model_heads.get_mod_stations(headgdf)
    modheads_stations = modheads_stations.reset_index(drop=True)

    modheads_allinfo = modheads_stations.loc[:, ['Station Name', 'Easting', 'Northing', 'Latitude',
                                                 "Longitude", 'Well_Depth_Category']]
    modheads_allinfo = modheads_allinfo.set_index(modheads_allinfo.loc[:, 'Station Name'])
    modheads_allinfo.loc[:, ['Site']] = 'Petaluma Valley'

    modheads_stat_info_wisk = modheads_stations.loc[:, ['Station Name', 'Easting', 'Northing', 'geometry']]
    modheads_stat_info_wisk.loc[:, ['Reference Elevation']] = np.nan
    modheads_stat_info_wisk.loc[:, ['Site']] = 'Petaluma Valley'

    return modheads_allinfo, modheads_stat_info_wisk, modheads_stations, modheads_all_obs