import xarray as xr
import os
import conda_scripts
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt


def get_ss(ml, layer, standard_units, spec_storage=True):
    '''
    get specific storage from ml
    :param ml:
    :param layer:
    :param spec_storage: True means input values in ft, false they're in M.
    :return: geodataframe of ss/sy
    '''
    ml.update_modelgrid()

    if spec_storage:
        ss = ml.upw.ss.array[layer]
        if not standard_units:
            ss = ss * 3.28084
            print('converting to standard units...multiplying ss by 3.28084')
        print(f'using specific storage with layer {layer}')
    else:
        ss = ml.upw.sy.array[0]
        print(f'using specific yield')

    # filter inactive
    c = ss <= 0
    ss[c] = np.nan

    print(np.nanmin(ss))
    assert np.nanmin(ss) > 0

    easting = ml.modelgrid.get_xcellcenters_for_layer(0)
    northing = ml.modelgrid.get_ycellcenters_for_layer(0)

    xxr, yyr, zr = np.ravel(easting), np.ravel(northing), np.ravel(ss)

    #epsg = ml.dis.sr.proj4_str
    epsg = ml.modelgrid.epsg

    ss = gpd.GeoDataFrame(zr, geometry=gpd.points_from_xy(xxr, yyr),
                          columns=['storage'], crs=epsg)
    print(ss.crs)
    ss = ss.to_crs(2226)

    # x = easting[0 ,:]
    # y = northing[: ,0]
    #
    # print(f"model coordinate system is {ml.epsg}")
    #
    # if ml.epsg != 2226:
    #     xx,yy = np.meshgrid()
    #
    # ss = xr.DataArray(ss,
    #                   coords={'easting':x,
    #                           'northing':y},
    #                   dims = ['northing' ,'easting'])
    #
    # ss = ss.where(ss > 0)

    return ss


def interp_ss(ss, wl):
    '''
    inerpolate specific storage values to the x/y locations of the wl grid vertices
    '''
    # # from scipy import interpolate
    # xx, yy = np.meshgrid(ss.easting, ss.northing)
    #
    # z = np.copy(ss.values)
    #
    # xxr, yyr, zr = np.ravel(xx), np.ravel(yy), np.ravel(z)
    xxr, yyr, zr = ss.geometry.x, ss.geometry.y, ss.storage.values

    c = np.isnan(zr)

    from sklearn.ensemble import ExtraTreesRegressor as regr
    f = regr()
    f.fit(np.hstack((np.expand_dims(xxr[~c], 1), (np.expand_dims(yyr[~c], 1)))), zr[~c])

    easting, northing = np.meshgrid(wl.easting, wl.northing)
    easting = easting.reshape((-1, 1))
    northing = northing.reshape((-1, 1))

    pred = f.predict(np.hstack((easting, northing)))

    pred = pred.reshape((wl.dims['northing'], wl.dims['easting']))

    ss_new = xr.DataArray(data=
                          pred,
                          dims=["northing", 'easting'],
                          coords=dict(
                              easting=(["easting"], wl.easting.data),
                              northing=(["northing"], wl.northing.data
                                        )))

    return ss_new


def get_mask(basin, filename=r'C:\GSP\waterlevel\regression_data\allbasin_mask.nc'):
    '''
    get mask for models
    :param basin: name of basin
    :param filename:
    :return:
    '''

    bas_dict = {'SRP': 1, 'SON': 2, 'PET': 0}
    if not (basin.upper() in bas_dict.keys()):
        raise ValueError(f"basin not in basin_dict"
                         'use {bas_dict.keys()}')
    else:
        print(f"getting mask for {basin}")

    basin_value = bas_dict[basin.upper()]
    m = xr.open_dataarray(filename)
    m = m.where(m == basin_value).notnull()
    m = m.rename({'lon': 'easting',
                  'lat': 'northing'})
    return m


def get_season_depth(deep, spring):
    '''
    helper function
    :param deep:
    :param spring:
    :return:
    '''
    if deep:
        depth = 'Deep'
    else:
        depth = "Shallow"

    if spring:
        season = "Spring"
    else:
        season = "Fall"

    return depth, season


def get_waterlevel_change_xr(folder, deep=True, spring=True):
    '''
    open waterlvel prediciton
    :param folder:
    :param deep:
    :param spring:
    :return:
    '''
    depth, season = get_season_depth(deep, spring)

    filename = f"wl_change_predictions_{depth}_{season}.netcdf"
    print(filename)
    file = os.path.join(folder, filename)

    wl = xr.open_dataset(file)

    return wl


def get_stor_estimate(ss, mask, basin, thickness=300, meanwl_ch=1.5):
    bas = conda_scripts.rich_gis.get_active_subbasins()
    bas_area = bas.loc[basin.upper()].geometry.area

    ss_average = ss.where(mask == 1).mean()

    storch = meanwl_ch * bas_area * thickness * ss_average / 43560
    print('------------showing for sanity check\n')
    print(
        f'bas area: {bas_area:,.0f} ft^2 ({bas_area / 43560:,.0f} acres) \ndepth: {thickness}ft\nss_average = {ss_average.data:,.4g} [1/ft]\
    \nmeanwl_ch = {meanwl_ch:+,.1f}ft\nTotal Storage change: {storch.data:,.0f} af')
    print('\n---------')


def get_average_wl_change(wl, mask, deep, spring, out_folder):
    '''

    :param wl:
    :param mask:
    :param deep:
    :param spring:
    :param out_folder:
    :return:
    '''
    wl_av = wl.where(mask == 1).groupby('year').mean(...).to_dataframe()

    depth, season = get_season_depth(deep, spring)
    ax = wl_av.plot()
    ax.set_ylabel('feet')
    ax.set_title(f'Average head change for the basin, year over year\n{depth} Aquifer - {season}')

    plt.savefig(os.path.join(out_folder, f"wl change {depth} Aquifer - {season} .png"))

    return wl_av


def get_storage_ts_in_af(ssnew, wl, thickness, deep, spring, out_folder, mask):
    dh = wl.easting.diff(dim='easting').data[0]
    dy = wl.northing.diff(dim='northing').data[0]
    cell_area = dh * dy

    # units in ft^3
    stor_xr = (ssnew * wl).where(mask == 1) * cell_area * thickness

    stor = stor_xr.groupby('year').sum(...).to_dataframe().multiply(1 / 43560)

    stor.columns = ['Storage Change (af)']

    ax = stor.plot.bar(figsize=(8, 8))
    ax.grid(True)
    depth, season = get_season_depth(deep, spring)
    ax.set_title(f'Total Storage Change, year over year\n{depth} Aquifer - {season}')
    plt.savefig(os.path.join(out_folder, f"storage change {depth} Aquifer - {season}.png"))

    stor.to_excel(os.path.join(out_folder, f"storage change {depth} Aquifer - {season}.xlsx"))


    return stor, stor_xr
