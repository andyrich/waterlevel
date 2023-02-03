from project2d import label_points
import os, sys

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import conda_scripts
import conda_scripts.plot_help as ph
import cartopy.crs as ccrs
import contextily as ctx
import xarray as xr
plt.rcParams['figure.figsize'] = (6, 8)
import post_process as pp


def plot_year(basin, year, depth, season, datafolder, out_folder):

    '''
    create pretty maps of waterlevel estimates
    :param basin: srp/son/pet
    :param year: int
    :param depth: Shallow/Deep
    :param season: Spring/Fall
    :param datafolder: folder where the output from from waterlevel GWLE is put
    :param out_folder: where the final figures will go
    :return: None
    '''

    path = os.path.join(datafolder, f"{basin.upper()}_{year}_{season}_{depth}")
    print(path)
    obs = gpd.read_file(path + '.shp').rename(columns={'Observatio': "Observations"})

    basshp = conda_scripts.rich_gis.get_active_subbasins()
    basshp = basshp.loc[[basin.upper()], :]
    cnt20 = gpd.read_file(path + '_cnt20ft.shp')
    cnt100 = gpd.read_file(path + '_cnt100ft.shp')
    cnt20 = gpd.clip(cnt20, basshp)
    cnt100 = gpd.clip(cnt100, basshp)

    gw = xr.open_rasterio(path + '.tif')

    levels = np.arange(-20, 300, 20)

    fig = plt.figure(figsize=(6, 8), dpi=250)


    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])

    ax = fig.add_subplot(gs[0:2, :], projection=ccrs.epsg(2226))
    conda_scripts.rich_gis.set_extent(ax, locname=basin)

    title = f"Waterlevel for {depth.capitalize()} Aquifer, {year} {season.capitalize()}"
    basshp.plot(ax=ax, facecolor="None", edgecolor='b')
    # mm = conda_scripts.make_map.make_map(title)

    # ax = axes[0]

    # ax = mm.plotloc(fig,ax, locname = basin, maptype = 'ctx.OpenStreetMap.Mapnik')
    mask = pp.get_mask(basin)
    # gw.where(mask)
    # gw.where(np.flipud(mask.values)).Band.plot.contourf(
    gw.where(np.flipud(mask.values)).isel(band=0).plot.contourf(ax=ax,
                                                                alpha=.7, extend='both', levels=levels, cmap='RdYlBu',
                                                                cbar_kwargs={'label': 'Waterlevel Elevation (feet)',
                                                                             'shrink': .5})

    CS = gw.where(np.flipud(mask.values)).isel(band=0).plot.contour(levels=[0, 100, 200, 300],
                                                                    linewidths=1,
                                                                    colors='k')

    obs.loc[:, 'label'] = obs.apply(
        lambda x: "{:.0f}".format(x['Observations']), axis=1)

    obs.plot(ax=ax, edgecolor='w', markersize=15, facecolor = 'purple')
    col2plot = 'label'
    already_str = True
    label_points(ax, obs, col2plot,
                 basin_name=[basin.upper()],
                 buffer=2000, limit=5, already_str=already_str)

    def fmt(x):
        s = f"{int(x):0d}ft"
        return s

    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=8, colors='k')

    cnt20.plot(ax=ax, lw=.5, ls='--', color='k')

    ctx.add_basemap(ax, source="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", crs=2226)
    ax.set_title(title, fontsize=12)

    hand = ph.add_marker_to_legend(ax, handles=None, color='k',
                                   marker=None, linestyle='-',
                                   label='Groundwater Contour (100 ft)')

    hand = ph.add_marker_to_legend(ax, handles=hand, color='k',
                                   marker=None, linestyle='--',
                                   label='Groundwater Contour (20 ft)')

    hand = ph.add_marker_to_legend(ax, color='purple', handles=hand, markersize=4,
                                   marker='o', linestyle='',
                                   label='Observed\nGroundwater\nElevation (ft)')

    hand = ph.add_marker_to_legend(ax, handles=hand, color='b',
                                   marker=None, linestyle='-',
                                   label='Basin Boundary')

    ax.legend(handles=hand, loc='upper left', ncol=2, bbox_to_anchor=(0, 0), fontsize=6)
    plt.savefig(os.path.join(out_folder, f'WL map {depth}_{season}.png'), dpi=250)
    plt.close()

def run(year = None):
    import datetime

    if year is None:
        year = datetime.datetime.now().year -1

    for basin in ['son', 'srp', 'pet']:
        mod = pp.wl_ch(basin)
        [[plot_year(basin, year, shallow, season, mod.folder, mod.share_folder)
            for shallow in ('Shallow', 'Deep')]
            for season in ('Fall', 'Spring')]

if __name__ == 'main':
    run()


    # shallow = 'Shallow'
    # season = "Spring"
    # [[plot_year(basin, year, shallow, season) for shallow in ('Shallow', 'Deep')] for season in ('Fall', 'Spring')]
    #
    # basin = mod.basin
    # shallow = 'Shallow'
    # season = "Spring"
    # [[plot_year(basin, year, shallow, season) for shallow in ('Shallow', 'Deep')] for season in ('Fall', 'Spring')]