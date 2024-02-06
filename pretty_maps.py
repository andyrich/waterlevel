import textwrap

import conda_scripts.plot_help

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
import rioxarray as rx
from matplotlib.colors import TwoSlopeNorm
import fig_setup
fig_setup.set_pub()

def plot_year(basin, year, depth, season, figure_num, datafolder, out_folder,title = None):

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

    print(f"opening the following with rioxarray\n{path+'.tif'}")
    # gw = xr.open_rasterio(path + '.tif')
    gw = rx.open_rasterio(path+'.tif')

    if title is None:
        title = f"Groundwater Waterlevels for the {depth.capitalize()} Aquifer, {year}, {season.capitalize()}"

    fig, ax, axlegend, axtitle, axscale, cbar = conda_scripts.map_figure.make_map_figure(
        title,
        figure_number_text=figure_num,  add_colorbar_axis='lower left')

    conda_scripts.rich_gis.set_extent(ax, locname=basin, keep_init_axis_ratio=True)

    if basin.upper() == 'SON':
        levels = np.arange(-160, 300, 20)
        # fig = plt.figure(figsize=(6, 8), dpi=250)
    else:
        levels = np.arange(-20, 300, 20)
        # fig = plt.figure(figsize=(8, 6), dpi=250)


    basshp.plot(ax=ax, facecolor="None", edgecolor='b', zorder = 500)

    mask = pp.get_mask(basin)

    linewidths = .5
    linewidths_minr = .5
    cntr_minr_col = 'darkgrey'
    cntr_majr_col = 'black'
    cntr_minr_style = '-'
    cntr_majr_style = '-'
    wshed_color = 'grey'

    cbar_kwargs = {'orientation': 'vertical', 'shrink': 1.0,  'label': 'Waterlevel Elevation (feet)',
                   'cax': cbar}

    #without doing via low-level mpl, not all contours were being shown.
    x, y = np.meshgrid(gw.where(np.flipud(mask.values)).isel(band=0).x, gw.where(np.flipud(mask.values)).isel(band=0).y)
    z = gw.where(np.flipud(mask.values)).isel(band=0).values
    CS3 = ax.contourf(x, y, z, levels, cmap='RdYlBu', extend='both', alpha = 0.8,
                      norm=TwoSlopeNorm(vmin=levels.min(), vcenter=0,
                                                    vmax=levels.max()) )

    fig.colorbar(CS3, **cbar_kwargs )

    # gw.where(np.flipud(mask.values)).isel(band=0).plot.contourf(ax=ax,
    #                                                             alpha=.8, extend='both', levels=levels, cmap='RdYlBu',
    #                                                             cbar_kwargs=cbar_kwargs,
    #                                                             norm=TwoSlopeNorm(vmin=levels.min(), vcenter=0,
    #                                                                               vmax=levels.max() ))


    # cnt20.loc[~cnt20.loc[:,'elev'].astype(int).isin(np.arange(-1000,1000,100))].plot(ax=ax, lw=linewidths_minr, ls=cntr_minr_style, color=cntr_minr_col)

    # contour 20ft levels
    cntr_20ft_levels = [x for x in levels if x not in np.arange(-1000,1000,100)]
    gw.where(np.flipud(mask.values)).isel(band=0).plot.contour(levels= cntr_20ft_levels,
                                                                    ax = ax,
                                                                    linewidths=linewidths_minr,
                                                                    linestyle = cntr_minr_style,
                                                                    colors=cntr_minr_col)

    #contour 100 foot levels
    CS = gw.where(np.flipud(mask.values)).isel(band=0).plot.contour(levels=[-200, -100,0, 100, 200],
                                                                    ax = ax,
                                                                    linewidths=linewidths,
                                                                    linestyle = cntr_majr_style,
                                                                    colors=cntr_majr_col)

    obs.loc[:, 'label'] = obs.apply(
        lambda x: "{:.0f}".format(x['Observations']), axis=1)

    obs.plot(ax=ax, edgecolor='w', markersize=15, facecolor = 'purple')
    col2plot = 'label'
    already_str = True

    conda_scripts.plot_help.label_points(
        ax, obs, col2plot,
        basin_name=[basin.upper()],
        already_str=already_str,
        kwds_dict={'x': obs.geometry.x.values, 'y': obs.geometry.y.values,
               # 'avoid_self': False,
                   'expand_text': (4, 4), 'expand_points': (1.6, 1.6),
               "force_points": (1., 1.)
               },
    )

    def fmt(x):
        s = f"{int(x):1d}ft"
        return s

    ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=8, colors='k')

    # ctx.add_basemap(ax, source="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", crs=2226)
    conda_scripts.arich_functions.add_basemaps(ax,maptype='ctx.Esri.NatGeoWorldMap',)
    conda_scripts.arich_functions.add_basemap_transpo(ax=ax)
    conda_scripts.arich_functions.add_basemap_placenames(ax = ax)
    conda_scripts.rich_gis.plot_eastside_fault(ax = ax)
    # conda_scripts.arich_functions.add_basemaps(ax, 'transpo', zorder = 10, wms_kwargs = {"zoom":8,'transparent':True})
    # conda_scripts.arich_functions.add_basemaps(ax, 'street_map', zorder = 10)

    conda_scripts.rich_gis.plot_water_shed(ax = ax, cur_basin=basin, edgecolor=wshed_color)
    ax.set_title('', fontsize=12)

    hand = ph.add_marker_to_legend(axlegend, handles=None, color=cntr_majr_col,
                                   marker=None, linestyle=cntr_majr_style,
                                   label='Groundwater Contour (100 ft)')

    hand = ph.add_marker_to_legend(axlegend, handles=hand, color=cntr_minr_col,
                                   marker=None, linestyle=cntr_minr_style,
                                   label='Groundwater Contour (20 ft)')

    hand = ph.add_marker_to_legend(axlegend, color='purple', handles=hand, markersize=2,
                                   marker='o', linestyle='',
                                   label='Observed\nGroundwater\nElevation (ft)')

    hand = ph.add_marker_to_legend(axlegend, handles=hand, color='b',
                                   marker=None, linestyle='-',
                                   label='Basin Boundary')

    hand = ph.add_marker_to_legend(axlegend, handles=hand, color=wshed_color,
                                   marker=None, linestyle='-',
                                   label='Basin Watershed')

    axlegend.legend(handles=hand, loc='lower left',  bbox_to_anchor=(.0, .0), fontsize=10,mode = 'expand', edgecolor = None)
    plt.savefig(os.path.join(out_folder, f'{figure_num} WL map {depth}_{season}.png'), dpi=250)
    plt.close()

def plot_year_diff(basin, year, depth, season, figure_num, datafolder, out_folder,title = None):

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

    # path = os.path.join(datafolder, f"{basin.upper()}_{year}_{season}_{depth}")

    # obs = gpd.read_file(path + '.shp').rename(columns={'Observatio': "Observations"})
    #
    basshp = conda_scripts.rich_gis.get_active_subbasins()
    basshp = basshp.loc[[basin.upper()], :]

    path = os.path.join(datafolder, f"wl_change_predictions_{depth}_{season}.netcdf")
    wlmap = xr.open_dataset(path)

    print(f"opening the following with rioxarray\n{path+'.tif'}")


    if title is None:
        title = f"Waterlevel Changes from {year-1} to {year}, {depth.capitalize()} Aquifer, {season.capitalize()}"

    fig, ax, axlegend, axtitle, axscale, cbar = conda_scripts.map_figure.make_map_figure(
        title,
        figure_number_text=figure_num,  add_colorbar_axis='lower left')

    conda_scripts.rich_gis.set_extent(ax, locname=basin, keep_init_axis_ratio=True)

    if basin.upper() == 'SON':
        levels = [-20, -15, -10, -5, -1, 1, 5, 10, 15, 20]
        # fig = plt.figure(figsize=(6, 8), dpi=250)
    else:
        levels = [-20, -15, -10, -5, -1, 1, 5, 10, 15, 20]
        # fig = plt.figure(figsize=(8, 6), dpi=250)


    basshp.plot(ax=ax, facecolor="None", edgecolor='b', zorder = 500)

    mask = pp.get_mask(basin)

    linewidths = .5
    linewidths_minr = .5
    cntr_minr_col = 'darkgrey'
    cntr_majr_col = 'black'
    cntr_minr_style = '-'
    cntr_majr_style = '-'
    wshed_color = 'grey'

    cbar_kwargs = {'orientation': 'vertical', 'shrink': 1.0, 'label': 'Waterlevel Elevation\nChange (feet)',
                   'cax': cbar}

    c = wlmap.sel(year=year)

    c.where(mask).Head.plot.contourf(ax=ax, alpha=.7, extend='both', levels=levels, cmap='RdYlBu',
                                     cbar_kwargs=
                                         cbar_kwargs)
    c.where(mask).Head.plot.contour(ax=ax, color='k', colors='k', linewidths=.5, linestyles='solid', levels=levels)


    # add background maps
    conda_scripts.arich_functions.add_basemaps(ax,maptype='ctx.Esri.NatGeoWorldMap',)
    conda_scripts.arich_functions.add_basemap_transpo(ax=ax)
    conda_scripts.arich_functions.add_basemap_placenames(ax = ax)
    conda_scripts.rich_gis.plot_eastside_fault(ax=ax)

    conda_scripts.rich_gis.plot_water_shed(ax = ax, cur_basin=basin, edgecolor=wshed_color)
    ax.set_title('', fontsize=12)
    hand  = None


    hand = ph.add_marker_to_legend(axlegend, handles=hand, color='b',
                                   marker=None, linestyle='-',
                                   label='Basin Boundary')

    hand = ph.add_marker_to_legend(axlegend, handles=hand, color=wshed_color,
                                   marker=None, linestyle='-',
                                   label='Basin Watershed')

    axlegend.legend(handles=hand, loc='center',  bbox_to_anchor=(.5, .5), fontsize=10,mode = 'expand', edgecolor = None)
    plt.savefig(os.path.join(out_folder, f'{figure_num} WL change {depth}_{season}.png'), dpi=250)
    plt.close()

def plot_year_storage_change(basin, year, depth, season, figure_num, datafolder, out_folder,mod = None, title = None):

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
    if mod is None:
        mod = pp.wl_ch(basin)

    dfall, stor_ch_dict = mod.load_hist()
    storch_xr = stor_ch_dict[f'{depth}, {season}']

    c = storch_xr.sel(year=year)
    mask = pp.get_mask(basin)
    # get cell area
    cell_area = get_cell_area(c)
    # convert storage change from ft^3 per cell, to ft^3 per ft
    c = c.where(mask).Head / cell_area

    basshp = conda_scripts.rich_gis.get_active_subbasins()
    basshp = basshp.loc[[basin.upper()], :]

    path = os.path.join(datafolder, f"wl_change_predictions_{depth}_{season}.netcdf")
    print(f"opening the following with rioxarray\n{path}")

    if title is None:
        title = f"Change in Groundwater Storage from {year-1} to {year}, {depth.capitalize()} Aquifer, {season.capitalize()}"

    fig, ax, axlegend, axtitle, axscale, cbar = conda_scripts.map_figure.make_map_figure(
        title,
        figure_number_text=figure_num,  add_colorbar_axis='lower left')

    conda_scripts.rich_gis.set_extent(ax, locname=basin, keep_init_axis_ratio=True)

    if depth.upper() == 'DEEP':
        levels = np.linspace(-.007, .007, 8)
        # fig = plt.figure(figsize=(6, 8), dpi=250)
    else:
        levels = np.linspace(-.7, .7, 8)
        # fig = plt.figure(figsize=(8, 6), dpi=250)


    basshp.plot(ax=ax, facecolor="None", edgecolor='b', zorder = 500)

    wshed_color = 'grey'


    label = 'Change in Groundwater Storage Due to Groundwter Elevation Changes, in acre-feet per acre'
    label = '\n'.join(textwrap.wrap(label, 30))

    cbar_kwargs = {'orientation': 'vertical', 'shrink': 1.0,  'label': label,
                   'cax': cbar}

    c.plot.contourf(ax=ax, alpha=.7, extend='both', cmap='RdYlBu', levels=levels,
                    cbar_kwargs=cbar_kwargs)
    c.plot.contour(ax=ax, color='k', colors='k', linewidths=.5, linestyles='solid', levels=levels)

    conda_scripts.arich_functions.add_basemaps(ax,maptype='ctx.Esri.NatGeoWorldMap',)
    conda_scripts.arich_functions.add_basemap_transpo(ax=ax)
    conda_scripts.arich_functions.add_basemap_placenames(ax = ax)
    conda_scripts.rich_gis.plot_eastside_fault(ax=ax)

    conda_scripts.rich_gis.plot_water_shed(ax = ax, cur_basin=basin, edgecolor=wshed_color)
    ax.set_title('', fontsize=12)
    hand  = None

    hand = ph.add_marker_to_legend(axlegend, handles=hand, color='b',
                                   marker=None, linestyle='-',
                                   label='Basin Boundary')

    hand = ph.add_marker_to_legend(axlegend, handles=hand, color=wshed_color,
                                   marker=None, linestyle='-',
                                   label='Basin Watershed')

    axlegend.legend(handles=hand, loc='center',  bbox_to_anchor=(.5, .5), fontsize=10,mode = 'expand', edgecolor = None)
    plt.savefig(os.path.join(out_folder, f'{figure_num} storage_change {depth}_{season}.png'), dpi=250)
    plt.close()


def get_cell_area(wl):
    dh = wl.easting.diff(dim='easting').data[0]
    dy = wl.northing.diff(dim='northing').data[0]
    cell_area = dh * dy

    return cell_area

def run_waterlevel(year = None):
    import datetime

    if year is None:
        year = datetime.datetime.now().year -1

    fignums = {'son':{'Deep':{'Spring':"Figure 3-5",'Fall':"Figure 3-6",},
                      'Shallow':{'Spring':"Figure 3-3",'Fall':"Figure 3-4",}},
               'srp':{'Deep':{'Spring':"Figure 3-5",'Fall':"Figure 3-6",},
                      'Shallow':{'Spring':"Figure 3-3",'Fall':"Figure 3-4",}},
              'pet':{'Deep':{'Spring':"erase",'Fall':"erase",},
                      'Shallow':{'Spring':"Figure 3-3",'Fall':"Figure 3-4", }}}

    fig_title = {'son':{'Deep':{'Spring':None,'Fall':None,},
                      'Shallow':{'Spring':None,'Fall':None}},
               'srp':{'Deep':{'Spring':None,'Fall':None,},
                      'Shallow':{'Spring':None,'Fall':None}},
                'pet':{'Deep':{'Spring':f"Groundwater Elevation Contour Map Spring {year}",
                        'Fall':f"Groundwater Elevation Contour Map Fall {year}"},
                    'Shallow':{'Spring':f"Groundwater Elevation Contour Map Spring {year}",
                        'Fall':f"Groundwater Elevation Contour Map Fall {year}", }}}

    for basin in [ 'srp', 'pet','son',]:
        mod = pp.wl_ch(basin)
        [[plot_year(basin, year, shallow, season, figure_num=fignums[basin][shallow][season], datafolder=mod.folder,
                    out_folder=mod.share_folder, title=fig_title[basin][shallow][season])
            for shallow in ('Shallow', 'Deep')]
            for season in ('Fall', 'Spring')]

def run_waterlevel_change(year):
    fignums = {'SON': {'Deep': {'Fall': "Figure 3-8", },
                       'Shallow': {'Fall': "Figure 3-7", }},
               'SRP': {'Deep': {'Fall': "Figure 3-8", },
                       'Shallow': {'Fall': "Figure 3-7", }},
               'PET': {'Deep': {'Fall': "erase", },
                       'Shallow': {'Fall': "Figure 3-5", }}}

    fig_title = {'SON': {'Deep': {'Spring': None, 'Fall': None, },
                         'Shallow': {'Spring': None, 'Fall': None}},
                 'SRP': {'Deep': {'Spring': None, 'Fall': None, },
                         'Shallow': {'Spring': None, 'Fall': None}},
                 'PET': {'Deep': {'Fall': f"Groundwater-Level Change, Fall {year - 1}-{year}"},
                         'Shallow': {'Fall': f"Groundwater-Level Change, Fall {year - 1}-{year}"}}}

    # for basin in [ 'SON', ]:
    for basin in ['SRP', 'PET', 'SON', ]:
        mod = pp.wl_ch(basin)
        for depth in ['Shallow', 'Deep']:
            for season in ['Fall']:
                plot_year_diff(basin, year=year, depth=depth, season=season,
                                       datafolder=mod.folder,
                                       figure_num=fignums[basin][depth][season],
                                       out_folder=mod.share_folder,
                                       title=fig_title[basin][depth][season])

def run_storage_change(year):
    fignums = {'SON': {'Deep': {'Fall': "Figure 3-12", },
                       'Shallow': {'Fall': "Figure 3-11", }},
               'SRP': {'Deep': {'Fall': "Figure 3-12", },
                       'Shallow': {'Fall': "Figure 3-11", }},
               'PET': {'Deep': {'Fall': "erase", },
                       'Shallow': {'Fall': "Figure 3-8", }}}

    fig_title = {'SON': {'Deep': {'Spring': None, 'Fall': None, },
                         'Shallow': {'Spring': None, 'Fall': None}},
                 'SRP': {'Deep': {'Spring': None, 'Fall': None, },
                         'Shallow': {'Spring': None, 'Fall': None}},
                 'PET': {'Deep': {'Fall': f"Change in Groundwater Storage, Fall {year - 1}-{year}"},
                         'Shallow': {'Fall': f"Change in Groundwater Storage, Fall {year - 1}-{year}"}}}

    # for basin in [ 'SON', ]:
    for basin in ['SRP', 'PET', 'SON', ]:
        mod = mod = pp.wl_ch(basin)
        mod.load_ml()

        for depth in ['Shallow', 'Deep']:
            for season in ['Fall']:
                plot_year_storage_change(basin, year=year, depth=depth, season=season,
                                                     datafolder=mod.folder,
                                                     figure_num=fignums[basin][depth][season],
                                                     out_folder=mod.share_folder,
                                                     title=fig_title[basin][depth][season],
                                                     mod=mod)



if __name__ == '__main__':
    # run_waterlevel(year = 2023)
    run_storage_change(2023)
    run_waterlevel_change(2023)

    # shallow = 'Shallow'
    # season = "Spring"
    # [[plot_year(basin, year, shallow, season) for shallow in ('Shallow', 'Deep')] for season in ('Fall', 'Spring')]
    #
    # basin = mod.basin
    # shallow = 'Shallow'
    # season = "Spring"
    # [[plot_year(basin, year, shallow, season) for shallow in ('Shallow', 'Deep')] for season in ('Fall', 'Spring')]