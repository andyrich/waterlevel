


def plot_stations():
    ax = stat_info_wisk.plot(figsize=(10, 10), marker='.', color='k')
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=2226, vmin=0, cmap='terrain')


def plot():
    ax = stat_info_wisk[stat_info_wisk.loc[:,'rasterelevation'].notnull()].plot(label = 'not missing elevations', marker = '.')
    stat_info_wisk[stat_info_wisk.loc[:,'rasterelevation'].isnull()].plot(ax= ax, c= 'r',marker = '.',
                                                                              label = 'missing elevations')
    ax.legend(bbox_to_anchor = (1,1))


if plot_all:
    ax = stat_info_wisk.plot('rasterelevation', legend=True, figsize=(10, 10))
    ctx.add_basemap(ax, crs=2226)