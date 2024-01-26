import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import conda_scripts.gwplot_wiski as gw
import conda_scripts.rich_gis as rich_gis
import conda_scripts.gwplot_fancy as gwp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D




def plot_rmp_hydro_pred(modelname, rmp_hydro, fancy=False, skip=True, errors_keep_going=True, plotnearby = False, observed = None):
    csvname = os.path.join('GIS', 'hydro_experiment', modelname, 'cokrig_wl_predict.csv')
    if not os.path.exists(csvname):
        print(f'exporting {csvname}')
        rmp_hydro.to_csv(csvname)

    def plot_fancy():
        plt.figure()
        nwp = gwp.fancy_plot(name, group=None,
                             filename='fancytest.png',
                             allinfo=None,
                             do_regress=False)

        en = [g.Easting.unique()[0], g.Northing.unique()[0]]

        input_long_lat = rich_gis.conv_coord(en[0], en[1],
                                             inepsg='epsg:2226', outepsg='epsg:4326')

        nwp.do_plot(close_plot=False, input_long_lat=input_long_lat,plot_wet =False,seasonal = False, plot_dry= False,)

        xx = g.loc[:,'datetime'].replace(np.NaN, pd.NaT).values.astype("datetime64[D]")
        yy = g.loc[:,'predicted']

        nwp.upleft.scatter(xx,
                           yy,
                           s=10, c='None',
                           edgecolors='b', zorder=20,
                           marker='o', label='Predicted')

        if observed is None:
            pass
        else:
            # print(observed.head())
            # print(observed.columns)
            # print(observed.filter(like = 'tation'))
            cur = observed[observed.index == name]
            print(f'shape of observed training data for {name},   {cur.shape[0]}')
            nwp.upleft.scatter(cur.Timestamp,
                               cur.loc[:,'Manual Measurement'].values,
                               s=5, c='None',
                               edgecolors='r', zorder=25,
                               marker='o', label='Training Data')

        bbox_props = dict(boxstyle="Roundtooth,pad=0.3", fc="w", ec="b", lw=2)
        boxtext = f"{g.loc[:, 'welltype'].unique()[0]}"

        # depth_text = "\nshallow:{:},deep{:},other{:}".format(g.loc[:, 'Shallow'].unique(),
        #                                                      g.loc[:, 'Deep'].unique(),
        #                                                      g.loc[:, 'Other'].unique())
        # boxtext = boxtext + depth_text

        # nwp.upleft.annotate(xy=(1, 0,),
        #                     text=boxtext,
        #                     xycoords='axes fraction', va='bottom', ha='right', color='blue',
        #                     bbox=bbox_props)
        # nwp.upleft.text(0, 1, "{:} feet".format(g.loc[:, 'rasterelevation'].unique()), va='bottom',
        #                 transform=nwp.upleft.transAxes)
        nwp.upleft.legend().remove()

        # plot nearby observations
        if plotnearby:
            custom_leg = plot_nearby(name, dis, nwp.upleft)

        #         #plot modeled data if possible
        #         modeled.plot(name,nwp.upleft,custom_leg)

        plt.savefig(filename, bbox_inches='tight', dpi=250)
        plt.close()

    def plot_simple():
        plt.figure()
        fig, ax = plt.subplots()
        t = gw.wiski_plot(name, ax=ax)
        t.get_station_pars()
        t.plot_gw(True, 'SRP')

        xx = g.loc[:,'datetime'].replace(np.NaN, pd.NaT).values.astype("datetime64[D]")
        yy = g.loc[:,'predicted']
        ax.scatter(xx,
                           yy,)
        # ax.scatter(g.datetime, g.predicted.values, s=3, c='None',
        #            edgecolors='b', zorder=20, marker='x', label='Predicted')
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.savefig(filename, bbox_inches='tight', dpi=250)

    for name, g in rmp_hydro.groupby('Station_Na'):
        print(modelname, name)
        filename = os.path.join('GIS', 'hydro_experiment', modelname, name + '.png')

        if skip:
            if os.path.exists(filename):
                print('skipping')
                continue
            else:
                print('does not exist')

        try:
            if fancy:
                plot_fancy()
            else:
                plot_simple()
        except Exception as e:
            print(f'failed {name}\n {e}')
            if errors_keep_going:
                pass
            else:
                # raise ValueError(f"\n\nSomething went wrong\n\n{str(e)}\n")
                raise e

            if fancy:
                try:
                    plot_simple()
                except:
                    print('really didnt work')

            plt.close()
            plt.close(plt.gcf())
            plt.close('all')
            plt.clf()

def plot_nearby(name, dis, ax, obs_count_to_stop=5):
    '''
    use this to plot nearby wells in hydrographs below

    name: name of well
    dis: df of distance from get_nearby_wells
    ax: axes to plot
    obs_count_to_stop: number of wells that will stop plotting of new wells (wells counted only if they have>5observations each)

    '''

    nearest_wells = get_nearby_wells(dis, name, nwells=15, nmiles=10)

    if nearest_wells.shape[0] == 0:
        print('no Nearby wells. ending plotting')
        return None

    viridis = plt.cm.get_cmap('jet', len(nearest_wells))
    viridis = [viridis(x) for x in np.arange(len(nearest_wells))]
    obs_count = 0
    custom_leg = []

    for near, dist in nearest_wells.iterrows():
        if seas_info.index.str.contains(near).any():

            __vals = seas_info.loc[near, :].copy()

            if obs_count < obs_count_to_stop:
                pass
            else:
                print('stopping printing nearby as 5 have been plotted')
                break

            if __vals.shape[0] > 5:
                obs_count = obs_count + 1

            if isinstance(__vals, pd.Series):
                pass
            else:
                c = viridis.pop()
                ax.scatter(__vals.loc[:, 'Timestamp'], __vals.loc[:, 'Manual Measurement'],
                           s=3, c='None',
                           edgecolors=c, zorder=1, facecolor=c,
                           marker='X', label=near)

                custom_leg.extend([Line2D([0], [0], marker='X', linewidth=0,
                                          color=c, label=f"{near} ({(dist.values[0]):,.0f} ft.)",
                                          markerfacecolor=c, markersize=5)])

                ax.legend(handles=custom_leg, prop={'size': 8})


        else:
            pass

    return custom_leg

def get_nearby_wells(df, wellname, nwells=5, nmiles=1):
    if df.index.str.contains(wellname).any():
        near = df < (5260. * nmiles)
        names = df.columns[near.loc[wellname, :]]  # names of wells <1 mile away
        distance = df.loc[wellname, names].sort_values().head(nwells).to_frame('distance2well')
        return distance
    else:
        print('no wells found')
        return pd.Series()
