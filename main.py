import os
import numpy as np
import setup_and_import
import predict
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
basin = "SRP"
filename_base = f'_FINALv8_{basin}_allmodmonths'
# filename_base = f'_FINAL_refractor_{basin}_regressonly_smoothed_wweightspt05_30daysoff_geol_model_v6'
# filename_base = f'_refractor_{basin}_basedataonly_v2'
# filename_base = f'_refractor_{basin}_only_scaled_v1'
# filename_base = '___test'

reload = False
smooth = False
# smooth_value = .25
smooth_value = {'SRP':.25,'SON':.5}[basin]
modweight=.05
add_modeled = True
nmonths = 120

plot_all = False
years = np.arange(1980,2022,1)

modeltype = 'regress_only'
# modeltype = "GradientBoostingRegressor"
monthlytimestep = 1

if reload:
    with open(f'regression_data\\krig_pickle_obj_{basin}.pickle','rb') as pick:
        print('loading pickle object')
        krigobj = pickle.load(pick)
        assert krigobj.basin == basin, 'basin name doesnt match'

    krigobj.map_foldername = 'maps_' + filename_base
    krigobj.hydros_foldname = 'hydros_' + filename_base

else:
    krigobj = setup_and_import.Krig(add_modeled = add_modeled,
                                    monthlytimestep = monthlytimestep,
                                    filename_base = filename_base,
                                    scale_data = True,
                                    basin = basin,
                                    dayoffset=30,
                                    deeplayer = 2,
                                    add_climate = True,
                                    plot_all=plot_all,
                                    nmonths=nmonths)

    krigobj.load_obs()

    krigobj.process_climate()

    krigobj.export_processed_ts()

    krigobj.filter_and_export_shp()

    # # keep observed elevations of observations (do not use DEM values)
    krigobj.add_krig_values()

    # krigobj.add_model_head()

    krigobj.join_well_data(depth_filter=None)

    krigobj.do_geophys()

    krigobj.load_elev_pred()

    krigobj.add_geophys_geol()

    krigobj.categorize_depths()

    krigobj.export_seas_info()

    pick = open(f'regression_data\\krig_pickle_obj_{basin}.pickle','wb')
    print('saving pickle object')
    pickle.dump(krigobj, pick)
    pick.close()


print('\n\n\n\nstarting predictions..')
pred_col = ['rasterelevation',
            'date_frac',
            'year_frac',
            'Shallow',
            'slope',
            'Deep',
            'Geol_Krig',
            'Simple_Bou',
            'Complete_B',
            'isostatic'
            ]

pred = predict.krig_predict(krigobj, pred_col=pred_col, option=modeltype, modweight=modweight,)
# pred.setup_prediction(test_size = .95)
pred.setup_prediction(test_size = .8)

pred.run_fit()
pred.run_prediction()
pred.plot_hydros(plot_train=False)


# #### maps
import project2d
gwmap = project2d.MapGW(pred,krigobj, smooth = smooth, smooth_value = smooth_value)



gwmap.plotmap(yearstep = years, seasons = ['Spring', 'Fall'], plot_residuals = True)


# #### maps
# import map_simple_obs
# gwmap = map_simple_obs.MapGW(pred,krigobj, smooth = smooth, smooth_value = smooth_value)
#
# gwmap.plotmap(yearstep = 1, seasons = ['Spring', 'Fall'], plot_residuals = False)
