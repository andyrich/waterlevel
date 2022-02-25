import os
import numpy as np
import setup_and_import
import predict
import pickle
import warnings
import basin_setup

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
basin = "PET"
filename_base = f'_v3_{basin}_allmodmonths_wtemp_editdata_manual_only_observed_to_2021'
# filename_base = 'temp'
filename_base, smooth, smooth_value, modweight, add_modeled, \
monthlytimestep, modeltype, nmonths, dayoffset,scale_data, \
deeplayer, add_temp, add_climate, filter_manual, obs_filename, xysteps = \
    basin_setup.basin_info(filename_base,basin)


reload = True
plot_all = False
netcdf_only = False
years = np.arange(1970,2022,1)



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
                                    scale_data = scale_data,
                                    basin = basin,
                                    dayoffset=dayoffset,
                                    deeplayer = deeplayer,
                                    xysteps = xysteps,
                                    add_climate = add_climate,
                                    add_temp=add_temp,
                                    plot_all=plot_all,
                                    nmonths=nmonths,
                                    filter_manual = filter_manual,
                                    obs_filename = obs_filename)

    krigobj.load_obs()

    krigobj.process_climate()

    krigobj.export_processed_ts()

    krigobj.filter_and_export_shp(allow_missing=True)

    # # keep observed elevations of observations (do not use DEM values)
    krigobj.add_krig_values()

    # krigobj.add_model_head()

    krigobj.join_well_data(depth_filter=None)

    krigobj.do_geophys()

    krigobj.load_elev_pred()

    krigobj.add_geophys_geol(pre_export=True)

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
pred.export_predicted()

pred.run_prediction_for_hydros()
# pred.plot_hydros(plot_train=False)

# #### maps
import project2d
gwmap = project2d.MapGW(pred,krigobj, smooth = smooth, smooth_value = smooth_value)

gwmap.plotmap(yearstep = years, seasons = ['Spring', 'Fall'],
              plot_residuals = True, netcdf_only=netcdf_only)


# #### maps
# import map_simple_obs
# gwmap = map_simple_obs.MapGW(pred,krigobj, smooth = smooth, smooth_value = smooth_value)
#
# gwmap.plotmap(yearstep = 1, seasons = ['Spring', 'Fall'], plot_residuals = False)
