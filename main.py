import os

import setup_and_import
import predict
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
basin = "SRP"
filename_base = f'_refractor_{basin}_regressonly_smoothed_wweightspt05_30daysoff_geol_model_v6'
# filename_base = f'_refractor_{basin}_basedataonly_v2'
# filename_base = f'_refractor_{basin}_only_scaled_v1'
# filename_base = '___test'

reload = False
if reload:
    with open(f'regression_data\\krig_pickle_obj_{basin}.pickle','rb') as pick:
        print('loading pickle object')
        krigobj = pickle.load(pick)
        assert krigobj.basin == basin, 'basin name doesnt match'

    krigobj.map_foldername = 'maps_' + filename_base
    krigobj.hydros_foldname = 'hydros_' + filename_base



else:
    krigobj = setup_and_import.Krig(add_modeled = True, filename_base = filename_base, scale_data = True, basin = basin,
                                    dayoffset=30)

    krigobj.load_obs()

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

    pick = open(f'regression_data\\krig_pickle_obj_{basin}.pickle','wb')
    print('saving pickle object')
    pickle.dump(krigobj, pick)
    pick.close()


print('\n\n\n\nstarting predictions..')
pred_col = ['rasterelevation',
            'Simple_Bou',
            'date_frac',
            'year_frac',
            'Shallow',
            'slope',
            'Deep',
            'Other',
            'Geol_Krig']

pred = predict.krig_predict(krigobj, pred_col=pred_col, option='regress_only', modweight=.05)
# pred.setup_prediction(test_size = .95)
pred.setup_prediction(test_size = .9999)
pred.run_prediction()
# pred.plot_predicted_hydros()

#### maps
import project2d
gwmap = project2d.MapGW(pred,krigobj,smooth = True)

gwmap.plotmap(yearstep = [2015, 2018], seasons = ['Fall Early', 'Spring', 'Fall'])