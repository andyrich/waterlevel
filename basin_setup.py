import numpy as np

def basin_info(filename_base = None,basin = 'SRP', experiment_number = 1,):
    if basin.upper() == 'SRP':
        if filename_base is None:
            filename_base = f'_FINALv8_{basin}_allmodmonths'

        smooth = False
        smooth_value = .25
        # smooth_value = {'SRP':.25,'SON':.5}[basin]
        # modweight=.05
        modweight = .25
        add_modeled = True
        nmonths = 120
        modeltype = 'regress_only'
        monthlytimestep = 1
        dayoffset = 30
        scale_data = True
        deeplayer = 2
        xysteps = 12
        add_temp = True
        add_climate = True
        filter_manual = True
        obs_filename = 'all_gw_for_surf_2022.csv'
        obs_filename = 'all_gw_for_surf_2023_03_09.csv'


    elif basin.upper() == "PET":
        if filename_base is None:
            filename_base = f'_GWLmod_{basin}'
        smooth = False
        smooth_value = -999
        modweight = .25
        add_modeled = True
        nmonths = 120
        modeltype = 'regress_only'
        monthlytimestep = 1
        dayoffset = 30
        scale_data = True
        deeplayer = 2
        xysteps = 3
        add_temp = True
        add_climate = True
        filter_manual = True
        # obs_filename = 'all_gw_for_surf_2022.csv'
        obs_filename = 'all_gw_for_surf_2023_03_09.csv'

    elif basin.upper() == 'SON' or basin.upper() == 'SV':
        if filename_base is None:
            filename_base = f'_FINAL_{basin}_allmodmonths'

        smooth = False
        smooth_value = .5
        # smooth_value = {'SRP':.25,'SON':.5}[basin]
        modweight=.05
        add_modeled = True
        nmonths = 120
        modeltype = 'regress_only'
        monthlytimestep = 1
        dayoffset = 30
        scale_data = True
        deeplayer = 3
        xysteps = 16
        add_temp = True
        add_climate = True
        filter_manual = True
        # obs_filename = 'all_gw_for_surf_2022.csv'
        obs_filename = 'all_gw_for_surf_2023_03_09.csv'
    else:
        raise ValueError('Enter correct basin name: SRP, PET, SON')


    return filename_base, smooth, smooth_value, modweight, add_modeled, monthlytimestep,\
           modeltype, nmonths, dayoffset,scale_data, deeplayer, add_temp, add_climate, filter_manual,\
           obs_filename, xysteps


