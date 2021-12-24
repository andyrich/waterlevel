import numpy as np

def basin_info(filename_base = None,basin = 'SRP'):
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
        add_temp = True
        add_climate = True

    elif basin.upper() == "PET":
        raise ValueError('Petaluma not yet implemented')
    elif basin.upper() == 'SON' or basin.upper() == 'SV':
        if filename_base is None:
            filename_base = f'_FINALv8_{basin}_allmodmonths'

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
        add_temp = True
        add_climate = True
    else:
        raise ValueError('Enter correct basin name: SRP, PET, SON')


    return filename_base, smooth, smooth_value, modweight, add_modeled, monthlytimestep, modeltype, nmonths, dayoffset,scale_data, deeplayer, add_temp, add_climate

plot_all = False
years = np.arange(1980,2022,1)
