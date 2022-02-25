import pandas as pd
from post_process_helpers import *

class wl_ch:

    def __init__(self, basin):
        import os
        from configparser import ConfigParser
        config_object = ConfigParser()

        p = 'config'
        file = os.path.join(p, "config_wlch.ini")

        if not os.path.exists(file):
            raise ValueError(f'file does not exist. {file}')

        config_object.read(file)
        basin = basin.lower()
        info = config_object[basin]
        data_folder = info['data_folder']
        main_folder = info['main_folder']

        print(f'data folder is {data_folder}')

        folder = os.path.join(main_folder, data_folder)

        self.folder = folder
        self.basin = basin
        self.model_workspace = None

        out_folder = os.path.join('summary', data_folder)

        self.out_folder = out_folder
        print(f'out folder is {out_folder}')

        self.ml = None
        self.mg = None
        self.sr = None
        self.ss = None
        self.sy = None
        self.mask = None

        if not os.path.exists(out_folder):
            print('making new folder')
            os.mkdir(out_folder)
        else:
            print('not making new folder')

        print('done setting up')

    def load_ml(self):

        workspace = conda_scripts.load_datasets.model_info.get_mod(self.basin, historical=True)
        self.model_workspace = workspace
        print(f'using model workspace: {workspace}')

        if self.basin == 'srp':
            ml = conda_scripts.srphm.load_srphm.load(workspace=workspace, extra_packages=['upw'])
            ml.update_modelgrid()
            # todo add mg, sr output to srp
            mg = None; sr = None
        elif self.basin == 'son':
            ml = conda_scripts.sv_budget.load_sv_model.load_alternate(workspace=workspace,
                                                                      extra_packages = None,
                                                                      verbose=False,
                                                                      load_aq_props=True)

            ml.update_modelgrid()
            # todo add mg, sr output to son
            mg = None; sr = None
        elif self.basin == 'pet':
            ml, mg, sr = conda_scripts.pvihm.load_pet_mod.get_model(fdis=None, new_ws = None,
                                                                    historical = True, load_aq_props=True)


            ml.update_modelgrid()
        else:
            raise ValueError(f"{self.basin} not yet done")

        print('done loading ml')

        self.ml = ml
        self.mg = mg
        self.sr = sr

    def load_hist(self):
        '''
        use this to
            interpolate ss/sy at ML-heads
            get waterlevel change between years, and between shallow/deep fall/spring


        :return:

        df of storage change for shallow/deep fall/spring
        '''

        standard_units = self.ml.dis.lenuni == 1

        dfall = pd.DataFrame()
        for deep in [True, False]:
            for spring in [True, False]:

                if deep:
                    layer = 3
                    spec_storage = True
                    depth = 1000
                    print(f"doing deep.\nfrom layer {layer},\ndepth {depth}\nspecific storage")
                else:
                    layer = np.nan
                    spec_storage = False
                    depth = 1.0
                    print(f"doing shallow. \nfrom layer {layer},\n depth {depth}\n specific yield")

                mask = get_mask()
                ss = get_ss(self.ml, layer, standard_units, spec_storage=spec_storage)
                wl = get_waterlevel_change_xr(self.folder, deep=deep, spring=spring)
                print(wl)
                ssnew = interp_ss(ss, wl)

                if spec_storage:
                    self.ss = ssnew
                else:
                    self.sy = ssnew
                self.mask = mask

                get_stor_estimate(ssnew, mask, depth=depth, basin=self.basin)

                get_average_wl_change(wl, mask, deep, spring, self.out_folder)
                                            # (ssnew, wl, depth, deep, spring, out_folder, mask)
                stor = get_storage_ts_in_af(ssnew, wl, depth, deep, spring,self.out_folder, mask)

                depth, season = get_season_depth(deep, spring)

                stor = pd.concat([stor], keys=[f"{depth}, {season}"], names=['Depth', 'Season'], axis=1)

                dfall = pd.concat([dfall, stor], axis=1)

        return dfall