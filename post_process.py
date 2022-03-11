import numpy as np
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
        share_folder = info['share_folder']

        print(f'data folder is {data_folder}')

        folder = os.path.join(main_folder, data_folder)

        self.folder = folder
        self.basin = basin
        self.model_workspace = None

        out_folder = os.path.join('summary', data_folder)

        self.out_folder = out_folder
        print(f'out folder is {out_folder}')

        self.share_folder = share_folder
        print(f'share_folder is [where files shared with others will be placed] {share_folder}')
        if not os.path.exists(share_folder):
            os.mkdir(share_folder)

        self.ml = None
        self.mg = None
        self.sr = None
        self.ss = None
        self.sy = None
        self.mask = None
        self.thickness_shallow = None
        self.thickness_deep = None

        self.shallow = False
        self.deep = False
        self.thickness_shallow = None
        self.thickness_deep = None

        self.standard_units = True
        self.ml_ss_layer = None
        self.ml_sy_layer = None

        self.ml_shallow_layer = None
        self.ml_deep_layer = None

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
            deep = True
            thickness_deep = 300
            shallow = True
            thickness_shallow = 200
            ml_ss_layer = 3
            ml_sy_layer = None
            ml_shallow_layer = [0]
            ml_deep_layer = [2,3]# zero-based
        elif self.basin == 'son':
            ml = conda_scripts.sv_budget.load_sv_model.load_alternate(workspace=workspace,
                                                                      extra_packages = None,
                                                                      verbose=False,
                                                                      load_aq_props=True)

            ml.update_modelgrid()
            # todo add mg, sr output to son
            mg = None; sr = None
            deep = True
            thickness_deep = 300
            shallow = True
            thickness_shallow = 200
            ml_ss_layer = 3
            ml_sy_layer = None
            ml_shallow_layer = [0]
            ml_deep_layer = [3]# zero-based

        elif self.basin == 'pet':
            ml, mg, sr = conda_scripts.pvihm.load_pet_mod.get_model(fdis=None, new_ws = None,
                                                                    historical = True, load_aq_props=True)

            ml.update_modelgrid()
            deep = True
            thickness_deep = 1300
            shallow = True
            thickness_shallow = 200
            ml_ss_layer = 3
            ml_sy_layer = None
            ml_shallow_layer = [0]
            ml_deep_layer = [1,2,3]
        else:
            raise ValueError(f"{self.basin} not yet done")

        print('done loading ml')

        self.ml = ml
        self.mg = mg
        self.sr = sr
        self.standard_units = self.ml.dis.lenuni == 1
        self.deep = deep
        self.thickness_shallow = thickness_shallow
        self.shallow = shallow
        self.thickness_deep = thickness_deep
        self.ml_ss_layer = ml_ss_layer
        self.ml_sy_layer = ml_sy_layer
        self.ml_shallow_layer = ml_shallow_layer
        self.ml_deep_layer = ml_deep_layer

        if self.standard_units:
            self.unit = 1.
        else:
            self.unit =3.28084

    def load_hist(self):
        '''
        use this to
            interpolate ss/sy at ML-heads
            get waterlevel change between years, and between shallow/deep fall/spring


        :return:

        df of storage change for shallow/deep fall/spring
        '''


        year_to_stor_ch_xr = {}
        dfall = pd.DataFrame()
        for deep in [True, False]:
            for spring in [True, False]:

                if deep:
                    layer = self.ml_ss_layer #3
                    spec_storage = True
                    thickness = self.thickness_deep
                    print(f"doing deep.\nfrom layer {layer},\nthickness {thickness}\nspecific storage")
                else:
                    layer = self.ml_sy_layer #None
                    spec_storage = False
                    thickness = 1.0
                    print(f"doing shallow. \nfrom layer {layer},\n thickness {thickness}\n specific yield")

                mask = get_mask(self.basin)
                ss = get_ss(self.ml, layer, self.standard_units, spec_storage=spec_storage)
                wl = get_waterlevel_change_xr(self.folder, deep=deep, spring=spring)

                ssnew = interp_ss(ss, wl)

                if spec_storage:
                    self.ss = ssnew
                else:
                    self.sy = ssnew
                self.mask = mask

                # these are just for checking outputs
                get_stor_estimate(ssnew, mask, thickness=thickness, basin=self.basin)
                get_average_wl_change(wl, mask, deep, spring, self.out_folder)

                stor, stor_xr = get_storage_ts_in_af(ssnew, wl, thickness, deep, spring, self.out_folder, mask)
                # stor, stor_xr = get_storage_ts_in_af(ssnew, wl, thickness, deep, spring,self.out_folder, mask)

                depth, season = get_season_depth(deep, spring)

                stor = pd.concat([stor], keys=[f"{depth}, {season}"], names=['Depth', 'Season'], axis=1)

                dfall = pd.concat([dfall, stor], axis=1)

                # save year over year storage to dict
                year_to_stor_ch_xr[f"{depth}, {season}"] = stor_xr

        return dfall, year_to_stor_ch_xr