
import geopandas as gpd

class geology(object):


    def get_simple_geol(self, buffer=10000, plot=False, simple=True):
        gdb = r"C:\GIS\shapefiles\parcels\SGMA_rate.gdb"
        allbas = gpd.read_file(gdb, driver='FileGDB', layer='allbasin_mod_bound')
        allbas.geometry = allbas.buffer(buffer)

        dict_g = {'K': "Franciscan-Basement",
                  'M': "Franciscan-Basement",
                  'u': "Franciscan-Basement",
                  'w': "Quaternary-WilsonGrove",
                  'T': 'Tertiary volcanics',
                  'Q': 'Quaternary-WilsonGrove',
                  'P': 'Quaternary-WilsonGrove'}

        print('loading geology...')
        geol = gpd.read_file(r"T:\arich\GIS\shapefiles\Geology\CGS map\CA_geo_SP\cageol_poly_SP.shp")

        geol = geol.to_crs(2226)
        print('clipping geology to basin')
        geol_clip = gpd.clip(geol, allbas)
        geol_clip.loc[:, 'relabel'] = geol_clip.ORIG_LABEL.str[0]
        geol_clip.loc[:, 'relabel'] = geol_clip.loc[:, 'relabel'].replace(dict_g)
        c = geol_clip.loc[:, 'UNIT_LINK'] == "CATK;0"
        geol_clip.loc[c, 'relabel'] = 'Quaternary-WilsonGrove'

        if simple:
            print('returning simplified geology, with 1|0 for Quaternary-WilsonGrove')
            c = geol_clip.loc[:, 'relabel'] == "Quaternary-WilsonGrove"
            geol_clip.loc[:, 'Geol_Krig'] = 0
            geol_clip.loc[c, 'Geol_Krig'] = 1

            if plot:
                geol_clip.plot('Geol_Krig', legend=True, figsize=(8, 8))

        self.geol_clip = geol_clip

    def add_geol_to_gdf(self, gdf):
        print('adding geology indicator to gdf...')
        df = gpd.sjoin(gdf, self.geol_clip.loc[:, ['geometry', 'Geol_Krig']])
        return df
