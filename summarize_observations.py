
import pandas as pd
import os


def load_obs(file):

    tab = pd.read_csv(file, usecols = ['Station Name', 'Timestamp'])
    tab.loc[:,'Timestamp'] = pd.to_datetime(tab.loc[:,'Timestamp'])

    return tab

def summarize():
    dfall = pd.DataFrame()

    print('summarizing measured and observed value counts for eacb basin...')
    for basin in ['SRP', "SON", 'PET']:
        file = os.path.join('regression_data', f"{basin}_seas_info.csv")
        print(f'loading {basin}')
        df = load_obs(file)

        df.loc[:,'type'] = df.loc[:,'Station Name'].apply(
            lambda x: x.startswith('r')).replace({True:'Model',False: 'Observed'})

        counts = df.loc[:,'type'].value_counts().to_frame("number of observations")
        summary = df.loc[:,['type', "Timestamp"]].groupby('type').describe()
        summary = summary.droplevel(0,1).loc[:,['first', 'last']].applymap(lambda x: x.strftime('%Y-%b'))
        counts = pd.concat([counts, summary], axis = 1)
        counts.columns = pd.MultiIndex.from_product([[basin], counts.columns])

        dfall = pd.concat([dfall, counts], axis=1)

    out = os.path.join('regression_data', f"seas_info_summary.xlsx")

    dfall = dfall.T
    print(dfall)
    dfall.to_excel(out)
    print(f"done. saved file to:\n\t{out}\n\n")

if __name__=='__main__':
    summarize()