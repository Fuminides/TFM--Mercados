import load_data as ld
import preprocessing as pr
from pylab import rcParams

def load_frame(name):
    print("Cargando los datos de: " + name)
    df = dfs[name]
    df2 = ld.restore_df(df.ticker[0])
    
    if df2 is None:
        pr
        df = pr.full_preprocess(df)
        ld.save_df(df)
        
        return d
    else:
        return df2
        
dfs = ld.load_full_stock_data()

san = load_frame('santander')
bbva = load_frame('bbva')
fluidra = load_frame('fdr')
ibex = load_frame('ibex')
inditex = load_frame('inditex')
tel = load_frame('tel')
vol = load_frame('vol')

rcParams['figure.figsize'] = 15, 10