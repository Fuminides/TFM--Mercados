import load_data as ld
import preprocessing as pr


dfs = ld.load_full_stock_data()
san = dfs['santander']
san = pr.full_preprocess(san)