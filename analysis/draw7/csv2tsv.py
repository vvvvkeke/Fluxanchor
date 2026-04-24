import pandas as pd

pd.read_csv('Eki2vivo_zscore_col_data.csv').to_csv('Eki2vivo_zscore_col_data.tsv', sep='\t', index=False)
pd.read_csv('Eki2vivo_zscore_row_data.csv').to_csv('Eki2vivo_zscore_row_data.tsv', sep='\t', index=False)
