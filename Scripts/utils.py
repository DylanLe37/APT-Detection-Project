import pandas as pd

def objectToCat(dataFrame):
    return pd.concat([dataFrame.select_dtypes([],['object']),
                      dataFrame.select_dtypes(['object']).apply(pd.Series.astype,dtype='category')]
                     ,axis=1).reindex(dataFrame.columns,axis=1)
