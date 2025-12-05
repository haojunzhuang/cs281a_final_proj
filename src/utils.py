import pandas as pd

def load_data(vitals_path='data/vitals_ts_small.csv', 
              cohort_path='data/icu_cohort_small.csv'):
    df_patients = pd.read_csv(cohort_path)
    df_vitals = pd.read_csv(vitals_path)
    df_vitals = df_vitals.sort_values(by=['stay_id', 'charttime'])
    df_vitals['charttime'] = pd.to_datetime(df_vitals['charttime'])
    return df_vitals, df_patients