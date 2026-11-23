import pandas as pd
import numpy as np
from scipy.stats import qmc
import os

def generate_lhs_params_corrected(n_samples=4000, seed=):
    param_config = {
        'Porosity':     [0.20, 0.80],       
        'CorrLength':   [2.0, 10.0],        
        'Thickness_um': [10.0, 100.0],      
        'Alpha_intrinsic': [2.0e4, 1.0e5]
    }

    keys = list(param_config.keys())
    l_bounds = [x[0] for x in param_config.values()]
    u_bounds = [x[1] for x in param_config.values()]

    sampler = qmc.LatinHypercube(d=len(l_bounds), seed=seed, optimization="lloyd")
    sample = sampler.random(n=n_samples)
    data = qmc.scale(sample, l_bounds, u_bounds)
    df = pd.DataFrame(data, columns=keys)
    
    df['Alpha_eff'] = df['Alpha_intrinsic'] * (1 - df['Porosity'])


    df['Porosity'] = df['Porosity'].round(4)
    df['CorrLength'] = df['CorrLength'].round(2)
    df['Thickness_um'] = df['Thickness_um'].round(2)
    df['Alpha_intrinsic'] = df['Alpha_intrinsic'].round(1)
    df['Alpha_eff'] = df['Alpha_eff'].round(1)

    df.insert(0, 'ID', [f'rve_{i:05d}' for i in range(n_samples)])
    df['Status'] = 0
    return df

if __name__ == '__main__':


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata_df = generate_lhs_params_corrected(n_samples=N_SAMPLES)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    metadata_df.to_csv(output_path, index=False)

    print(metadata_df.head())
    print(metadata_df.describe())
