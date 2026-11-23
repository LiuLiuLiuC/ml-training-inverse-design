import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm  
import cv2             
RVE_PIXEL_SIZE = 128     
RVE_PHYSICAL_SIZE = 100.0 

def generate_periodic_grf(porosity, corr_length_um, rve_pixel_size=RVE_PIXEL_SIZE, physical_size=RVE_PHYSICAL_SIZE):

    sigma_px = (corr_length_um / physical_size) * rve_pixel_size

    noise = np.random.normal(0, 1, (rve_pixel_size, rve_pixel_size))

    field = gaussian_filter(noise, sigma=sigma_px, mode='wrap')

    threshold = np.percentile(field, porosity * 100)

    binary_mask = field < threshold

    img_array = binary_mask.astype(np.uint8) * 255

    return img_array

if __name__ == '__main__':

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        meta_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        exit()
    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df), unit="img"):
        img_id = row['ID']
        porosity = row['Porosity']
        corr_length = row['CorrLength']

        rve_img = generate_periodic_grf(
            porosity=porosity,
            corr_length_um=corr_length
        )

        save_path = os.path.join(OUTPUT_DIR, f"{img_id}.png")

        cv2.imwrite(save_path, rve_img)

    last_porosity = np.count_nonzero(rve_img) / rve_img.size
