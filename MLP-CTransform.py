import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
import random
from tensorflow.keras import layers, models, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_global_seed(42)

PROJECT_ROOT = ""  
metadata_path = os.path.join(PROJECT_ROOT, "", "")
results_path = os.path.join(PROJECT_ROOT, "", "")
img_dir = os.path.join(PROJECT_ROOT, "")
SAVE_DIR = os.path.join(PROJECT_ROOT, "")
os.makedirs(SAVE_DIR, exist_ok=True)


df_meta = pd.read_csv(metadata_path)
df_results = pd.read_csv(results_path)
df_merged = pd.merge(df_meta, df_results, on='ID', how='inner')

X_img_list = []
valid_indices = []
for idx, row in df_merged.iterrows():
    p = os.path.join(img_dir, f"{row['ID']}.png")
    if os.path.exists(p):
        img = load_img(p, color_mode='grayscale', target_size=(128, 128))
        X_img_list.append(img_to_array(img)/255.0)
        valid_indices.append(idx)

X_images = np.array(X_img_list).astype('float32')
df_final = df_merged.iloc[valid_indices]

X_tab_raw = df_final[['Porosity', 'CorrLength', 'Thickness_um', 'Alpha_eff']].values
Y_raw = df_final[['T_max', 'Efficiency']].values


X_train_img, X_test_img, X_train_tab_raw, X_test_tab_raw, Y_train_raw, Y_test_raw = train_test_split(
    X_images, X_tab_raw, Y_raw, test_size=0.15, random_state=42
)

scaler_tab = StandardScaler().fit(X_train_tab_raw)
scaler_tmax = StandardScaler().fit(Y_train_raw[:, 0].reshape(-1, 1))
scaler_eff = StandardScaler().fit(Y_train_raw[:, 1].reshape(-1, 1))

X_train_tab = scaler_tab.transform(X_train_tab_raw)
X_test_tab = scaler_tab.transform(X_test_tab_raw)

Y_train_tmax = scaler_tmax.transform(Y_train_raw[:, 0].reshape(-1, 1))
Y_train_eff = scaler_eff.transform(Y_train_raw[:, 1].reshape(-1, 1))
Y_train = np.hstack([Y_train_tmax, Y_train_eff])

Y_test_tmax = scaler_tmax.transform(Y_test_raw[:, 0].reshape(-1, 1))
Y_test_eff = scaler_eff.transform(Y_test_raw[:, 1].reshape(-1, 1))
Y_test = np.hstack([Y_test_tmax, Y_test_eff])

joblib.dump(scaler_tab, os.path.join(SAVE_DIR, "scaler_tabular.pkl"))
joblib.dump(scaler_tmax, os.path.join(SAVE_DIR, "scaler_tmax.pkl"))
joblib.dump(scaler_eff, os.path.join(SAVE_DIR, "scaler_efficiency.pkl"))

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Add()([x, inputs])
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(y)
    y = layers.Add()([y, x])
    return y

def build_hybrid_model_4param():

    inputs = layers.Input(shape=(128, 128, 1), name="img_input")
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)  
    
    h, w, c = x.shape[1:]
    x = layers.Reshape((h * w, c))(x)
    positions = tf.range(start=0, limit=h*w, delta=1)
    pos_emb = layers.Embedding(input_dim=h*w, output_dim=c)(positions)
    x = x + pos_emb
    
    for _ in range(3):
        x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    
    tab_input = layers.Input(shape=(4,), name="tab_input") 
    
    y = layers.Dense(32)(tab_input)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    
    y = layers.Dense(32)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    
    combined = layers.Concatenate()([x, y])
    z = layers.Dense(128, activation="relu")(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(64, activation="relu")(z)
    output = layers.Dense(2, activation="linear")(z)
    
    return Model(inputs=[inputs, tab_input], outputs=output)

model = build_hybrid_model_4param()
model.compile(optimizer=Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
model.summary()

callbacks_list = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=12, 
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,       
        patience=5, 
        min_lr=1e-6, 
        verbose=1
    )
]

history = model.fit(
    [X_train_img, X_train_tab], Y_train,
    validation_data=([X_test_img, X_test_tab], Y_test),
    epochs=80,            
    batch_size=32,
    callbacks=callbacks_list,
    verbose=1
)
save_path = os.path.join(SAVE_DIR, "transformer_model_final.h5")
model.save(save_path)