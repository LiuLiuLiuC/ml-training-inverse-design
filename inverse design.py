import os
import joblib
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

PROJECT_ROOT = ""  # 
SAVE_DIR = os.path.join(PROJECT_ROOT, "")

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

try:
    scaler_tab = joblib.load(os.path.join(SAVE_DIR, ""))
    scaler_tmax = joblib.load(os.path.join(SAVE_DIR, ""))
    scaler_eff = joblib.load(os.path.join(SAVE_DIR, ""))
except Exception as e:
    print(f"❌ 无法加载 Scalers: {e}\n请检查路径 {SAVE_DIR} 中是否存在 .pkl 文件。")

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
    x = layers.GaussianNoise(0.01)(inputs) 
    
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    
    h, w, c = x.shape[1:]
    x = layers.Reshape((h * w, c))(x)
    positions = tf.range(start=0, limit=h*w, delta=1)
    pos_emb = layers.Embedding(input_dim=h*w, output_dim=c)(positions)
    x = x + pos_emb
    
    for _ in range(2):  
        x = transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1)
    
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
model_path = os.path.join(SAVE_DIR, "transformer_model_final.h5")
try:
    model.load_weights(model_path)
    print("✅ 预训练模型权重加载成功！")
except Exception as e:
    print(f"❌ 无法加载模型权重: {e}\n请检查 {model_path} 是否存在。")

def generate_periodic_grf(porosity, corr_length_um, rve_pixel_size=128, physical_size=100.0):
    sigma_px = (corr_length_um / physical_size) * rve_pixel_size
    noise = np.random.normal(0, 1, (rve_pixel_size, rve_pixel_size))
    field = gaussian_filter(noise, sigma=sigma_px, mode='wrap')
    threshold = np.percentile(field, porosity * 100)
    binary_mask = field > threshold
    return binary_mask.astype(np.float32) * 255.0

TARGET_T_MAX = 305
TARGET_EFFICIENCY = 0.3

BOUNDS = {
    'Porosity': (0.2, 0.8),
    'Thickness_um': (10, 100),
    'Alpha_eff': (10000, 50000),
    'CorrLength': (2.0, 10.0)
}

def batch_predict_performance(population):
    imgs, tabs = [], []
    for gene in population:
        porosity, thickness, alpha, corr_len = gene
        imgs.append(generate_periodic_grf(porosity, corr_len) / 255.0)
        tabs.append([porosity, corr_len, thickness, alpha])
        
    imgs_np = np.array(imgs).reshape(-1, 128, 128, 1)
    tabs_np = scaler_tab.transform(np.array(tabs))
    
    preds_scaled = model.predict([imgs_np, tabs_np], batch_size=len(population), verbose=0)
    
    preds_tmax = scaler_tmax.inverse_transform(preds_scaled[:, 0].reshape(-1, 1)).flatten()
    preds_eff = scaler_eff.inverse_transform(preds_scaled[:, 1].reshape(-1, 1)).flatten()
    
    return preds_tmax, preds_eff

class AdvancedGeneticOptimizer:
    def __init__(self, target_t, target_eff, pop_size=50, generations=30):
        self.target_t = target_t
        self.target_eff = target_eff
        self.pop_size = pop_size
        self.generations = generations
        self.population = []

    def init_population(self):
        self.population = []
        for _ in range(self.pop_size):
            gene = [random.uniform(*BOUNDS[key]) for key in BOUNDS.keys()]
            self.population.append(gene)

    def run(self):
        self.init_population()
        best_gene, best_fit = None, -1
        global_best_t, global_best_eff = 0, 0

        print(f"\n🚀 逆向设计启动 | 目标靶点: T_max={self.target_t}K, Efficiency={self.target_eff:.2%}")

        for gen in range(self.generations):
            preds_tmax, preds_eff = batch_predict_performance(self.population)
            
            scores = []
            for i in range(self.pop_size):
                loss_t = ((preds_tmax[i] - self.target_t) / self.target_t) ** 2
                loss_eff = ((preds_eff[i] - self.target_eff) / self.target_eff) ** 2
                fit_val = 1.0 / (loss_t + 2.0 * loss_eff + 1e-6) 
                scores.append(fit_val)
                
                if fit_val > best_fit:
                    best_fit = fit_val
                    best_gene = self.population[i]
                    global_best_t, global_best_eff = preds_tmax[i], preds_eff[i]

            if (gen + 1) % 5 == 0 or gen == 0:
                idx = np.argmax(scores)
                print(f"Gen {gen+1:02d} | 最佳预测: T_max={preds_tmax[idx]:.1f}, Efficiency={preds_eff[idx]:.2%} | 孔隙率={self.population[idx][0]:.2f}")

            sorted_indices = np.argsort(scores)[::-1]
            elite_count = int(self.pop_size * 0.4)
            survivors = [self.population[i] for i in sorted_indices[:elite_count]]

            new_pop = survivors[:] 
            decay_factor = 1.0 - (gen / self.generations) 

            while len(new_pop) < self.pop_size:
                parent1, parent2 = random.sample(survivors, 2)
                child = [0, 0, 0, 0]
                
                for idx, key in enumerate(BOUNDS.keys()):
                    if random.random() < 0.7:
                        alpha_weight = random.random()
                        child[idx] = alpha_weight * parent1[idx] + (1 - alpha_weight) * parent2[idx]
                    else:
                        child[idx] = parent1[idx] if random.random() < 0.5 else parent2[idx]
                    
                    if random.random() < 0.2:
                        max_delta = (BOUNDS[key][1] - BOUNDS[key][0]) * 0.15 * decay_factor
                        delta = random.uniform(-max_delta, max_delta)
                        child[idx] = np.clip(child[idx] + delta, BOUNDS[key][0], BOUNDS[key][1])
                new_pop.append(child)
            self.population = new_pop

        return best_gene, global_best_t, global_best_eff

try:
    optimizer = AdvancedGeneticOptimizer(TARGET_T_MAX, TARGET_EFFICIENCY, pop_size=50, generations=30)
    best_recipe, final_t, final_eff = optimizer.run()

    print("\n=== 最终推荐逆向设计配方 === ")
    print(f"孔隙率 (Porosity):    {best_recipe[0]:.4f}")
    print(f"厚度 (Thickness):     {best_recipe[1]:.2f} um")
    print(f"吸收系数 (Alpha_eff): {best_recipe[2]:.0f}")
    print(f"相关长度 (CorrLength): {best_recipe[3]:.2f} um")
    print(f"\n[验证] 预测 T_max: {final_t:.2f}K (目标 {TARGET_T_MAX}), 预测 Efficiency: {final_eff:.2%} (目标 {TARGET_EFFICIENCY:.2%})")

    print("\n生成最优 RVE 物理结构图...")
    physical_size_um = 100.0
    opt_rve_img = generate_periodic_grf(best_recipe[0], best_recipe[3], physical_size=physical_size_um)

    plt.rcParams.update({
        'font.size': 12, 'font.family': 'sans-serif',
        'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
        'axes.labelsize': 14
    })

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(opt_rve_img, cmap='gray', 
                   extent=[0, physical_size_um, 0, physical_size_um], 
                   origin='lower')

    ax.set_xlabel('X ($\mu m$)', fontweight='bold')
    ax.set_ylabel('Y ($\mu m$)', fontweight='bold')
    ax.set_title(f'Optimized RVE Topology\n(Porosity = {best_recipe[0]:.2f}, $L_c$ = {best_recipe[3]:.2f} $\mu m$)', 
                 pad=15, fontweight='bold')
    ax.grid(False)
    plt.tight_layout()

    save_dir = os.path.join(PROJECT_ROOT, "evaluation_plots")
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, '305,30.png')

    plt.savefig(plot_path, transparent=True, dpi=300)
    plt.show()

    print(f"RVE 微观结构高清图已保存至: {plot_path}")

except Exception as e:
    print(f"\n 运行时错误: {e}")