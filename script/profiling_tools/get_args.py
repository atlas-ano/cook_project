import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

h = 8192
m = 29568
term1_const = 4 * h**2 + 2 * h * m

json_file = ""
with open(json_file, 'r') as f:
    data = json.load(f)

tpots = data["itls"]
num_requests = len(tpots)
if num_requests == 0:
    raise ValueError("itls is empty!")

print(f"Found {num_requests} requests")

all_t = []
all_tpots = []
for step in range(1, len(tpots[0]) + 1):
    tpot = 0
    t = step * num_requests
    for req_id in range(len(tpots)):
        tpot += tpots[req_id][step - 1]
    tpot /= len(tpots)
    all_t.append(t)
    all_tpots.append(tpot)

all_t = np.array(all_t)
all_tpots = np.array(all_tpots)
print(f"Original samples: {len(all_tpots)}")

term1 = np.full_like(all_t, term1_const, dtype=np.float64)
term2 = 3 * h * all_t
X = np.column_stack([term1, term2])

model_init = LinearRegression(fit_intercept=False)
model_init.fit(X, all_tpots)
pred_init = model_init.predict(X)
residuals_init = all_tpots - pred_init

def is_outlier_mad(data, threshold=3.0):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.zeros_like(data, dtype=bool)
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.abs(modified_z_scores) > threshold

outlier_mask = is_outlier_mad(residuals_init, threshold=2.5)
inlier_mask = ~outlier_mask

print(f"Outliers detected: {np.sum(outlier_mask)}")
print(f"Samples retained: {np.sum(inlier_mask)}")

X_clean = X[inlier_mask]
y_clean = all_tpots[inlier_mask]

model = LinearRegression(fit_intercept=False)
model.fit(X_clean, y_clean)
C1, C2 = model.coef_

tpot_pred_full = model.predict(X)
tpot_pred_clean = model.predict(X_clean)

print(f"\nFitting results (after outlier removal):")
print(f"  C1 = {C1:.6e}")
print(f"  C2 = {C2:.6e}")

mae = mean_absolute_error(y_clean, tpot_pred_clean)
rmse = np.sqrt(mean_squared_error(y_clean, tpot_pred_clean))
r2 = r2_score(y_clean, tpot_pred_clean)

print(f"\nError metrics (inliers only):")
print(f"  MAE:  {mae:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  R²:   {r2:.6f}")

plt.figure(figsize=(10, 6))

plt.scatter(all_t[inlier_mask], all_tpots[inlier_mask], 
            alpha=0.6, s=5, label='Inliers', color='steelblue')

plt.scatter(all_t[outlier_mask], all_tpots[outlier_mask], 
            alpha=0.8, s=2.5, label='Outliers', color='red', marker='x')

t_unique = np.unique(all_t)
tpot_pred_mean = [model.predict(np.column_stack([
    np.full_like([t], term1_const),
    np.array([3 * h * t])
]))[0] for t in t_unique]

plt.plot(t_unique, tpot_pred_mean, 'r--', lw=2, label='Fitted Model')

y_inliers = all_tpots[inlier_mask]
if len(y_inliers) > 0:
    y_min, y_max = y_inliers.min(), y_inliers.max()
    margin = (y_max - y_min) * 0.05
    plt.ylim(y_min - margin, y_max + margin)

plt.xlabel('Num_tokens (t = step × num_requests)')
plt.ylabel('TPOT')
plt.title('TPOT vs Num_tokens (with Outlier Detection)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("tpot_vs_t_fit_clean.png", dpi=150)
print("Saved plot: tpot_vs_t_fit_clean.png")

residuals_clean = y_clean - tpot_pred_clean
plt.figure(figsize=(10, 4))
plt.scatter(all_t[inlier_mask], residuals_clean, alpha=0.6, s=15, color='purple')
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Num_tokens')
plt.ylabel('Residual (GT - Pred)')
plt.title('Residuals (Inliers Only)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("tpot_residuals_clean.png", dpi=150)
print("Saved plot: tpot_residuals_clean.png")

FIXED_C2 = 2.222142e-12

term1_clean = X_clean[:, 0]
term2_clean = X_clean[:, 1]

y_adj = y_clean - FIXED_C2 * term2_clean

if np.allclose(term1_clean, term1_const):
    C1_fixedC2 = np.mean(y_adj) / term1_const
else:
    C1_fixedC2 = np.dot(y_adj, term1_clean) / np.dot(term1_clean, term1_clean)

print(f"\n[Fixed C2 Analysis]")
print(f"Fixed C2 = {FIXED_C2:.6e}")
print(f"Fitted C1 (for current batch size) = {C1_fixedC2:.6e}")

tpot_pred_fixedC2 = C1_fixedC2 * term1_const + FIXED_C2 * (3 * h * all_t)

plt.figure(figsize=(10, 6))

plt.scatter(all_t[inlier_mask], all_tpots[inlier_mask], 
            alpha=0.6, s=5, label='Inliers', color='steelblue')

plt.scatter(all_t[outlier_mask], all_tpots[outlier_mask], 
            alpha=0.8, s=2.5, label='Outliers', color='red', marker='x')

plt.plot(t_unique, tpot_pred_mean, 'r--', lw=2, label='Fitted Model (free C1,C2)')

plt.plot(all_t, tpot_pred_fixedC2, 'g-.', lw=2, label=f'Fixed C2={FIXED_C2:.1e}')

y_inliers = all_tpots[inlier_mask]
if len(y_inliers) > 0:
    y_min, y_max = y_inliers.min(), y_inliers.max()
    margin = (y_max - y_min) * 0.05
    plt.ylim(y_min - margin, y_max + margin)

plt.xlabel('Num_tokens (t = step × num_requests)')
plt.ylabel('TPOT')
plt.title('TPOT vs Num_tokens (Fixed C2 Analysis)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("tpot_vs_t_fixedC2.png", dpi=150)
print("Saved plot: tpot_vs_t_fixedC2.png")


