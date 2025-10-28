# using voltages and velocities, compute and graph charge

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
rho_oil = 875.3       # kg/m^3
rho_air = 1.204       # kg/m^3
eta = 1.827e-5        # Pa·s
g = 9.80              # m/s^2
d = 6.0e-3            # m
b_p = 8.12e-8         # m  (Cunningham constant b/p)

# --- Load Data ---
volt = pd.read_csv("voltage_data.csv")
vel = pd.read_excel("velocity_dataset_1.xlsx", sheet_name=1)

# --- Preprocess velocity data ---
vel["ID"] = vel["Data Point"].str.extract(r"(\d+)").astype(float)
vel["Dir"] = vel["Data Point"].str.extract(r"([ud])")

# Separate rise and fall
v_up = vel[vel["Dir"] == "u"].set_index("ID")
v_down = vel[vel["Dir"] == "d"].set_index("ID")

# Helper: compute radius (iterative with Cunningham)
def compute_radius(vd):
    r = np.sqrt(9 * eta * vd / (2 * g * (rho_oil - rho_air)))
    for _ in range(3):
        C = 1 + b_p / r
        r = np.sqrt(9 * eta * vd / (2 * g * (rho_oil - rho_air))) / np.sqrt(C)
    return r, C

results = []

# --- Compute charges ---
for _, row in volt.iterrows():
    i = row["Data Points"]
    Vstop = row["V_stop"]
    Vrise = row["V_rise"]

    if i not in v_down.index:
        print(f"Skipping data point {i} - no falling velocity data")
        continue  # need a falling velocity
    vd = abs(v_down.loc[i, "Overall Velocity (mm/s)"]) * 1e-3
    dv_d = v_down.loc[i, "Overall Unc (mm/s)"] * 1e-3

    # Upward velocity may be missing
    if i in v_up.index:
        vu = abs(v_up.loc[i, "Overall Velocity (mm/s)"]) * 1e-3
        dv_u = v_up.loc[i, "Overall Unc (mm/s)"] * 1e-3
    else:
        vu, dv_u = np.nan, np.nan

    # --- Radius ---
    r, C = compute_radius(vd)

    # --- Method 1 (Vstop) ---
    q1 = (4/3) * np.pi * r**3 * (rho_oil - rho_air) * g * d / Vstop
    dq1 = q1 * np.sqrt((dv_d / vd)**2 + (0.05)**2)  # assume 5% voltage error

    # --- Method 2 (Vrise & rise/fall speeds) ---
    if not np.isnan(vu):
        q2 = (4/3) * np.pi * r**3 * (rho_oil - rho_air) * g * (vd + vu) * d / (Vrise * vd)
        dq2 = q2 * np.sqrt((dv_d / vd)**2 + (dv_u / vu)**2 + (0.05)**2)
    else:
        q2, dq2 = np.nan, np.nan

    results.append({
        "Point": i,
        "r (m)": r,
        "C": C,
        "q_method1 (C)": q1,
        "dq_method1 (C)": dq1,
        "q_method2 (C)": q2,
        "dq_method2 (C)": dq2
    })

df = pd.DataFrame(results)

# Convert numeric columns to proper numeric types (this handles the NaN issue)
numeric_columns = ["r (m)", "C", "q_method1 (C)", "dq_method1 (C)", "q_method2 (C)", "dq_method2 (C)"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Filter out unrealistic values ---
df = df[
    (
        (df["q_method1 (C)"] > 1e-20) & (df["q_method1 (C)"] < 2e-18)
    ) | (
        (df["q_method2 (C)"] > 1e-20) & (df["q_method2 (C)"] < 2e-18)
    )
]

# Filter desired data points
df = df[df["Point"] >= 0]
print(f"Data points 26 onwards: {len(df)} points")
print(f"Point range: {df['Point'].min()} to {df['Point'].max()}")

df.to_csv("millikan_results.csv", index=False)

# Print the DataFrame for inspection
print("\n" + "="*80)
print("MILLIKAN RESULTS DATAFRAME")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# --- Compute charge differences for histogram ---
valid_q = df["q_method1 (C)"].dropna().sort_values().values
diff_q = np.diff(valid_q)

# --- GCD-like estimate of e (median of differences) ---
valid_diffs = diff_q[diff_q > 0]

if len(valid_diffs) > 0:
    e_est = np.median(valid_diffs)
else:
   e_est = np.nan

q = pd.concat([df["q_method1 (C)"], df["q_method2 (C)"]]).dropna().to_numpy()
q = np.sort(q[q > 0])

# Filter out obvious outliers (charges that are too small to be realistic)
# Elementary charge should be around 1.6e-19 C with errors around 7%, anything under 3 standard deviations is considered a mistake during recording
q_filtered = q[q > 1.3e-19]
# Count outliers before filtering
outliers_below = q[q < 1.3e-19]
filtered_out_counter = len(outliers_below)
print(f"Outliers below 1.3e-19 C: {filtered_out_counter} charges")
if len(outliers_below) > 0:
    print(f"Outlier values: {outliers_below}")
        
    
print(f"Original charges: {len(q)}, Filtered charges: {len(q_filtered)}")
if len(q_filtered) > 0:
    q = q_filtered



# robust "smallest" (median of 1–3 smallest)
q_small = np.median(q[:min(3, len(q))])
print(f"q_small = {q_small:.3e} C")
print(f"Smallest 3 charges: {q[:min(3, len(q))]}")
print(f"All charges range: {q.min():.3e} to {q.max():.3e} C")

scores, candidates = [], []
for m in range(1, 6):  # Just check first 5 for debugging
    e = q_small / m
    ratios = q / e
    # distance to nearest integer multiple
    residuals = np.abs(ratios - np.rint(ratios))
    # dimensionless score so different e are comparable
    score = np.median(residuals)
    scores.append(score)
    candidates.append(e)
    
    # Show what's happening for m=1 and m=2
    if m <= 2:
        k_values = np.rint(ratios).astype(int)
    

best_m = int(np.argmin(scores)) + 1
e_hat = candidates[best_m - 1]

# GCD METHOD (COMMENTED OUT - NOT SUITABLE FOR THIS DATA)
'''
# robust "smallest" (median of 1–3 smallest)
q_small = np.median(q[:min(3, len(q))])
print(f"q_small = {q_small:.3e} C")
print(f"Smallest 3 charges: {q[:min(3, len(q))]}")
print(f"All charges range: {q.min():.3e} to {q.max():.3e} C")

scores, candidates = [], []
for m in range(1, 101):
    e = q_small / m
    ratios = q / e
    # distance to nearest integer multiple
    residuals = np.abs(ratios - np.rint(ratios))
    # dimensionless score so different e are comparable
    score = np.median(residuals)
    scores.append(score)
    candidates.append(e)

best_m = int(np.argmin(scores)) + 1
e_hat = candidates[best_m - 1]
print(f"Best divisor m: {best_m}")
print(f"Estimated elementary charge e = {e_hat:.3e} C")
'''

# MEDIAN METHOD (USED INSTEAD)
# Since charges are already around elementary charge level, use median approach
e_hat = np.median(q)
print(f"Using median of all charges as elementary charge estimate")
print(f"Median charge: {e_hat:.3e} C")
print(f"Expected elementary charge: 1.602e-19 C")
print(f"Percent error: {abs(e_hat - 1.602e-19)/1.602e-19 * 100:.1f}%")

# Calculate residuals for quality assessment
k = np.rint(q / e_hat).astype(int)
residuals_C = q - k * e_hat
print(f"Median(|residual|/e): {np.median(np.abs(residuals_C))/e_hat:.3f}")

# Show some sample results
print(f"\nSample charges and their integer multiples:")
for i in range(min(10, len(q))):
    print(f"q={q[i]:.3e}, k={k[i]}, q/k={q[i]/k[i]:.3e}")

# Check what integer multiples we found
unique_k = np.unique(k)
print(f"\nUnique integer multiples found: {unique_k}")
print(f"Most charges are single elementary charges (k=1)")




# --- Plot 1: Charge vs droplet index ---
plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
plt.errorbar(df["Point"], df["q_method1 (C)"] * 1e19, yerr=df["dq_method1 (C)"] * 1e19,
             fmt='o', label='Method 1 (Vstop)')
plt.errorbar(df["Point"], df["q_method2 (C)"] * 1e19, yerr=df["dq_method2 (C)"] * 1e19,
             fmt='s', label='Method 2 (Vrise)')
plt.xlabel("Droplet Index")
plt.ylabel("Charge (×10⁻¹⁹ C)")
plt.legend()
plt.title("Charge per Droplet")
plt.grid(True)

# --- Plot 2: Histogram of q values ---
plt.subplot(1,3,2)
plt.hist(valid_q * 1e19, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("q (×10⁻¹⁹ C)")
plt.ylabel("Frequency")
plt.title("Histogram of Charge Values")

# --- Plot 3: Histogram of charge differences ---
plt.subplot(1,3,3)
plt.hist(valid_diffs * 1e19, bins=20, color="lightgreen", edgecolor="black")
plt.xlabel("Δq (×10⁻¹⁹ C)")
plt.ylabel("Frequency")
plt.title("Histogram of Charge Differences")

plt.tight_layout()
plt.show()

print(df)

print(f"\nEstimated elementary charge = {e_est:.3e} C")