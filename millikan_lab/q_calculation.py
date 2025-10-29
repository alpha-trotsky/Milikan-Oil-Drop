def stats_elementary_charge(qs, median_e, label):
    # Classify as 1e if rounded(q/median_e) == 1
    k = np.rint(qs / median_e)
    mask_1e = (k == 1)
    n_1e = np.sum(mask_1e)
    if n_1e > 0:
        median_1e = np.median(qs[mask_1e])
        mean_1e = np.mean(qs[mask_1e])
        print(f"  Median charge for 1e droplets ({label}): {median_1e:.3e} C (n={n_1e})")
        print(f"  Mean   charge for 1e droplets ({label}): {mean_1e:.3e} C (n={n_1e})")
    else:
        print(f"  No 1e droplets found for {label}.")
# Example usage (uncomment to run):
# plot_radius_and_c_vs_index(df)

def export_radius_c_table(df, filename="radius_C_table.csv"):
    """
    Exports a CSV table with columns: droplet index, r, dr, C, dC.
    Args:
        df (pd.DataFrame): DataFrame with columns 'Point', 'r (m)', 'dr (m)', 'C', 'dC'.
        filename (str): Output CSV filename.
    """
    export_cols = ["Point", "r (m)", "dr (m)", "C", "dC"]
    df_export = df[export_cols].copy()
    df_export.to_csv(filename, index=False)
    print(f"Exported radius/C table to {filename}")

# Example usage (uncomment to run):
# export_radius_c_table(df)
# using voltages and velocities, compute and graph charge

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Constants with uncertainties ---
rho_oil = 875.3       # kg/m^3 ± 0.5
drho_oil = 0.05        # kg/m^3 uncertainty
rho_air = 1.204       # kg/m^3 ± 0.1  
drho_air = 0.0005        # kg/m^3 uncertainty
eta = 1.827e-5        # Pa·s ± 0.1e-5
deta = 0.0005e-5         # Pa·s uncertainty
g = 9.80              # m/s^2 ± 0.01
dg = 0.005             # m/s^2 uncertainty
d = 6.0e-3            # m ± 0.1e-3
dd = 0.5e-5          # m uncertainty
b_p = 8.12e-8         # m ± 0.1e-8 (Cunningham constant b/p)
db_p = 0.005e-8         # m uncertaint

# --- Load Data ---
import os

# Get the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths
volt = pd.read_csv(os.path.join(current_dir, "voltage_full.csv"))
vel = pd.read_excel(os.path.join(current_dir, "..", "velocity_dataset_1.xlsx"), sheet_name=1)

# --- Preprocess velocity data ---
vel["ID"] = vel["Data Point"].str.extract(r"(\d+)").astype(float)
vel["Dir"] = vel["Data Point"].str.extract(r"([ud])")

# Separate rise and fall
v_up = vel[vel["Dir"] == "u"].set_index("ID")
v_down = vel[vel["Dir"] == "d"].set_index("ID")

# Helper: compute radius (iterative with Cunningham) with uncertainty
def compute_radius(vd, dv_d):
    # Initial radius calculation
    r = np.sqrt(9 * eta * vd / (2 * g * (rho_oil - rho_air)))
    
    # Initial radius uncertainty
    dr = r * np.sqrt((deta/eta)**2 + (dv_d/vd)**2 + (dg/g)**2 + 
                     (drho_oil + drho_air)**2 / (rho_oil - rho_air)**2) / 2
    
    # Iterative Cunningham correction
    for _ in range(3):
        C = 1 + b_p / r
        dC = db_p / r + b_p * dr / r**2  # Cunningham uncertainty
        r = np.sqrt(9 * eta * vd / (2 * g * (rho_oil - rho_air))) / np.sqrt(C)
        # Update radius uncertainty with Cunningham correction
        dr = r * np.sqrt((deta/eta)**2 + (dv_d/vd)**2 + (dg/g)**2 + 
                         (drho_oil + drho_air)**2 / (rho_oil - rho_air)**2 + 
                         (dC/C)**2) / 2
    return r, C, dr, dC

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

    # --- Radius with uncertainty ---
    r, C, dr, dC = compute_radius(vd, dv_d)

    # --- Method 1 (Vstop) with comprehensive error propagation ---
    q1 = (4/3) * np.pi * r**3 * (rho_oil - rho_air) * g * d / Vstop
    
    # Error propagation for Method 1 (no double-counting of velocity uncertainty)
    rel_dr = 3 * dr / r
    rel_dd = dd / d
    rel_dg = dg / g
    rel_drho = (drho_oil + drho_air) / (rho_oil - rho_air)
    rel_dV = 0.05
    dq1 = q1 * np.sqrt(
        rel_dr**2 +
        rel_dd**2 +
        rel_dg**2 +
        rel_drho**2 +
        rel_dV**2
    )

    # --- Method 2 (Vrise & rise/fall speeds) with comprehensive error propagation ---
    if not np.isnan(vu):
        q2 = (4/3) * np.pi * r**3 * (rho_oil - rho_air) * g * (vd + vu) * d / (Vrise * vd)
        # Error propagation for Method 2 (no double-counting of velocity uncertainty)
        dq2 = q2 * np.sqrt(
            (3 * dr / r)**2 +  # radius uncertainty (cubed, includes both velocities)
            (dd / d)**2 +      # plate separation uncertainty
            (dg / g)**2 +      # gravity uncertainty
            (drho_oil + drho_air)**2 / (rho_oil - rho_air)**2 +  # density uncertainty
            (0.05)**2          # voltage uncertainty (5%)
        )
    else:
        q2, dq2 = np.nan, np.nan

    results.append({
        "Point": i,
        "r (m)": r,
        "dr (m)": dr,
        "C": C,
        "dC": dC,
        "q_method1 (C)": q1,
        "dq_method1 (C)": dq1,
        "q_method2 (C)": q2,
        "dq_method2 (C)": dq2
    })

df = pd.DataFrame(results)

# Convert numeric columns to proper numeric types (this handles the NaN issue)
numeric_columns = ["r (m)", "dr (m)", "C", "dC", "q_method1 (C)", "dq_method1 (C)", "q_method2 (C)", "dq_method2 (C)"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Filter out unrealistic values ---
df = df[
    (
        (df["q_method1 (C)"] > 1e-20)
    ) | (
        (df["q_method2 (C)"] > 1e-20) 
    )
]

# Filter desired data points
df = df[df["Point"] >= 0]

df.to_csv("millikan_results.csv", index=False)

# Print the DataFrame for inspection
print("\n" + "="*80)
print("MILLIKAN RESULTS DATAFRAME")
print("="*80)
print(df.to_string(index=False))
print("\nRadius uncertainties dr (um):")
print((df['dr (m)'] * 1e6).to_numpy())
print("\nCunningham uncertainties dC:")
print(df['dC'].to_numpy())
print("="*80)

# --- Compute charge differences for histogram ---
valid_q = df["q_method1 (C)"].dropna().sort_values().values
diff_q = np.diff(valid_q)


# --- GCD-like estimate of e (median of differences) after filtering out small/rubbish charges ---
q_all = pd.concat([df["q_method1 (C)"], df["q_method2 (C)"]]).dropna().to_numpy()
q_all = np.sort(q_all[q_all > 0])
q_filtered = q_all[q_all >= 1.3e-19]
print(f"\nGCD-like estimate of e after filtering charges < 1.3e-19:")
print(f"  Number of valid charges: {len(q_filtered)}")
if len(q_filtered) > 1:
    diff_q = np.diff(q_filtered)
    valid_diffs = diff_q[diff_q > 0]
    if len(valid_diffs) > 0:
        e_est = np.median(valid_diffs)
        print(f"  Median of differences (GCD estimate): {e_est:.3e} C")
    else:
        print("  No valid charge differences for GCD estimate.")
else:
    print("  Not enough valid charges for GCD estimate.")

# robust "smallest" (median of 1–3 smallest) after filtering
if len(q_filtered) > 0:
    q_small = np.median(q_filtered[:min(3, len(q_filtered))])
    print(f"  q_small = {q_small:.3e} C")
    print(f"  Smallest 3 charges: {q_filtered[:min(3, len(q_filtered))]}")
    print(f"  All charges range: {q_filtered.min():.3e} to {q_filtered.max():.3e} C")

    scores, candidates = [], []
    for m in range(1, 6):  # Just check first 5 for debugging
        e = q_small / m
        ratios = q_filtered / e
        # distance to nearest integer multiple
        residuals = np.abs(ratios - np.rint(ratios))
        # dimensionless score so different e are comparable
        score = np.median(residuals)
        scores.append(score)
        candidates.append(e)
    best_m = int(np.argmin(scores)) + 1
    e_hat = candidates[best_m - 1]
    print(f"  Best divisor m: {best_m}")
    print(f"  Estimated elementary charge e = {e_hat:.3e} C")
else:
    print("  Not enough valid charges for robust GCD analysis.")

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


# --- Method 1 and Method 2: Final charge and uncertainty summary ---
q1_vals = df["q_method1 (C)"].dropna().values
q2_vals = df["q_method2 (C)"].dropna().values
dq1_vals = df["dq_method1 (C)"].dropna().values
dq2_vals = df["dq_method2 (C)"].dropna().values

def charge_stats(qs, dqs, label):
    mean = np.mean(qs)
    mean_unc = np.sqrt(np.sum(dqs**2)) / len(qs)
    median = np.median(qs)
    typical_unc = np.median(dqs) / np.sqrt(len(qs))
    print(f"\n{label} charge estimates:")
    print(f"  Mean   = ({mean:.3e} +/- {mean_unc:.3e}) C")
    print(f"  Median = ({median:.3e} +/- {typical_unc:.3e}) C")
    print(f"  Percent error (mean):   {abs(mean - 1.602176634e-19)/1.602176634e-19 * 100:.1f}%")
    print(f"  Percent error (median): {abs(median - 1.602176634e-19)/1.602176634e-19 * 100:.1f}%")
    # Distribution of integer multiples
    charges_per_drop = np.rint(qs / median)
    unique_charges, charge_counts = np.unique(charges_per_drop, return_counts=True)
    print("  Charge distribution (in units of e):")
    for n, count in zip(unique_charges, charge_counts):
        print(f"    {int(n)}e: {count} droplets ({count/len(charges_per_drop)*100:.1f}%)")
    # Count 1e, 2e, and others
    n1 = np.sum(charges_per_drop == 1)
    n2 = np.sum(charges_per_drop == 2)
    not_either = len(charges_per_drop) - n1 - n2
    print(f"    1e: {n1}, 2e: {n2}, other: {not_either}")

charge_stats(q1_vals, dq1_vals, "Method 1")
stats_elementary_charge(q1_vals, np.median(q1_vals), "Method 1")
stats_elementary_charge(q2_vals, np.median(q2_vals), "Method 2")
charge_stats(q2_vals, dq2_vals, "Method 2")




# --- Plot 1: Charge vs droplet index with error bars ---
plt.figure(figsize=(10, 6))
plt.errorbar(df["Point"], df["q_method1 (C)"] * 1e19, yerr=df["dq_method1 (C)"] * 1e19,
            fmt='o', label='Method 1 (Stopping Potential)', color='#4878CF', capsize=3)  # Muted blue
plt.errorbar(df["Point"], df["q_method2 (C)"] * 1e19, yerr=df["dq_method2 (C)"] * 1e19,
            fmt='s', label='Method 2 (Rising Potential)', color='#B03A2E', capsize=3)  # Muted red

plt.xlabel(r"Droplet Index")
plt.ylabel(r"Charge ($\times 10^{-19}$ C)")
plt.title("Measured Charges of Oil Droplets")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Histogram of charge values ---
plt.figure(figsize=(10, 6))
plt.hist(valid_q * 1e19, bins=25, color='#8FAE5D', edgecolor='#556B2F', alpha=0.7)  # Sage green

#plt.axvline(e_median * 1e19, color='#B03A2E', linestyle='--', linewidth=2,  # Darker red
           #label=r'$e = {:.2f} \times 10^{{-19}}$ C'.format(e_median*1e19))

plt.xlabel(r"Charge ($\times 10^{-19}$ C)")
plt.ylabel("Frequency")
plt.title("Distribution of Measured Charges")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 3: Histogram of charge differences ---
plt.figure(figsize=(10, 6))
plt.hist(valid_diffs * 1e19, bins=20, color='#F4D03F', edgecolor='#D4AC0D', alpha=0.7)  # Muted yellow
plt.xlabel(r"Charge Difference ($\times 10^{-19}$ C)")
plt.ylabel("Frequency")
plt.title("Distribution of Charge Differences")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()

# --- Publication Plot: Radius vs Droplet Index with C Overlay ---
def plot_radius_and_c_vs_index(df):
    """
    Plots droplet radius (with error bars) vs droplet index, and overlays the Cunningham correction factor C (with its uncertainty) on a secondary y-axis.
    Args:
        df (pd.DataFrame): DataFrame with columns 'Point', 'r (m)', 'dr (m)', 'C', 'dC'.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color_r = "#4274D2"  # Muted blue
    color_c = '#B03A2E'  # Muted red

    # Sort by droplet index for visual clarity
    df_sorted = df.sort_values('Point')
    x = np.array(df_sorted['Point'])
    r = np.array(df_sorted['r (m)']) * 1e6  # microns
    print(r)
    dr = np.array(df_sorted['dr (m)']) * 1e6
    print(dr)
    C = np.array(df_sorted['C'])
    dC = np.array(df_sorted['dC'])
    print(dC)


    # Radius with error bars
    ax1.errorbar(x, r, yerr=dr, fmt='o', color=color_r, capsize=3, label='Radius $r$')
    ax1.set_xlabel('Droplet Index')
    ax1.set_ylabel(r'Radius $r$ ($\mu$m)', color=color_r)
    ax1.tick_params(axis='y', labelcolor=color_r)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Cunningham correction on secondary axis
    ax2 = ax1.twinx()
    ax2.errorbar(x, C, yerr=dC, fmt='s', color=color_c, capsize=3, label='Cunningham $C$')
    ax2.set_ylabel('Cunningham Correction $C$', color=color_c)
    ax2.tick_params(axis='y', labelcolor=color_c)

    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Droplet Radius and Cunningham Correction vs Index')
    plt.tight_layout()
    plt.show()

# Example usage (uncomment to run):
plot_radius_and_c_vs_index(df)

print(df)

print(e_est)