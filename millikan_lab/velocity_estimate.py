def print_example_velocity_tables(csv_path, droplet_indices=[1,25,50], px_per_mm=520, pos_unc_px=0.5, px_per_mm_unc=1, max_rows=8):
    """Output example velocity tables (time, position, uncertainty) for specified droplets (up/down tracks if available)."""
    import pandas as pd
    import numpy as np
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path))
    time = df.iloc[:,0].to_numpy()
    for idx in droplet_indices:
        for suffix in ['u','d']:
            col = f"{idx}{suffix}"
            if col not in df.columns:
                continue
            y_px = pd.to_numeric(df[col], errors='coerce').to_numpy()
            y_mm = y_px / px_per_mm
            y_unc = np.sqrt((pos_unc_px/px_per_mm)**2 + (px_per_mm_unc*y_px/px_per_mm**2)**2) * y_mm
            mask = np.isfinite(time) & np.isfinite(y_mm)
            t = time[mask]
            y = y_mm[mask]
            yerr = y_unc[mask]
            print(f"\nExample velocity table for Droplet {idx}{suffix} (first {max_rows} rows):")
            print(f"{'Time (s)':>10}  {'Position (mm)':>15}  {'Uncertainty (mm)':>18}")
            for i in range(min(len(t), max_rows)):
                print(f"{t[i]:10.4f}  {y[i]:15.5f}  {yerr[i]:18.5f}")
    print("\n===============================================\n")
def print_average_regression_fit_stats(csv_path, max_idx=66, px_per_mm=520, pos_unc_px=0.5, px_per_mm_unc=1):
    """Compute and print average regression/fit statistics (slope, R², reduced chi²) for all droplets up to index max_idx (inclusive), for both up and down tracks if available."""
    import pandas as pd
    import numpy as np
    from scipy.stats import linregress
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path))
    time = df.iloc[:,0].to_numpy()
    slopes, stderrs, r2s, red_chi2s, counts = [], [], [], [], 0
    for idx in range(1, max_idx+1):
        for suffix in ['u','d']:
            col = f"{idx}{suffix}"
            if col not in df.columns:
                continue
            y_px = pd.to_numeric(df[col], errors='coerce').to_numpy()
            y_mm = y_px / px_per_mm
            y_unc = np.sqrt((pos_unc_px/px_per_mm)**2 + (px_per_mm_unc*y_px/px_per_mm**2)**2) * y_mm
            mask = np.isfinite(time) & np.isfinite(y_mm)
            t = time[mask]
            y = y_mm[mask]
            yerr = y_unc[mask]
            if len(t) < 2:
                continue
            slope, intercept, r, p, stderr = linregress(t, y)
            y_fit = slope * t + intercept
            residuals = y - y_fit
            dof = len(t) - 2 if len(t) > 2 else 1
            chi2 = np.sum((residuals / (yerr + 1e-12))**2)
            red_chi2 = chi2 / dof if dof > 0 else np.nan
            r2 = r**2
            slopes.append(slope)
            stderrs.append(stderr)
            r2s.append(r2)
            red_chi2s.append(red_chi2)
            counts += 1
    if counts == 0:
        print("No droplets found up to index", max_idx)
        return
    print(f"\n=== Average Regression/Fit Statistics for Droplets 1 to {max_idx} ===")
    print(f"  Mean Slope (velocity): {np.mean(slopes):.5f} mm/s ± {np.std(slopes):.5f} mm/s")
    print(f"  Mean sq: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    print(f"  Mean Reduced chi sq: {np.mean(red_chi2s):.3f} ± {np.std(red_chi2s):.3f}")
    print(f"  N tracks: {counts}")
    print("===============================================\n")
def print_regression_fit_stats(csv_path, droplet_indices=[1,25,50], px_per_mm=520, pos_unc_px=0.5, px_per_mm_unc=1, num_segments=5):
    """Extract and print regression/fit statistics (slope, R², reduced chi²) for specified droplets (up/down tracks if available)."""
    import pandas as pd
    import numpy as np
    from scipy.stats import linregress
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path))
    time = df.iloc[:,0].to_numpy()
    print("\n=== Regression/Fit Statistics for Selected Droplets ===")
    for idx in droplet_indices:
        for suffix in ['u','d']:
            col = f"{idx}{suffix}"
            if col not in df.columns:
                continue
            y_px = pd.to_numeric(df[col], errors='coerce').to_numpy()
            y_mm = y_px / px_per_mm
            y_unc = np.sqrt((pos_unc_px/px_per_mm)**2 + (px_per_mm_unc*y_px/px_per_mm**2)**2) * y_mm
            mask = np.isfinite(time) & np.isfinite(y_mm)
            t = time[mask]
            y = y_mm[mask]
            yerr = y_unc[mask]
            if len(t) < 2:
                continue
            # Overall fit
            slope, intercept, r, p, stderr = linregress(t, y)
            y_fit = slope * t + intercept
            residuals = y - y_fit
            dof = len(t) - 2 if len(t) > 2 else 1
            chi2 = np.sum((residuals / (yerr + 1e-12))**2)
            red_chi2 = chi2 / dof if dof > 0 else np.nan
            r2 = r**2
            print(f"\nDroplet {idx}{suffix}:")
            print(f"  Slope (velocity): {slope:.5f} mm/s ± {stderr:.5f} mm/s")
            print(f"  R sq: {r2:.4f}")
            print(f"  Reduced chi sq: {red_chi2:.3f}")
            print(f"  N points: {len(t)}")
    print("\n===============================================\n")
def plot_velocity_segments(csv_path, droplet_idx=1, px_per_mm=520, pos_unc_px=0.5, px_per_mm_unc=1, num_segments=5):
    """Plot a single droplet's position vs time with segment fits, overlaps, and overall regression."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path))
    time = df.iloc[:,0].to_numpy()
    # Try both up and down, prefer 'u' if available
    for suffix in ['u','d']:
        col = f"{droplet_idx}{suffix}"
        if col in df.columns:
            y_px = pd.to_numeric(df[col], errors='coerce').to_numpy()
            break
    else:
        print(f"Droplet {droplet_idx} not found.")
        return
    y_mm = y_px / px_per_mm
    y_unc = np.sqrt((pos_unc_px/px_per_mm)**2 + (px_per_mm_unc*y_px/px_per_mm**2)**2) * y_mm
    mask = np.isfinite(time) & np.isfinite(y_mm)
    t = time[mask]
    y = y_mm[mask]
    yerr = y_unc[mask]
    if len(t) < 2:
        print("Not enough data for fit.")
        return
    # Segment indices
    n = len(t)
    seg_bounds = np.linspace(0, n, num_segments+1, dtype=int)
    segs = [(seg_bounds[i], seg_bounds[i+1]) for i in range(num_segments)]
    # Overlap segments (midpoints)
    overlaps = []
    for i in range(len(segs)-1):
        a_mid = (segs[i][0] + segs[i][1]) // 2
        b_mid = (segs[i+1][0] + segs[i+1][1]) // 2
        overlaps.append((a_mid, b_mid))
    # Fit segments
    seg_fits = []
    for a, b in segs:
        if b-a < 2:
            seg_fits.append((np.nan, np.nan, np.nan, np.nan))
            continue
        slope, intercept, r, p, stderr = linregress(t[a:b], y[a:b])
        seg_fits.append((slope, intercept, stderr, (a, b)))
    # Fit overlaps
    overlap_fits = []
    for a, b in overlaps:
        if b-a < 2:
            overlap_fits.append((np.nan, np.nan, np.nan, np.nan))
            continue
        slope, intercept, r, p, stderr = linregress(t[a:b], y[a:b])
        overlap_fits.append((slope, intercept, stderr, (a, b)))
    # Overall fit
    slope_all, intercept_all, r, p, stderr_all = linregress(t, y)
    # Plot
    plt.figure(figsize=(9,6))
    # Plot data first, then fits above
    plt.errorbar(t, y, yerr=yerr, fmt='o', ms=4, capsize=2, color='#4878CF', label='Data', alpha=0.8, zorder=1)
    # Plot segment fits (shortened)
    colors = plt.cm.viridis(np.linspace(0,1,num_segments))
    for i, (slope, intercept, stderr, (a, b)) in enumerate(seg_fits):
        if np.isnan(slope): continue
        tseg = t[a:b]
        if len(tseg) < 2: continue
        # Shorten: use only the central 60% of each segment
        seglen = len(tseg)
        start = int(seglen*0.2)
        end = int(seglen*0.8) if int(seglen*0.8) > start+1 else seglen
        tseg_short = tseg[start:end]
        plt.plot(tseg_short, slope*tseg_short+intercept, color=colors[i], lw=2, label=f'Segment {i+1}\nv={slope:.3f}±{stderr:.3f}', zorder=3)
    # Plot overlap fits (shortened)
    for i, (slope, intercept, stderr, (a, b)) in enumerate(overlap_fits):
        if np.isnan(slope): continue
        tseg = t[a:b]
        if len(tseg) < 2: continue
        seglen = len(tseg)
        start = int(seglen*0.2)
        end = int(seglen*0.8) if int(seglen*0.8) > start+1 else seglen
        tseg_short = tseg[start:end]
        plt.plot(tseg_short, slope*tseg_short+intercept, color='orange', lw=1.5, ls='--', alpha=0.7, label=None if i>0 else 'Overlaps', zorder=2)
    # Plot overall fit
    plt.plot(t, slope_all*t+intercept_all, color='black', lw=2.5, ls='-', label=f'Overall\nv={slope_all:.3f}±{stderr_all:.3f}', zorder=4)
    plt.xlabel('Time (s)', fontsize=13, fontweight='bold')
    plt.ylabel('Position (mm)', fontsize=13, fontweight='bold')
    plt.title(f'Velocity Estimation for Droplet {droplet_idx}', fontsize=15, fontweight='bold', pad=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l is not None:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    plt.legend(new_handles, new_labels, fontsize=11, frameon=True, loc='upper left')
    plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np
from scipy.stats import linregress
from openpyxl import Workbook

# ==============================
# CONFIGURATION
# ==============================
PX_PER_MM = 520        # conversion factor
PX_PER_MM_UNC = 1      # uncertainty of conversion
POS_UNC_PX = 0.5       # measurement uncertainty in pixels
TIME_UNC = 0.05        # time uncertainty (s)

TRIM_START = 0.0     # trim 25% of start
TRIM_END = 0.0      # trim 15% of end
NUM_SEGMENTS = 5       # number of main segments
STUCK_THRESHOLD = 10   # number of repeated points to remove

# ==============================
# HELPER FUNCTIONS
# ==============================


def remove_stuck_points(t, y, threshold=STUCK_THRESHOLD):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    keep = np.ones_like(y, dtype=bool)
    count = 0
    for i in range(1, len(y)):
        if y[i] == y[i - 1]:
            count += 1
        else:
            if count >= threshold - 1:
                keep[i - count - 1:i - 1] = False
            count = 0
    if count >= threshold - 1:
        keep[-count - 1:-1] = False
    return t[keep], y[keep]

def pixels_to_mm(y_px):
    y_px = np.asarray(y_px, dtype=float)
    y_mm = y_px / PX_PER_MM
    abs_unc = np.sqrt((POS_UNC_PX / PX_PER_MM) ** 2 + (PX_PER_MM_UNC * y_px / PX_PER_MM**2) ** 2) * y_mm
    return y_mm, abs_unc

def fit_velocity_safe(t, y):
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if len(t) < 2:
        return np.nan, np.nan, len(t)
    slope, intercept, r, p, stderr = linregress(t, y)
    return slope, stderr, len(t)

def segment_indices_proportional(n, num_segments):
    boundaries = np.linspace(0, n, num_segments + 1, dtype=int)
    return [(boundaries[i], boundaries[i+1]) for i in range(num_segments)]

def overlap_segments(segments):
    overlaps = []
    for i in range(len(segments) - 1):
        a_mid = (segments[i][0] + segments[i][1]) // 2
        b_mid = (segments[i+1][0] + segments[i+1][1]) // 2
        overlaps.append((a_mid, b_mid))
    return overlaps

# ==============================
# MAIN FUNCTION
# ==============================
import os 

def analyze_velocity_csv(csv_path, output_path="velocity_dataset_1.xlsx"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, csv_path))
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    time = df.iloc[:, 0].to_numpy()
    position_columns = df.columns[1:]

    for col in position_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Converted Positions"
    ws2 = wb.create_sheet("Velocity Summary")

    # Sheet1 header
    ws1.append(["time (s)"] + [c for col in position_columns for c in (col + " (mm)", col + " unc (mm)")])

    # Sheet2 header
    ws2.append(["Data Point", "Overall Velocity (mm/s)", "Overall Unc (mm/s)", "Points used"] +
               [f"Seg{i+1} (mm/s)" for i in range(NUM_SEGMENTS)] +
               [f"Seg{i+1} unc (mm/s)" for i in range(NUM_SEGMENTS)] +
               [f"Seg{i+1} points" for i in range(NUM_SEGMENTS)] +
               [f"Overlap{i+1} (mm/s)" for i in range(NUM_SEGMENTS-1)] +
               [f"Overlap{i+1} unc (mm/s)" for i in range(NUM_SEGMENTS-1)] +
               [f"Overlap{i+1} points" for i in range(NUM_SEGMENTS-1)] +
               ["Mean Velocity (mm/s)", "Std Dev (mm/s)"])

    # Sheet1: converted positions
    converted_data = [time]
    for col in position_columns:
        t_clean, y_clean = remove_stuck_points(time, df[col].to_numpy())
        y_mm, y_unc = pixels_to_mm(y_clean)
        y_full = np.full(len(df), np.nan)
        y_unc_full = np.full(len(df), np.nan)
        y_full[:len(y_mm)] = y_mm
        y_unc_full[:len(y_unc)] = y_unc
        converted_data.extend([y_full, y_unc_full])

    for i in range(len(df)):
        ws1.append([df.iloc[i, 0]] + [col[i] if not np.isnan(col[i]) else "" for col in converted_data[1:]])

    # Sheet2: velocities
    for col in position_columns:
        t_clean, y_clean = remove_stuck_points(time, df[col].to_numpy())
        y_mm, y_unc = pixels_to_mm(y_clean)

        # Trimmed
        n_clean = len(t_clean)
        start_idx = int(n_clean * TRIM_START)
        end_idx = int(n_clean * (1 - TRIM_END))
        t_trim = t_clean[start_idx:end_idx]
        y_trim = y_mm[start_idx:end_idx]

        # Overall
        slope_all, unc_all, pts_all = fit_velocity_safe(t_trim, y_trim)

        # Segments
        n_trim = len(t_trim)
        segs = segment_indices_proportional(n_trim, NUM_SEGMENTS)
        seg_slopes, seg_uncs, seg_pts = [], [], []
        for a,b in segs:
            v,v_unc,n_pts = fit_velocity_safe(t_trim[a:b], y_trim[a:b])
            seg_slopes.append(v)
            seg_uncs.append(v_unc)
            seg_pts.append(n_pts)

        # Overlaps
        overlaps = overlap_segments(segs)
        overlap_slopes, overlap_uncs, overlap_pts = [], [], []
        for a,b in overlaps:
            v,v_unc,n_pts = fit_velocity_safe(t_trim[a:b], y_trim[a:b])
            overlap_slopes.append(v)
            overlap_uncs.append(v_unc)
            overlap_pts.append(n_pts)

        # Mean/std
        all_vels = [v for v in seg_slopes + overlap_slopes + [slope_all] if not np.isnan(v)]
        mean_v = np.mean(all_vels) if len(all_vels) > 0 else np.nan
        std_v = np.std(all_vels) if len(all_vels) > 0 else np.nan

        ws2.append([col, slope_all, unc_all, pts_all] +
                   seg_slopes + seg_uncs + seg_pts +
                   overlap_slopes + overlap_uncs + overlap_pts +
                   [mean_v, std_v])

    wb.save(output_path)
    print(f" Analysis complete. Results saved to {output_path}")

import matplotlib.pyplot as plt

def plot_selected_droplets(csv_path, droplet_indices=[1,25,50], px_per_mm=520, pos_unc_px=0.5, px_per_mm_unc=1):
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_path))
    time = df.iloc[:,0].to_numpy()
    # Build column names for up and down
    cols = []
    for idx in droplet_indices:
        for suffix in ['u','d']:
            col = f"{idx}{suffix}"
            if col in df.columns:
                cols.append(col)
    # Assign yellow, green, blue to the three droplets (up/down same color)
    # Warm, tame, solid colors (no transparency)
    droplet_colors = ['#FFB347', '#A3D977', '#6EC6FF']  # warm orange, soft green, gentle blue
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8,5))
    color_idx = 0
    for idx in droplet_indices:
        color = droplet_colors[color_idx % len(droplet_colors)]
        for suffix in ['u', 'd']:
            col = f"{idx}{suffix}"
            if col in df.columns:
                y_px = pd.to_numeric(df[col], errors='coerce').to_numpy()
                y_mm = y_px / px_per_mm
                y_unc = np.sqrt((pos_unc_px/px_per_mm)**2 + (px_per_mm_unc*y_px/px_per_mm**2)**2) * y_mm
                label = f"Drop {idx} {'up' if suffix=='u' else 'down'}"
                ax.errorbar(time, y_mm, yerr=y_unc, fmt='o', ms=3, capsize=2, label=label, color=color, alpha=1, elinewidth=0.8, linewidth=0.8)
        color_idx += 1
    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Position (mm)', fontsize=13, fontweight='bold')
    ax.set_title('Position vs Time for Selected Droplets', fontsize=15, fontweight='bold', pad=12)
    ax.legend(frameon=True, fontsize=11, loc='upper right')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=11, width=0.8, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    fig.tight_layout()
    plt.show()


# running the script 
if __name__ == "__main__":
    plot_velocity_segments("millikan_full.csv", droplet_idx=1)
    plot_selected_droplets("millikan_full.csv", droplet_indices=[1,25,50])
    print_regression_fit_stats("millikan_full.csv", droplet_indices=[1,25,50])
    print_average_regression_fit_stats("millikan_full.csv", max_idx=66)
    print_example_velocity_tables("millikan_full.csv", droplet_indices=[1,25,50])