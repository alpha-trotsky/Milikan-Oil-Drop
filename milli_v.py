# Computes  the velocity for a dataset of time vs position datapoints, outputting the mm values for the positions as well as some velocity values


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

TRIM_START = 0.00     # trim 25% of start
TRIM_END = 0.00       # trim 15% of end
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

def analyze_velocity_csv(csv_path, output_path="velocity_dataset_1.xlsx"):
    df = pd.read_csv(csv_path, sep=None, engine="python")
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

# ==============================
# RUN SCRIPT
# ==============================
if __name__ == "__main__":
    analyze_velocity_csv("milikandatashared.csv")