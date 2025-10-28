# (used for quick checks, the values will be in m/s)
# Computes and graphs the velocity for a single data point


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys

def analyze_velocity_from_position(
    csv_path,
    trim_start=0.25,
    trim_end=0.15,
    num_splices=5,
    time_unc=0.05,
    pos_unc_px=0.5,
    px_per_mm=520.0,
    px_per_mm_unc=1.0
):
    # --- Load CSV ---
    df = pd.read_csv(csv_path, header=None, names=["time", "position"])
    
    # Convert to numeric, handling Excel error values
    t = pd.to_numeric(df["time"], errors='coerce').values
    y_px = pd.to_numeric(df["position"], errors='coerce').values

    # --- Clean & sort ---
    # Remove 0 values, NaN values, and any other invalid data
    mask = np.isfinite(t) & np.isfinite(y_px) & (y_px != 0)
    t, y_px = t[mask], y_px[mask]
    idx = np.argsort(t)
    t, y_px = t[idx], y_px[idx]

    if len(t) < 10:
        raise ValueError("Not enough valid data points.")

    # --- Remove plateaus (10+ identical positions) ---
    diff = np.diff(y_px)
    keep_mask = np.ones_like(y_px, dtype=bool)
    i = 0
    while i < len(y_px) - 10:
        if np.all(diff[i:i+10] == 0):
            j = i + 10
            while j < len(y_px) - 1 and y_px[j] == y_px[j+1]:
                j += 1
            keep_mask[i:j] = False
            i = j + 1
        else:
            i += 1
    t = t[keep_mask]
    y_px = y_px[keep_mask]

    # --- Convert px → mm → m ---
    y_mm = y_px / px_per_mm
    y_m = y_mm / 1000.0
    # Propagate uncertainty
    y_m_unc = np.sqrt((pos_unc_px / px_per_mm / 1000.0) ** 2 +
                      ((y_px * px_per_mm_unc) / (px_per_mm**2) / 1000.0) ** 2)

    # --- Trim data ---
    n = len(t)
    start_idx = int(n * trim_start)
    end_idx = int(n * (1 - trim_end))
    t_trimmed = t[start_idx:end_idx]
    y_trimmed = y_m[start_idx:end_idx]

    print(f"\nTrimming: removed first {trim_start*100:.1f}% and last {trim_end*100:.1f}%")
    print(f"Remaining points: {len(t_trimmed)} / {n}")

    # --- Helper: linear regression ---
    def linear_fit(x, y):
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        slope = model.coef_[0]
        intercept = model.intercept_
        y_pred = model.predict(x.reshape(-1, 1))
        residuals = y - y_pred
        s_res = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        return slope, intercept, s_res, y_pred

    # --- 1️⃣ Overall fit ---
    overall_slope, overall_int, overall_res, y_overall = linear_fit(t_trimmed, y_trimmed)
    overall_velocity = overall_slope  # m/s

    # --- 2️⃣ Main segments ---
    segments = np.array_split(np.arange(len(t_trimmed)), num_splices)
    segment_velocities = []
    segment_uncs = []

    for seg in segments:
        seg_t = t_trimmed[seg]
        seg_y = y_trimmed[seg]
        if len(seg_t) < 2:
            continue
        slope, intercept, s_res, y_pred = linear_fit(seg_t, seg_y)
        s_x = np.std(seg_t)
        slope_unc = s_res / (np.sqrt(len(seg_t)) * s_x)
        segment_velocities.append(slope)
        segment_uncs.append(slope_unc)

    # --- 3️⃣ Overlapping segments (half-shifted) ---
    overlap_velocities = []
    overlap_uncs = []
    seg_size = len(t_trimmed) // num_splices
    half_shift = seg_size // 2

    for i in range(num_splices - 1):
        start = i * seg_size + half_shift
        end = (i + 1) * seg_size + half_shift
        if end > len(t_trimmed):
            end = len(t_trimmed)
        if start >= end:
            continue
        seg_t = t_trimmed[start:end]
        seg_y = y_trimmed[start:end]
        if len(seg_t) < 2:
            continue
        slope, intercept, s_res, y_pred = linear_fit(seg_t, seg_y)
        s_x = np.std(seg_t)
        slope_unc = s_res / (np.sqrt(len(seg_t)) * s_x)
        overlap_velocities.append(slope)
        overlap_uncs.append(slope_unc)

    # --- 4️⃣ Stats ---
    mean_segment_v = np.mean(segment_velocities)
    std_segment_v = np.std(segment_velocities, ddof=1)
    mean_overlap_v = np.mean(overlap_velocities)
    std_overlap_v = np.std(overlap_velocities, ddof=1)

    # Measurement + calibration uncertainty
    meas_unc = np.sqrt((pos_unc_px / (px_per_mm * 1000.0) / (t_trimmed[-1] - t_trimmed[0])) ** 2 +
                       (time_unc * overall_velocity / (t_trimmed[-1] - t_trimmed[0])) ** 2)
    total_unc = np.sqrt(((std_segment_v + std_overlap_v) / 2) ** 2 + meas_unc ** 2)

    # --- 5️⃣ Print summary ---
    print("\n--- Velocity Analysis (Trimmed Region) ---")
    print(f"Overall best-fit velocity: {overall_velocity:.4e} m/s")
    print(f"Main segments: {[round(v, 4) for v in segment_velocities]}")
    print(f"Overlaps: {[round(v, 4) for v in overlap_velocities]}")
    print(f"Mean (main): {mean_segment_v:.4e} ± {std_segment_v:.4e} m/s")
    print(f"Mean (overlap): {mean_overlap_v:.4e} ± {std_overlap_v:.4e} m/s")
    print(f"Combined uncertainty: ±{total_unc:.4e} m/s")

    # --- 6️⃣ Plot ---
    plt.figure(figsize=(10, 6))
    plt.scatter(t, y_m, s=8, color='gray', alpha=0.3, label='All data')
    plt.scatter(t_trimmed, y_trimmed, s=10, color='black', alpha=0.6, label='Trimmed data')

    plt.plot(t_trimmed, y_overall, 'r-', lw=2.5,
             label=f'Overall fit: {overall_velocity:.3e} m/s')

    colors_main = plt.cm.plasma(np.linspace(0, 1, len(segments)))
    colors_overlap = plt.cm.cividis(np.linspace(0, 1, len(overlap_velocities)))

    for i, seg in enumerate(segments):
        seg_t = t_trimmed[seg]
        seg_y = y_trimmed[seg]
        if len(seg_t) < 2:
            continue
        slope, intercept, _, y_pred = linear_fit(seg_t, seg_y)
        plt.plot(seg_t, y_pred, color=colors_main[i], lw=2,
                 label=f'Main {i+1}: {slope:.3e} m/s')

    for i in range(len(overlap_velocities)):
        start = i * seg_size + half_shift
        end = (i + 1) * seg_size + half_shift
        if end > len(t_trimmed):
            end = len(t_trimmed)
        seg_t = t_trimmed[start:end]
        seg_y = y_trimmed[start:end]
        if len(seg_t) < 2:
            continue
        slope, intercept, _, y_pred = linear_fit(seg_t, seg_y)
        plt.plot(seg_t, y_pred, '--', color=colors_overlap[i], lw=1.5,
                 label=f'Overlap {i+1}: {slope:.3e} m/s')

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"Position vs Time (Trimmed & Segmented: {num_splices} main + {num_splices-1} overlap)")
    plt.legend(fontsize=8, loc='best', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        "overall_velocity": overall_velocity,
        "overall_unc": total_unc,
        "segment_velocities": segment_velocities,
        "segment_uncs": segment_uncs,
        "overlap_velocities": overlap_velocities,
        "overlap_uncs": overlap_uncs,
    }


# --- Main ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python millikan_velocity_segments.py <data.csv> [trim_start trim_end num_splices]")
        sys.exit(1)

    csv_path = sys.argv[1]
    trim_start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    trim_end = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    num_splices = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    results = analyze_velocity_from_position(csv_path, trim_start, trim_end, num_splices)
    v = results["overall_velocity"]
    dv = results["overall_unc"]
    print(f"\nFinal result: v = {v:.4e} ± {dv:.4e} m/s")