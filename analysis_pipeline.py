import pandas as pd 
import numpy as np
from data import method1_data, method2_data
import re 
import math
# constant variables 
# E - charge to move up 
# v_down - terminal velocity of the drop down 
# v_up - terminal velocity of the drop up 

#finding V_UP for method 2
#
E_list = method2_data.iloc[0, 1:].astype(float).tolist()
print(E_list)


#Before proceeding with analysis, terminal velocities - v_up and v_down will be identified 

time_gap = 0.1 # (s) 
'''
def velocity_extrema_table(
    df,
    dt,
    meta_rows,
    interpolate_gaps: int | None = 2,   
    use_centered: bool = True,          
    clip_bounds: tuple[float,float] | None = None 
) -> list[tuple[str, float, float]]:
    """
    Returns a list of tuples: (column_name, vmax_mm_s, vmin_mm_s)
    No columns are added to df.
    """
    data = df.iloc[meta_rows:].copy()
    Y = data.apply(pd.to_numeric, errors="coerce")

    if interpolate_gaps is not None:
        Y = Y.interpolate(limit=interpolate_gaps, limit_direction="both")

    if use_centered:
        V = (Y.shift(-1) - Y.shift(1)) / (2*dt)      # centered derivative
    else:
        V = (Y.diff()) / dt                           # forward diff (v at i from i-1→i)

    if clip_bounds is not None:
        V = V.clip(lower=clip_bounds[0], upper=clip_bounds[1])

    # Per-column extrema ignoring NaNs
    vmax = V.max(skipna=True)
    vmin = V.min(skipna=True)

    # Produce the requested list of tuples
    out = [(col, float(vmax[col]), float(vmin[col])) for col in V.columns]
    return out

method1_velocities = velocity_extrema_table(method1_data, dt = time_gap, meta_rows = 2)
print(method1_velocities)
method2_velocities = velocity_extrema_table(method2_data, dt = time_gap, meta_rows = 5)
print(method2_velocities)'''

def _rolling_slope(Y: pd.DataFrame, dt: float, window: int = 5) -> pd.DataFrame:
    """
    Robust velocity (mm/s) over a centered rolling window.
    For each window: slope = (last - first) / (dt * (len-1)).
    """
    def w_slope(x):
        x = x.dropna()
        if len(x) >= 2:
            return (x.iloc[-1] - x.iloc[0]) / (dt * (len(x) - 1))
        return np.nan
    return Y.rolling(window, center=True, min_periods=2).apply(w_slope, raw=False)

# integrated to combate spikes 

def velocity_extrema_table_rolling(
    df: pd.DataFrame,
    dt: float = 0.1,
    start_row: int = 0,             # number of rows to skip at the top (meta rows)
    start_col: int = 0,             # number of columns to skip at the left (meta columns)
    window: int = 5,                # rolling window size - 5 selected to be most accurate
    interpolate_gaps: int | None = 2,   # fill up to N consecutive NaNs; None = no interp
    #select_rule: str | None = r"^\d+(\.\d+)?$",  # keep only columns whose *names* look numeric
    clip_bounds: tuple[float, float] | None = None,  # e.g., (-5, 5) mm/s
    convert_to_mps: bool = False    # True to convert mm/s -> m/s at the very end
) -> list[tuple[str, float, float]]:
    """
    Returns a list of (column_name, vmax, vmin) using a rolling-slope velocity (mm/s by default).
    Does not modify df.
    """
    # 1) slice data block (skip metadata rows/cols)
    block = df.iloc[start_row:, start_col:].copy()

    # 2) column selection: keep only position columns
    cols = list(block.columns)
    #if select_rule is not None:
       # rx = re.compile(select_rule)
        #cols = [c for c in cols if rx.match(str(c))]
    Y = block[cols].apply(pd.to_numeric, errors="coerce")

    # 3) interpolate small gaps only (optional)
    if interpolate_gaps is not None:
        Y = Y.interpolate(limit=interpolate_gaps, limit_direction="both")

    # 4) rolling slope (mm/s)
    V = _rolling_slope(Y, dt=dt, window=window)

    # 5) optional clipping to physical bounds
    if clip_bounds is not None:
        V = V.clip(lower=clip_bounds[0], upper=clip_bounds[1])

    # 6) extrema per column (ignore NaNs)
    vmax = V.max(skipna=True)
    vmin = V.min(skipna=True)

    # 7) units: convert to m/s if requested
    if convert_to_mps:
        vmax = vmax * 1e-3
        vmin = vmin * 1e-3

    return [(str(col), float(vmax[col]), float(vmin[col])) for col in V.columns]

method1_velocities = velocity_extrema_table_rolling(df = method1_data, dt = time_gap, start_row=1, start_col=1, window = 5, interpolate_gaps= None, clip_bounds= None, convert_to_mps=True)
print(method1_velocities)
method2_velocities = velocity_extrema_table_rolling(df = method2_data, dt = time_gap, start_row=4, start_col=1, window = 5, interpolate_gaps= None, clip_bounds= None, convert_to_mps=True)

# method 1 
# for method 1, both data sets will be used 

# q = C1 * (v_down)^ (3/2) / V_stop

def C1_stokes(eta, rho_oil, rho_air, g, d):
    rho_delta = rho_oil - rho_air
    return ((9*eta)/(2*rho_delta*g))**1.5 * ((4/3)*np.pi*rho_delta*g*d)

def q_method1_table(
    df: pd.DataFrame | list,
    vcol: str,             # column with v_down in m/s (use absolute value inside)
    Vstop_col: str,        # column with stopping voltage in V
    d: float,              # plate spacing [m]
    eta: float,            # dynamic viscosity [Pa·s]
    rho_oil: float,        # oil density [kg/m^3]
    rho_air: float = 1.204,  # air density [kg/m^3]
    g: float = 9.80,         # m/s^2
    use_cunningham: bool = False,
    b_const: float = 6.17e-8,  # [Pa·m]
    p_air: float = 101325.0    # [Pa]
) -> pd.DataFrame:
    """
    Returns a DataFrame with q (C) for each row where v_down (m/s) and V_stop (V) are present.
    If use_cunningham=True, solve r(v) and compute q = (4/3)π Δρ g d r^3 / V_stop.
    Otherwise, use the constant C1 from Stokes: q = C1 * |v|^(3/2) / V_stop.

    Accepts either:
      - a pandas DataFrame containing columns [Vstop_col, vcol], or
      - a list of triples: [(V_stop, v_up, v_down), ...]  (V_stop may be a string).
    """
    # --- Normalize input to a DataFrame with columns: ["Vstop_V", "v_mps"]
    if isinstance(df, (list, tuple)):
        # Expect triples: (V_stop, v_up, v_down); we take index 0 and 2
        raw = []
        for t in df:
            if len(t) < 3:
                raise ValueError("Each list entry must be a triple: (V_stop, v_up, v_down).")
            V_stop, _, v_down = t
            raw.append({"Vstop_V": V_stop, "v_mps": v_down})
        out = pd.DataFrame(raw)
    else:
        # DataFrame path: select and rename the two needed columns
        out = df[[Vstop_col, vcol]].copy()
        out = out.rename(columns={vcol: "v_mps", Vstop_col: "Vstop_V"})

    # Coerce to numeric (handles string voltages, etc.)
    out["Vstop_V"] = pd.to_numeric(out["Vstop_V"], errors="coerce")
    out["v_mps"]   = pd.to_numeric(out["v_mps"],   errors="coerce")
    out["v_abs"]   = out["v_mps"].abs()

    rho_delta = rho_oil - rho_air

    if not use_cunningham:
        C1 = C1_stokes(eta, rho_oil, rho_air, g, d)
        out["q_C"] = np.where(
            (out["Vstop_V"].ne(0)) & out["v_abs"].notna(),
            C1 * (out["v_abs"] ** 1.5) / out["Vstop_V"],
            np.nan
        )
        out["method"] = "stokes_v32"
        return out

    # Cunningham route
    A = 2.0 * rho_delta * g / (9.0 * eta)
    alpha = b_const / p_air

    # r(v) = -alpha/2 + sqrt((alpha/2)^2 + v/A)  with v = |v_down|
    inside = (0.5*alpha)**2 + (out["v_abs"] / A)
    r = -0.5*alpha + np.sqrt(np.maximum(inside, 0.0))  # guard tiny negatives due to fp error
    out["r_m"] = r.replace([np.inf, -np.inf], np.nan)

    # q = (4/3)π Δρ g d r^3 / Vstop
    out["q_C"] = np.where(
        (out["Vstop_V"].ne(0)) & out["r_m"].notna(),
        ((4.0/3.0) * np.pi * rho_delta * g * d * (out["r_m"]**3)) / out["Vstop_V"],
        np.nan
    )
    out["method"] = "cunningham"
    return out
# for method 1, we can use both of the data sets as they supply us with v_stable and v_down
# constants you plug in (examples)
eta = 1.827e-5       # Pa s viscosity of air # Pa·s (air) 
rho_oil = 875.3    # kg/m^3 
rho_air = 1.204     # kg/m^3
d = 6.00e-3        # m 

q_table = q_method1_table(
    df=method1_velocities,
    vcol="v_down_mps",
    Vstop_col="V_stop_V",
    d=d, eta=eta, rho_oil=rho_oil, rho_air=rho_air,
    use_cunningham=True
)

q_table2 = q_method1_table(
    df=method2_velocities,
    vcol="v_down_mps",
    Vstop_col="V_stop_V",
    d=d, eta=eta, rho_oil=rho_oil, rho_air=rho_air,
    use_cunningham=True
)

print(q_table)
print(q_table2)
# method 2 

def _r_from_vdown_cunningham(vd, eta, rho_oil, rho_air, g, b_const, p_air):
    # r = -α/2 + sqrt( (α/2)^2 + vd/A ),  with A = 2 Δρ g / (9 η), α = b/p
    rho_delta = rho_oil - rho_air
    A = 2.0 * rho_delta * g / (9.0 * eta)
    alpha = b_const / p_air
    inside = (0.5*alpha)**2 + np.maximum(vd, 0.0)/A
    return -0.5*alpha + np.sqrt(np.maximum(inside, 0.0))

def q_method2_table(
    data: pd.DataFrame | list,
    # For DataFrame input:
    vdown_col: str = "v_down_mps",
    vup_col:   str = "v_up_mps",
    E_col:     str = "E_V",        # you said E := V_UP (use volts directly per your spec)
    # For list-of-triples input:
    E_list: list[float] | None = None,   # aligned list of E (one per triple) if data is a list
    # Physics constants:
    eta: float = 1.827e-5,         # Pa·s
    rho_oil: float = 875.3,        # kg/m^3
    rho_air: float = 1.204,        # kg/m^3
    g: float = 9.80,               # m/s^2
    b_const: float = 6.17e-8,      # Pa·m
    p_air: float = 101325.0        # Pa
) -> pd.DataFrame:
    """
    Computes q = (m_oil - m_air) g (v_d + v_u) / (E v_d), with
    m_oil - m_air = Δρ * (4/3)π r^3, and r inferred from v_d (Stokes or Cunningham).
    Inputs v_d, v_u in m/s and E in V (per your instruction E := V_UP).
    Returns columns: [v_down_mps, v_up_mps, E_V, r_m, m_diff_kg, q_C, method]
    """
    # Normalize input to a DataFrame with v_d, v_u, E
    if isinstance(data, (list, tuple)):
        if E_list is None or len(E_list) != len(data):
            raise ValueError("For list input, provide E_list with same length as data.")
        rows = []
        for (trip, E) in zip(data, E_list):
            if len(trip) < 3:
                raise ValueError("Triples must be (V_stop, v_up_mps, v_down_mps).")
            _, v_up, v_down = trip
            rows.append({"v_down_mps": v_down, "v_up_mps": v_up, "E_V": E})
        df = pd.DataFrame(rows)
    else:
        df = data[[vdown_col, vup_col, E_col]].copy()
        df = df.rename(columns={vdown_col:"v_down_mps", vup_col:"v_up_mps", E_col:"E_V"})

    # Coerce numeric & take magnitudes
    df["v_down_mps"] = pd.to_numeric(df["v_down_mps"], errors="coerce").abs()
    df["v_up_mps"]   = pd.to_numeric(df["v_up_mps"],   errors="coerce").abs()
    df["E_V"]        = pd.to_numeric(df["E_V"],        errors="coerce")

    # Radius from v_down
    
    r = _r_from_vdown_cunningham(df["v_down_mps"].values, eta, rho_oil, rho_air, g, b_const, p_air)
    method = "method2_cunningham"
    df["r_m"] = r

    # Buoyant mass and q
    rho_delta = rho_oil - rho_air
    df["m_diff_kg"] = (4.0/3.0) * math.pi * rho_delta * (df["r_m"]**3)
    # Guard E=0 or NaN
    denom = (df["E_V"]/d).replace(0, np.nan) * df["v_down_mps"]
    df["q_C"] = (df["m_diff_kg"] * g * (df["v_down_mps"] + df["v_up_mps"])) / denom
    df["method"] = method

    return df[["v_down_mps","v_up_mps","E_V","r_m","m_diff_kg","q_C","method"]]

q_table3 = q_method2_table(
    data=method2_velocities,
    E_list=E_list,
    eta=eta, rho_oil=rho_oil, rho_air=rho_air, g=9.80,
 
)
print(q_table3)
#q analysis -

#candidate GCD and then divide - generate them and find which one fits best
# integer values would have a separation of integer multiples 

#second method - histogram of q values. Figure this out

#uncertainty propogation - figure that bitch out 