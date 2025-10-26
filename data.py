import pandas as pd
import numpy as np

#measurement constants
r0m1, c0m1 = 0, 1
r0m2, c0m2 = 3, 1 
factor_right = 520
factor_left = 540
top = 1024

#loading the data 
def load_millikan_data(excel_file_path: str):
    """
    Load data from Excel file with two sheets for Millikan oil drop experiment.
    
    Args:
        excel_file_path: Path to the Excel file containing two sheets
        
    Returns:
        tuple: (velocity_df, voltage_df) - two pandas DataFrames
    """
    # Read both sheets from the Excel file
    method1_df = pd.read_excel(excel_file_path, sheet_name=0)  # First sheet
    method2_df = pd.read_excel(excel_file_path, sheet_name=1)   # Second sheet
    
    return method1_df, method2_df

# converting to mm 
def apply_px_to_mm(df, start_row, start_col, factor, top, ncols=None, inplace=True):
    """Convert pixels→mm for a rectangular region.
       start_row/start_col are 0-based. Set ncols=None to go to the last column."""
    c_end = None if ncols is None else start_col + ncols
    block = df.iloc[start_row:, start_col:c_end].apply(pd.to_numeric, errors="coerce")
    if pd.api.types.is_scalar(factor):
        out = (top - block) / factor
    else:
        # factor can be list or Series with length ncols
        f = pd.Series(factor, index=block.columns)
        out = (top - block).divide(f, axis="columns")
    if inplace:
        df.iloc[start_row:, start_col:c_end] = out
        return df
    return out

excel_file = "milikandata.xlsx"

method1_data, method2_data = load_millikan_data(excel_file)

apply_px_to_mm(method1_data, r0m1, c0m1, factor_right, top)
apply_px_to_mm(method2_data, r0m2, c0m2, factor_right, top)

'''if __name__ == "__main__":
    
    
    
    try:
        method1_data, method2_data = load_millikan_data(excel_file)
        
        print("Method 1 data shape:", method1_data.shape)
        print("Method 2 data shape:", method2_data.shape)
        print("\nMethod 1 data columns:", method1_data.columns.tolist())
        print("Method 2 data columns:", method2_data.columns.tolist())
        
        # Display first few rows
        print("\nMethod 1 data preview:")
        print(method1_data.head())
        print("\nMethod 2 data preview:")
        print(method2_data.head())
        
    except Exception as e:
        print(f"Error loading data: {e}")

    r0m1, c0m1 = 0, 1
    r0m2, c0m2 = 3, 1 
    factor_right = 520
    factor_left = 540
    top = 1024
    
    apply_px_to_mm(method1_data, r0m1, c0m1, factor_right, top)
    apply_px_to_mm(method2_data, r0m2, c0m2, factor_right, top)
    
    print("\nMethod 1 data millimetres:")
    print(method1_data.head())
    print("\nMethod 2 data millimetres:")
    print(method2_data.head())
        
    # Fake to check the conversion 
    df = pd.DataFrame({
        'A': ['100', '103', '#NV', '110'],
        'B': ['200', 'bad', '205', '210'],
        'C': ['x', 'y', 'z', '220'],
    })

    
    apply_px_to_mm(df, 0, 0, 520, 1024)
 
    print(df)'''