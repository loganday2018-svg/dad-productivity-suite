"""
Debug script to inspect Excel file structure.
"""

import sys
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python debug_excel.py path/to/file.xlsx")
    sys.exit(1)

file_path = sys.argv[1]

print(f"Loading: {file_path}\n")

# Load workbook
xls = pd.ExcelFile(file_path, engine='openpyxl')

print(f"Sheet names ({len(xls.sheet_names)}):")
for i, name in enumerate(xls.sheet_names, 1):
    print(f"  {i}. {name}")

print("\n" + "="*60)

# Inspect RWW sheet
sheet_name = 'RWW'
print(f"\nInspecting sheet: {sheet_name}")
print("="*60)

df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

print(f"\nShape: {df.shape} (rows x columns)")

print("\nFirst 20 rows:")
print(df.head(20).to_string())

print("\n" + "="*60)
print("Looking for header row...")

for idx in range(min(20, len(df))):
    row_str = ' | '.join([str(val)[:20] for val in df.iloc[idx].values[:10] if pd.notna(val)])
    print(f"Row {idx:2d}: {row_str}")

print("\n" + "="*60)
print("Row 14 in detail (detected as header):")
if len(df) > 14:
    header_row = df.iloc[14]
    for idx, val in enumerate(header_row[:20]):
        if pd.notna(val):
            print(f"  Col {idx:2d}: {val}")
