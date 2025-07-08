import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Define matrices A and B as float64
A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.float64)

B = np.array([[7, 8],
              [9, 10],
              [11, 12]], dtype=np.float64)

# Transpose B for inner product logic (cols become rows)
B_T = B.T  # Shape: (2, 3)

# Convert A and B_T to DataFrames with separate columns
df_A = pd.DataFrame(A, columns=['a1', 'a2', 'a3'])
df_A['row_id'] = df_A.index

df_BT = pd.DataFrame(B_T, columns=['b1', 'b2', 'b3'])
df_BT['col_id'] = df_BT.index

# Connect and register in DuckDB
con = duckdb.connect()
con.register("a", df_A)
con.register("bt", df_BT)

# DuckDB matrix multiplication using array_inner_product with explicit casting
start_duck = time.perf_counter()
result_df = con.execute("""
SELECT
  a.row_id AS row,
  bt.col_id AS col,
  array_inner_product(
    CAST([a.a1, a.a2, a.a3] AS DOUBLE[]),
    CAST([bt.b1, bt.b2, bt.b3] AS DOUBLE[])
  ) AS value
FROM a
CROSS JOIN bt
ORDER BY row, col
""").fetchdf()
end_duck = time.perf_counter()
time_duck = end_duck - start_duck

# Reconstruct the result matrix
rows = result_df['row'].max() + 1
cols = result_df['col'].max() + 1
duckdb_result = np.zeros((rows, cols))
for _, r in result_df.iterrows():
    duckdb_result[int(r['row']), int(r['col'])] = r['value']

# NumPy einsum matrix multiplication
start_np = time.perf_counter()
numpy_result = np.einsum('ik,kj->ij', A, B)
end_np = time.perf_counter()
time_np = end_np - start_np

# Plot result matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(duckdb_result, cmap='Blues', interpolation='nearest')
axes[0].set_title('DuckDB array_inner_product')
axes[0].set_xticks(range(cols))
axes[0].set_yticks(range(rows))

# Add text annotations for values
for i in range(rows):
    for j in range(cols):
        axes[0].text(j, i, f'{duckdb_result[i, j]:.0f}', 
                    ha='center', va='center', color='white', fontweight='bold')

axes[1].imshow(numpy_result, cmap='Greens', interpolation='nearest')
axes[1].set_title('NumPy einsum')
axes[1].set_xticks(range(cols))
axes[1].set_yticks(range(rows))

# Add text annotations for values
for i in range(rows):
    for j in range(cols):
        axes[1].text(j, i, f'{numpy_result[i, j]:.0f}', 
                    ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

# Plot timing comparison
plt.figure(figsize=(5, 4))
plt.bar(['NumPy einsum', 'DuckDB inner_product'], [time_np, time_duck], color=['green', 'blue'])
plt.ylabel("Execution Time (seconds)")
plt.title("Matrix Multiplication Performance")
plt.tight_layout()
plt.show()

# Print result matrices and comparison
print("DuckDB result:\n", duckdb_result)
print("NumPy result:\n", numpy_result)
print("Match:", np.allclose(duckdb_result, numpy_result))
print(f"NumPy einsum time: {time_np:.6f} seconds")
print(f"DuckDB inner_product time: {time_duck:.6f} seconds")