#!/usr/bin/env python3
"""
Enhanced DuckDB Matrix Operations and Performance Analysis

This script demonstrates advanced matrix operations using DuckDB with focus on:
1. Basic vector operations (from previous work)
2. Matrix multiplication implementations
3. Performance benchmarking and comparison
4. DuckDB-optimized approaches vs Python-centric approaches
"""

import pandas as pd
import numpy as np
import duckdb
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

def create_test_matrices(rows: int = 100, cols: int = 100) -> Dict[str, Any]:
    """Create test matrices of various sizes for performance testing"""
    
    # Generate matrix A (rows x cols)
    matrix_a = np.random.randint(1, 10, size=(rows, cols))
    
    # Generate matrix B (cols x rows) for multiplication compatibility
    matrix_b = np.random.randint(1, 10, size=(cols, rows))
    
    # Also create vectors for basic operations
    vector_a = np.random.randint(1, 100, size=cols)
    vector_b = np.random.randint(1, 100, size=cols)
    
    return {
        'matrix_a': matrix_a,
        'matrix_b': matrix_b,
        'vector_a': vector_a,
        'vector_b': vector_b,
        'dimensions': (rows, cols)
    }

def setup_duckdb_matrices(con, test_data, table_suffix=""):
    """Store matrices in DuckDB using optimized table structure"""
    
    table_name = f"matrices{table_suffix}"
    
    # Drop existing table
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Create table for storing matrix data
    con.execute(f"""
        CREATE TABLE {table_name} (
            matrix_id VARCHAR,
            row_idx INTEGER,
            col_idx INTEGER,
            value INTEGER
        )
    """)
    
    # Insert matrix A
    matrix_a = test_data['matrix_a']
    for i in range(matrix_a.shape[0]):
        for j in range(matrix_a.shape[1]):
            con.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?)", 
                       ('A', i, j, int(matrix_a[i, j])))
    
    # Insert matrix B
    matrix_b = test_data['matrix_b']
    for i in range(matrix_b.shape[0]):
        for j in range(matrix_b.shape[1]):
            con.execute(f"INSERT INTO {table_name} VALUES (?, ?, ?, ?)", 
                       ('B', i, j, int(matrix_b[i, j])))
    
    # Also create vector table for basic operations
    vector_table = f"vectors{table_suffix}"
    con.execute(f"DROP TABLE IF EXISTS {vector_table}")
    con.execute(f"""
        CREATE TABLE {vector_table} (
            vector_a INTEGER[],
            vector_b INTEGER[]
        )
    """)
    
    # Insert vectors
    vector_a = [int(x) for x in test_data['vector_a']]
    vector_b = [int(x) for x in test_data['vector_b']]
    con.execute(f"INSERT INTO {vector_table} VALUES (?, ?)", (vector_a, vector_b))
    
    return table_name, vector_table

def benchmark_basic_operations(con, vector_table, label):
    """Benchmark basic vector operations using DuckDB"""
    
    results = {}
    
    print(f"\n{label} - Basic Operations Benchmarks:")
    print("-" * 40)
    
    # Element-wise addition
    start = time.time()
    result = con.execute(f"""
        SELECT list_apply(list_zip(vector_a, vector_b), x -> x[1] + x[2]) AS result
        FROM {vector_table}
    """).fetchall()
    add_time = time.time() - start
    results['addition'] = add_time
    print(f"Element-wise Addition: {add_time:.6f} seconds")
    
    # Element-wise multiplication
    start = time.time()
    result = con.execute(f"""
        SELECT list_apply(list_zip(vector_a, vector_b), x -> x[1] * x[2]) AS result
        FROM {vector_table}
    """).fetchall()
    mult_time = time.time() - start
    results['multiplication'] = mult_time
    print(f"Element-wise Multiplication: {mult_time:.6f} seconds")
    
    # Dot product
    start = time.time()
    result = con.execute(f"""
        SELECT list_sum(list_apply(list_zip(vector_a, vector_b), x -> x[1] * x[2])) AS dot_product
        FROM {vector_table}
    """).fetchall()
    dot_time = time.time() - start
    results['dot_product'] = dot_time
    print(f"Dot Product: {dot_time:.6f} seconds")
    
    return results

def matrix_multiply_duckdb_optimized(con, matrix_table, rows_a, cols_a, cols_b):
    """Optimized matrix multiplication using DuckDB with better indexing"""
    
    query = f"""
    WITH matrix_mult AS (
        SELECT 
            a.row_idx,
            b.col_idx,
            a.value * b.value as product
        FROM 
            {matrix_table} a,
            {matrix_table} b
        WHERE 
            a.matrix_id = 'A' 
            AND b.matrix_id = 'B'
            AND a.col_idx = b.row_idx
    )
    SELECT 
        row_idx,
        col_idx,
        SUM(product) as result_value
    FROM matrix_mult
    GROUP BY row_idx, col_idx
    ORDER BY row_idx, col_idx
    """
    
    start = time.time()
    result = con.execute(query).fetchall()
    end = time.time()
    
    return result, end - start

def benchmark_matrix_multiplication(con, matrix_table, test_data, label):
    """Benchmark different matrix multiplication approaches"""
    
    rows_a, cols_a = test_data['dimensions']
    cols_b = rows_a  # Since matrix_b is cols_a x rows_a
    
    results = {}
    
    print(f"\n{label} - Matrix Multiplication Benchmarks:")
    print("-" * 50)
    
    # NumPy baseline for comparison
    start = time.time()
    numpy_result = np.dot(test_data['matrix_a'], test_data['matrix_b'])
    numpy_time = time.time() - start
    results['numpy'] = numpy_time
    print(f"NumPy Matrix Multiplication: {numpy_time:.6f} seconds")
    
    # DuckDB optimized approach
    duckdb_opt_result, duckdb_opt_time = matrix_multiply_duckdb_optimized(con, matrix_table, rows_a, cols_a, cols_b)
    results['duckdb_optimized'] = duckdb_opt_time
    print(f"DuckDB Optimized Approach: {duckdb_opt_time:.6f} seconds")
    
    # Performance comparison
    print(f"\nPerformance Ratios (vs NumPy):")
    print(f"DuckDB Optimized: {duckdb_opt_time/numpy_time:.2f}x slower")
    
    return results

def compare_approaches(con, matrix_table, test_data):
    """Compare DuckDB-focused vs Python-focused approaches"""
    
    results = {}
    
    # Python-focused approach
    df_a = pd.DataFrame(test_data['matrix_a'])
    df_b = pd.DataFrame(test_data['matrix_b'])
    
    start = time.time()
    result = df_a.dot(df_b)
    python_time = time.time() - start
    results['python_pandas'] = python_time
    
    # DuckDB-focused approach
    rows_a, cols_a = test_data['dimensions']
    
    start = time.time()
    con.execute(f"DROP TABLE IF EXISTS matrix_result_{matrix_table}")
    con.execute(f"""
        CREATE TABLE matrix_result_{matrix_table} AS
        WITH matrix_mult AS (
            SELECT 
                a.row_idx,
                b.col_idx,
                a.value * b.value as product
            FROM 
                {matrix_table} a,
                {matrix_table} b
            WHERE 
                a.matrix_id = 'A' 
                AND b.matrix_id = 'B'
                AND a.col_idx = b.row_idx
        )
        SELECT 
            row_idx,
            col_idx,
            SUM(product) as result_value
        FROM matrix_mult
        GROUP BY row_idx, col_idx
    """)
    
    duckdb_time = time.time() - start
    results['duckdb_native'] = duckdb_time
    
    return results

def create_performance_visualization(basic_results, matrix_results, sizes):
    """Create comprehensive performance charts"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Basic operations performance
    operations = ['addition', 'multiplication', 'dot_product']
    for i, size in enumerate(sizes):
        times = [basic_results[i][op] for op in operations]
        ax1.plot(operations, times, marker='o', label=size)

    ax1.set_title('Basic Vector Operations Performance')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    ax1.set_yscale('log')

    # Matrix multiplication comparison
    matrix_ops = ['numpy', 'duckdb_optimized']
    for i, size in enumerate(sizes):
        times = [matrix_results[i][op] for op in matrix_ops]
        ax2.plot(matrix_ops, times, marker='s', label=size)

    ax2.set_title('Matrix Multiplication Performance')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend()
    ax2.set_yscale('log')

    # Performance scaling with matrix size
    matrix_sizes = [100, 2500, 10000]  # 10x10, 50x50, 100x100 flattened
    numpy_times = [res['numpy'] for res in matrix_results]
    duckdb_times = [res['duckdb_optimized'] for res in matrix_results]

    ax3.plot(matrix_sizes, numpy_times, 'bo-', label='NumPy')
    ax3.plot(matrix_sizes, duckdb_times, 'ro-', label='DuckDB Optimized')
    ax3.set_title('Scaling Performance with Matrix Size')
    ax3.set_xlabel('Matrix Elements')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.set_xscale('log')

    # Performance ratio (DuckDB vs NumPy)
    ratios = [duckdb_times[i]/numpy_times[i] for i in range(len(sizes))]
    ax4.bar(sizes, ratios, color=['lightblue', 'lightgreen', 'lightcoral'])
    ax4.set_title('DuckDB Performance Ratio vs NumPy')
    ax4.set_ylabel('Ratio (DuckDB Time / NumPy Time)')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('/workspace/duckdb_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    
    print("="*70)
    print("ENHANCED DUCKDB MATRIX OPERATIONS AND PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Initialize DuckDB connection
    con = duckdb.connect()
    print(f"DuckDB version: {duckdb.__version__}")
    
    # Create test data with different sizes
    test_data_small = create_test_matrices(10, 10)
    test_data_medium = create_test_matrices(50, 50)
    test_data_large = create_test_matrices(100, 100)
    
    print("\nTest matrices created:")
    print(f"Small: {test_data_small['dimensions']}")
    print(f"Medium: {test_data_medium['dimensions']}")
    print(f"Large: {test_data_large['dimensions']}")
    
    # Setup DuckDB tables
    small_matrix_table, small_vector_table = setup_duckdb_matrices(con, test_data_small, "_small")
    medium_matrix_table, medium_vector_table = setup_duckdb_matrices(con, test_data_medium, "_medium")
    large_matrix_table, large_vector_table = setup_duckdb_matrices(con, test_data_large, "_large")
    
    print("\nDuckDB tables created successfully!")
    
    # Run basic operations benchmarks
    sizes = ['Small (10x10)', 'Medium (50x50)', 'Large (100x100)']
    
    small_basic_results = benchmark_basic_operations(con, small_vector_table, "Small (10x10)")
    medium_basic_results = benchmark_basic_operations(con, medium_vector_table, "Medium (50x50)")
    large_basic_results = benchmark_basic_operations(con, large_vector_table, "Large (100x100)")
    
    basic_results = [small_basic_results, medium_basic_results, large_basic_results]
    
    # Run matrix multiplication benchmarks
    small_matrix_results = benchmark_matrix_multiplication(con, small_matrix_table, test_data_small, "Small (10x10)")
    medium_matrix_results = benchmark_matrix_multiplication(con, medium_matrix_table, test_data_medium, "Medium (50x50)")
    large_matrix_results = benchmark_matrix_multiplication(con, large_matrix_table, test_data_large, "Large (100x100)")
    
    matrix_results = [small_matrix_results, medium_matrix_results, large_matrix_results]
    
    # Compare approaches
    print("\n" + "="*60)
    print("PYTHON-FOCUSED vs DUCKDB-FOCUSED APPROACH COMPARISON")
    print("="*60)
    
    python_results = compare_approaches(con, medium_matrix_table, test_data_medium)
    
    print(f"\nMedium (50x50) Matrix Multiplication:")
    print(f"Python-focused (Pandas): {python_results['python_pandas']:.6f} seconds")
    print(f"DuckDB-focused (Native): {python_results['duckdb_native']:.6f} seconds")
    
    if python_results['python_pandas'] < python_results['duckdb_native']:
        ratio = python_results['duckdb_native'] / python_results['python_pandas']
        print(f"\nPython is {ratio:.2f}x faster for computation")
    else:
        ratio = python_results['python_pandas'] / python_results['duckdb_native']
        print(f"\nDuckDB is {ratio:.2f}x faster for computation")
    
    # Create visualization
    print("\nGenerating performance visualization...")
    create_performance_visualization(basic_results, matrix_results, sizes)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL PERFORMANCE ANALYSIS AND RECOMMENDATIONS")
    print("="*70)
    
    print("\n1. BASIC OPERATIONS SUMMARY:")
    print("-" * 30)
    for i, size in enumerate(sizes):
        print(f"\n{size}:")
        for op in ['addition', 'multiplication', 'dot_product']:
            print(f"  {op.replace('_', ' ').title()}: {basic_results[i][op]:.6f}s")
    
    print("\n2. MATRIX MULTIPLICATION SUMMARY:")
    print("-" * 35)
    for i, size in enumerate(sizes):
        print(f"\n{size}:")
        print(f"  NumPy: {matrix_results[i]['numpy']:.6f}s")
        print(f"  DuckDB Optimized: {matrix_results[i]['duckdb_optimized']:.6f}s")
        
        ratio = matrix_results[i]['duckdb_optimized'] / matrix_results[i]['numpy']
        print(f"  DuckDB vs NumPy Ratio: {ratio:.2f}x")
    
    print("\n3. RECOMMENDATIONS FOR DUCKDB USAGE:")
    print("-" * 40)
    print("✓ DuckDB excels at:")
    print("  - Large-scale data processing with SQL")
    print("  - Complex data transformations and aggregations")
    print("  - Integration with existing SQL workflows")
    print("  - Memory-efficient processing of large datasets")
    
    print("\n⚠ Consider NumPy/Python for:")
    print("  - Pure numerical computations (matrix multiplication)")
    print("  - Small to medium-sized matrices")
    print("  - Scientific computing workflows")
    
    print("\n🔧 OPTIMIZATION STRATEGIES:")
    print("-" * 25)
    print("1. Use DuckDB for data preprocessing and feature engineering")
    print("2. Export processed data to NumPy for intensive computations")
    print("3. Store results back in DuckDB for persistence and analysis")
    print("4. Leverage DuckDB's columnar storage for large datasets")
    print("5. Use DuckDB extensions for specialized operations")
    
    avg_ratio = sum([matrix_results[i]['duckdb_optimized'] / matrix_results[i]['numpy'] for i in range(len(sizes))]) / len(sizes)
    print(f"\nAverage DuckDB vs NumPy performance ratio: {avg_ratio:.2f}x")
    
    if avg_ratio > 10:
        print("→ NumPy is significantly faster for matrix operations")
    elif avg_ratio > 2:
        print("→ NumPy has moderate performance advantage")
    else:
        print("→ DuckDB and NumPy have comparable performance")
    
    print("\nAnalysis complete! Check the generated visualization for detailed insights.")
    
    # Close connection
    con.close()

if __name__ == "__main__":
    main()