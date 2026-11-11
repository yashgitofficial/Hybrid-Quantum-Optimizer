import os
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver

# Load configurations
from config import RANDOM_SEED, DATA_FILE, BUDGET, CAPACITY, SELECTION_COUNT

def run_stage2():
    print("Loading Stage 1 Predictions and Stage 2 constraints...")
    
    # Robust path handling to ensure files are read from the root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preds_file = os.path.join(BASE_DIR, 'stage1_predictions.csv')
    excel_file = os.path.join(BASE_DIR, DATA_FILE)
    
    # 1. Load Data
    # (Note: If your stage1_predictions.csv saved to your Desktop instead of the project folder, 
    # simply drag it into the Atomiq_Hackathon folder in VS Code first).
    df_preds = pd.read_csv(preds_file)
    df_top10 = df_preds[df_preds['is_top_10'] == 1].copy()
    
    # Load Stage 2 operational inputs
    df_data = pd.read_excel(excel_file, sheet_name='Stage2_Data')
    
    # Merge Top 10 with their operational data
    df = pd.merge(df_top10, df_data, on='customer_id')
    
    # Calculate Expected Net Value[cite: 2]
    # ENV = (Expected Deposit Value * Predicted Probability) - Contact Cost
    df['expected_net_value'] = (df['expected_deposit_value'] * df['predicted_probability']) - df['contact_cost']
    
    print("\n--- Top 10 Candidate Customers ---")
    print(df[['customer_id', 'expected_net_value', 'contact_cost', 'capacity_units']])

    # 2. Formulate Binary Optimization (QUBO)[cite: 2]
    qp = QuadraticProgram('Campaign_Selection')
    
    # Add binary decision variables (x_i)
    for _, row in df.iterrows():
        qp.binary_var(name=row['customer_id'])
        
    # Objective: Maximize ENV
    linear_obj = {row['customer_id']: row['expected_net_value'] for _, row in df.iterrows()}
    qp.maximize(linear=linear_obj)
    
    # Constraint 1: Exact selection of 6 customers (SUM(x_i) == 6)
    qp.linear_constraint(linear={row['customer_id']: 1 for _, row in df.iterrows()}, sense='==', rhs=SELECTION_COUNT, name='exact_6')
    
    # Constraint 2: Budget <= 25
    qp.linear_constraint(linear={row['customer_id']: row['contact_cost'] for _, row in df.iterrows()}, sense='<=', rhs=BUDGET, name='budget')
    
    # Constraint 3: Capacity <= 22
    qp.linear_constraint(linear={row['customer_id']: row['capacity_units'] for _, row in df.iterrows()}, sense='<=', rhs=CAPACITY, name='capacity')
    
    print("\n--- QUBO Formulation Output ---")
    print(qp.export_as_lp_string())

    # 3. Exact Classical Solution (Brute Force Reference)
    # This solves the QUBO formulation exactly to give us our classical benchmark and optimality gap.
    print("\nRunning Exact Classical Eigensolver (Brute Force Reference)...")
    exact_mes = NumPyMinimumEigensolver()
    
    # MinimumEigenOptimizer automatically converts our linear constraints into QUBO penalties using auxiliary/slack variables[cite: 2]
    exact_optimizer = MinimumEigenOptimizer(exact_mes)
    exact_result = exact_optimizer.solve(qp)
    
    # Extract selected customers and metrics
    selected_ids_exact = [var.name for var, val in zip(qp.variables, exact_result.x) if val == 1.0]
    exact_obj_val = exact_result.fval

    # Calculate utilized budget and capacity
    selected_df = df[df['customer_id'].isin(selected_ids_exact)]
    total_budget_used = selected_df['contact_cost'].sum()
    total_capacity_used = selected_df['capacity_units'].sum()

    print(f"\nExact Classical Optimum Objective Value: {exact_obj_val:.4f}")
    print(f"Exact 6 Selected IDs: {selected_ids_exact}")
    print(f"Budget Used: {total_budget_used} / {BUDGET}")
    print(f"Capacity Used: {total_capacity_used} / {CAPACITY}")
    
    # 4. Generate Root-Level Result File[cite: 3]
    result_payload = {
        "candidates": df['customer_id'].tolist(),
        "selected_customers": selected_ids_exact,
        "total_budget_used": int(total_budget_used),
        "total_capacity_used": int(total_capacity_used),
        "objective_value": float(exact_obj_val),
        "classical_brute_force_optimum": float(exact_obj_val),
        "optimality_gap": 0.0,
        "best_bitstrings": "".join(str(int(val)) for val in exact_result.x)
    }
    
    out_file = os.path.join(BASE_DIR, 'stage2_selection.json')
    with open(out_file, 'w') as f:
        json.dump(result_payload, f, indent=4)
        
    print(f"\n[+] stage2_selection.json successfully generated at {out_file}!")

if __name__ == "__main__":
    run_stage2()