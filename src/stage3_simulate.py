import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

# Load configurations
from config import RANDOM_SEED

np.random.seed(RANDOM_SEED)

def exact_probability_at_least_3(probs):
    """Calculates exact classical probability of >= 3 successes out of 6 independent events."""
    # This is a Poisson Binomial distribution calculation
    n = len(probs)
    total_prob_ge_3 = 0.0
    
    # Calculate exactly k successes for k in [3, 4, 5, 6]
    for k in range(3, n + 1):
        for combo in combinations(range(n), k):
            p_combo = 1.0
            for i in range(n):
                if i in combo:
                    p_combo *= probs[i]
                else:
                    p_combo *= (1 - probs[i])
            total_prob_ge_3 += p_combo
            
    return total_prob_ge_3

def monte_carlo_simulation(probs, shots=10000):
    """Simulates the campaign using classical random sampling."""
    successes = 0
    for _ in range(shots):
        # Flip a weighted coin for each customer
        outcomes = [1 if np.random.random() < p else 0 for p in probs]
        if sum(outcomes) >= 3:
            successes += 1
    return successes / shots

def run_stage3():
    print("Loading Stage 2 selections and Stage 1 probabilities...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stage2_file = os.path.join(BASE_DIR, 'stage2_selection.json')
    stage1_file = os.path.join(BASE_DIR, 'stage1_predictions.csv')
    
    with open(stage2_file, 'r') as f:
        stage2_data = json.load(f)
        
    selected_customers = stage2_data['selected_customers']
    
    df_preds = pd.read_csv(stage1_file)
    df_selected = df_preds[df_preds['customer_id'].isin(selected_customers)].copy()
    
    # Keep the order consistent with the selection
    df_selected['customer_id'] = pd.Categorical(df_selected['customer_id'], categories=selected_customers, ordered=True)
    df_selected = df_selected.sort_values('customer_id')
    
    probs = df_selected['predicted_probability'].tolist()
    
    print("\nSelected Customers and Probabilities:")
    for cust, p in zip(selected_customers, probs):
        print(f"{cust}: {p:.4f}")

    # 1. Exact & Monte Carlo classical baselines
    exact_prob = exact_probability_at_least_3(probs)
    mc_prob = monte_carlo_simulation(probs, shots=10000)
    
    # 2. Quantum Circuit Design[cite: 2]
    num_qubits = len(probs)
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # Prepare states: RY gate rotates qubit 0->1 based on probability
    # P(1) = sin^2(theta / 2)  =>  theta = 2 * arcsin(sqrt(P(1)))
    thetas = [2 * np.arcsin(np.sqrt(p)) for p in probs]
    
    for i, theta in enumerate(thetas):
        qc.ry(theta, i)
        
    qc.measure(range(num_qubits), range(num_qubits))
    
    # Generate circuit metrics for the report[cite: 3]
    backend = GenericBackendV2(num_qubits=num_qubits)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled_qc = pm.run(qc)
    
    print(f"\nCircuit Depth (Original): {qc.depth()}")
    print(f"Circuit Depth (Transpiled): {transpiled_qc.depth()}")
    print(f"Gate Counts: {dict(transpiled_qc.count_ops())}")

    # 3. Quantum Simulation Execution[cite: 3]
    print("\nRunning Quantum Simulation...")
    sampler = StatevectorSampler(seed=RANDOM_SEED)
    shots = 10000
    # Qiskit V2 Sampler usage
    job = sampler.run([qc], shots=shots)
    result = job.result()[0]
    
    # Extract bitstring counts
    counts = result.data.c.get_counts()
    
    # Calculate P(>=3 successes) from quantum measurements
    # Note: Qiskit bitstrings are right-to-left (q_5 q_4 q_3 q_2 q_1 q_0)
    quantum_successes = 0
    for bitstring, count in counts.items():
        if bitstring.count('1') >= 3:
            quantum_successes += count
            
    quantum_prob = quantum_successes / shots
    
    print("\n--- Final Campaign Outcomes ---")
    print(f"Exact Analytical Probability: {exact_prob:.4f}")
    print(f"Monte Carlo Probability:      {mc_prob:.4f}")
    print(f"Quantum Circuit Probability:  {quantum_prob:.4f}")
    print(f"Quantum Error vs Exact:       {abs(quantum_prob - exact_prob):.4f}")
    
    # 4. Save Final Result Deliverable[cite: 3]
    result_payload = {
        "selected_ids": selected_customers,
        "qubit_order": "q0 to q5 maps to selected_ids sequentially",
        "probabilities": probs,
        "circuit_parameters_thetas": thetas,
        "measurement_shots": shots,
        "quantum_estimated_probability": quantum_prob,
        "exact_analytical_result": exact_prob,
        "monte_carlo_benchmark": mc_prob,
        "absolute_quantum_error": abs(quantum_prob - exact_prob)
    }
    
    out_file = os.path.join(BASE_DIR, 'stage3_simulation.json')
    with open(out_file, 'w') as f:
        json.dump(result_payload, f, indent=4)
        
    print(f"\n[+] stage3_simulation.json successfully generated at {out_file}!")

if __name__ == "__main__":
    run_stage3()