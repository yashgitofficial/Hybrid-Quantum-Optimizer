# Hybrid Quantum Campaign Optimizer

## 🏆 Devpost Submission Details
- **Team Name:** DevQubix
- **Members:** Yash Pandey , Sritama Sengupta
- **Emails:** yash2006fighter@gmail.com,sritamasengupta415@gmail.com
- **Discord Handles:** yashbeastera56,sritama_2195
- **Presentation Deck:** https://docs.google.com/presentation/d/1dNshYkCc1sJpZj3JmSuCgTrFiA1UWUbv/edit?usp=drivesdk&rtpof=true&sd=true


---

## 1. Pipeline Architecture (Overview)
This project implements an end-to-end hybrid quantum-classical pipeline to optimize a bank's term deposit marketing campaign.
* **Stage 1 (Predict):** A classical Logistic Regression baseline and a Variational Quantum Classifier (VQC) were trained on customer features. Continuous probabilities were propagated forward to ensure data continuity.
* **Stage 2 (Select):** The top 10 candidates were formulated into a Quadratic Unconstrained Binary Optimization (QUBO) problem. An exact classical eigensolver (NumPyMinimumEigensolver) was used to maximize Expected Net Value (ENV) under strict budget and capacity constraints.
* **Stage 3 (Simulate):** The selected 6 customers were mapped to a 6-qubit quantum circuit. $RY$ gates were used to encode individual subscription probabilities, and a `StatevectorSampler` simulated the probability of achieving $\ge 3$ conversions.

---

## 2. Setup & Execution
### Prerequisites
- Python 3.13
- Qiskit 1.0.2

### Installation & Execution
```bash
git clone https://github.com/yashgitofficial/Hybrid-Quantum-Optimizer
cd Hybrid-Quantum-Optimizer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run the pipeline sequentially:
python src/stage1_predict.py
python src/stage2_select.py
python src/stage3_simulate.py
