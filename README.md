# 📘 Thesis Defence Scheduling – MOMILP Implementation 

Link to paper : https://doi.org/10.1016/j.ejor.2023.06.031

This repository contains  **paper implementation** of the two-stage multi-objective MILP model for thesis defence scheduling, including:

- **Stage 1**: Maximization of scheduled defences (`g*`)
- **Stage 2**:
  - Algorithm 1 → Ideal and Approximate Nadir computation
  - Algorithm 5 → Augmented ε-constraint enumeration
- Scalability experiments
- Markov-based availability generator (Algorithm 6 from paper)
- Full CSV export of results


---

# 📂 Project Structure

```
src/
 ├── common/
 │   ├── symbols.py
 │   ├── parameters.py
 │   ├── bounds.py
 │   ├── dominance.py
 │   └── solve_tracker.py
 │
 ├── model/
 │   ├── variables.py
 │   ├── constraints.py
 │   ├── objectives.py
 │   ├── zexpr.py
 │   └── build.py
 │
 ├── algorithms/
 │   ├── stage1_g.py
 │   ├── ideal_nadir.py
 │   └── augmented_epsilon.py
 │
 ├── instance_generator/
 │   ├── config.py
 │   ├── availability.py
 │   ├── presets.py
 │   ├── generator.py
 │   └── io.py
 │
 ├── experiments/
 │   └── csv_export.py
 │
 └── run/
     └── main.py

test.py
scalability.py
```

---



# ⚙️ How To Run

---

# ✅ 1️⃣ FAST SANITY TEST

This verifies the entire pipeline works end-to-end on a small instance.

## Command

```bash
python3 test.py
```

## Optional parameters

```bash
python3 test.py \
    --seed 1 \
    --steps 1 \
    --tl_stage1 5 \
    --tl_ideal 5 \
    --tl_eps 5
```

## What It Does

- Generates a small instance (`n_i=8`, `n_j=6`)
- Runs:
  - Stage 1
  - Ideal/Nadir
  - Augmented ε
- Prints:
  - `g*`
  - `|N|`
  - `|I|`
  - skip counts
  - time spent
  - `z_ideal`
  - `z_nadir`
  - Non-dominated solutions

## Output File

```
data/generated/test_run.csv
```

---

# 📊 2️⃣ SCALABILITY EXPERIMENTS (Tables C.1, C.2, C.3)

This reproduces the paper computational tables.

---

## 🔹 PAPER IMPLIMENTATION

Implements Section 6.1.2 exactly:

- Stage 1 → 30 minutes
- Algorithm 1 → 2 hours total
- Algorithm 5 → 12 hours TOTAL (dynamic per-iteration division)

### Example

```bash
python3 scalability.py \
    --steps 5 \
    --tl_stage1 1800 \
    --tl_ideal 7200 \
    --budget_eps 43200 \
    --seed_start 1
```

Where:

- `1800` = 30 minutes
- `7200` = 2 hours
- `43200` = 12 hours

Algorithm 5 will automatically divide the remaining budget across iterations.

---

## 🔹 FAST MODE (Testing Only)

Fixed time per ε-iteration:

```bash
python3 scalability.py \
    --steps 3 \
    --tl_stage1 60 \
    --tl_ideal 60 \
    --tl_eps 120 \
    --seed_start 1
```

⚠️ add --save_instances to save the instances and look at them but it takes more time 

## 🔹 One I Used 

Fixed time per ε-iteration:

```bash
python3 scalability.py \
    --steps 3 \
    --tl_stage1 300 \
    --tl_ideal 300 \
    --tl_eps 300 \
    --seed_start 1
```


---

# 📁 Generated Output

After running `scalability.py`:

```
data/results/
 ├── table_C1.csv
 ├── table_C2.csv
 └── table_C3.csv
```

Each CSV contains:

| Column | Meaning |
|--------|----------|
| N | Instance number |
| p(...) | Size tuple |
| d | Duration |
| u_i | Member weights |
| e_ijt | Fixed roles |
| c_i | Max committees |
| lik | Availability distribution |
| mkp | Room availability |
| v_i | Compactness vector probabilities |
| h_i | Room-change vector probabilities |
| r_iq | Subjects per member |
| t_iq | Subjects per defence |
| \|N\| | Non-dominated solutions |
| \|I\| | Infeasible ε |
| skip^N | Dominance skips |
| skip^I | Infeasibility skips |
| time^N | Time spent on feasible solves |
| time^I | Time spent on infeasible solves |
| g | g* |
| CPU(seconds) | Total runtime |

---




# 📈 Expected Results

## For test mode

- Small `|N|`
- Fast runtime (< 10 seconds)
- Valid ideal and nadir vectors

## For scalability

- Table C.1 → small instances
- Table C.2 → medium
- Table C.3 → large
- Larger CPU times
- `|N|` and `|I|` similar scale to paper

---



# 🏁 Complete Pipeline Summary

```
test.py
    ↓
Stage 1 (maximize g)
    ↓
Algorithm 1 (ideal + nadir)
    ↓
Algorithm 5 (augmented ε)
    ↓
CSV export

scalability.py
    ↓
Tables C.1 – C.3 reproduction
```

---

