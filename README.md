# LP Optimization: Sifting to Overcome CPLEX Community Edition Limits

Solving a **5000 variable, 500 constraint** linear program with the free/community version of CPLEX - which caps problems at **1000 variables**, by implementing the **sifting algorithm**, a simplex-based column-generation method described in Bixby et al., *Very Large-Scale Linear Programming: A Case Study in Combining Interior Point and Simplex Methods* (1992).

Instead of solving the full problem at once (impossible under the 1000-variable cap), sifting iteratively works with a small "active" subset of columns, uses dual prices to identify which of the excluded columns actually want to enter the solution, adds the best candidates, and repeats until no improving column remains outside the active set. What's left is probably optimal for the original 5000 variable problem, despite never having solved more than a few hundred variables at a time.

## Results

| Metric | Value |
|---|---|
| Decision variables (full problem) | 5,000 |
| Constraints | 500 |
| CPLEX Community Edition variable limit | 1,000 |
| Rank of constraint matrix A | 500 (full row rank - all constraints independent) |
| Initial working set size | 5000 columns |
| **Final working set size** | **896 columns** |
| Columns eliminated from consideration | 4,104 (**82.1%** of the problem never needed solving) |
| Sifting iterations to convergence | 8 (set size stabilized by iteration 4) |
| Columns added per iteration | ≤100 (lambda pricing) |
| **Final objective value** | **1,515,714.6139719822** |
| Feasibility check tolerance | 1e-9 |
| Constraint violations found | **0 / 500** - solution verified feasible |

**Bottom line:** the sifting procedure solved a 5000-variable LP that the solver itself is incapable of handling directly, converged in 8 iterations to a working set of just 896 columns (well under the 1000-variable ceiling), and every one of the 500 constraints checks out to within 1e-9 — effectively an exact match to the true optimum.

## How it works

The problem:

```
Maximize    c^T x
Subject to  A x <= b
```

with dual:

```
Minimize    b^T y
Subject to  A^T y >= c
```

1. **Initialize**  start with a feasible working set W of columns (here, the first 500, chosen because they were verified to yield a feasible solution).
2. **Solve restricted primal/dual** solve Maximize c_W^T x_W, s.t. A_W x_W <= b` using CPLEX (via docplex) to get x_W* and the dual prices pi*.
3. **Price out excluded columns** for every column `j` not in `W`, check whether `pi*^T A_j < c_j`. Columns that fail this dual-feasibility condition are candidates to enter the working set.
4. **Select and add** rank candidates using the **lambda pricing rule** (chosen per the reference paper for faster convergence) and add up to t = 100 of the most attractive columns to W.
5. **Repeat** until no candidate columns remain at that point W supports a solution that is optimal for the *entire* original problem, not just the subset.
6. **Reconstruct and verify** set all variables outside the final W to 0, compute the objective value, and re-check every constraint against the full A, b (with a 1e-9 tolerance for floating-point error) to confirm feasibility.

## Implementation notes

- **Data extraction**: the raw .lp file is parsed manually into `A`, `b`, `c` matrices (500 constraints × 5000 variables). One missing decision variable (`x_513`) in the source file was assumed to have zero cost.
- **Vectorization**: plain matrix multiplication was used rather than fast/sparse vectorization - the problem size made this fast enough without added complexity.
- **Pricing rule**: lambda pricing was used over alternatives, per the reference paper's guidance on convergence speed.
- **Batch size t**: tested at different values, larger t converges in fewer iterations but retains more columns in the final set; smaller t takes more iterations but keeps the working set leaner. t = 100 was used as a middle ground; the objective value was unaffected by this choice.
- **Solvers used**: docplex with IBM CPLEX for both the primal and dual sub-problems at each iteration.

## Requirements

- Python 3
- numpy
- docplex (with a working CPLEX installation/license - Community Edition is sufficient since the working set never exceeds ~900 variables)

## Repository contents

| File | Description |
|---|---|
| Code.py | Full implementation - data parsing, sifting loop, and feasibility checker |
| Approach.docx | Write-up of the approach and design decisions |
| output.txt | Full run log: iteration by iteration set growth, final solution vector, objective value, and feasibility check |
| Bixby-LargeScaleLinearProgramming-1992.pdf | Reference paper the sifting algorithm is adapted from |

## Running it

```bash
python Code.py
```

Expects an input file `test1.lp` (500 constraints, 5000 variables) in the working directory. Writes a full log — including the iteration trace, final solution vector, and feasibility check — to `output.txt`.

## Reference

R. E. Bixby et al., *"Very Large-Scale Linear Programming: A Case Study in Combining Interior Point and Simplex Methods,"* Operations Research, 1992.
