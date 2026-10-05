!pip install -q tensorly

import itertools
import pandas as pd
import numpy as np
import torch
import tensorly as tl
from tensorly.decomposition import parafac, tucker
import matplotlib.pyplot as plt

tl.set_backend('pytorch')

# ---------- 1. Load ----------
from google.colab import files
up = files.upload()
df = pd.read_csv(next(iter(up)))

df['items'] = df['Difficulties'].str.split(',').apply(
    lambda L: sorted({x.strip() for x in L}))
nodes = sorted({d for L in df['items'] for d in L})
idx = {n: i for i, n in enumerate(nodes)}
S, N = len(df), len(nodes)

# ---------- 2. Build tensor: student x difficulty x difficulty ----------
A = torch.zeros(S, N, N)
for s, L in enumerate(df['items']):
    ids = [idx[d] for d in L]
    for i, j in itertools.combinations(ids, 2):   # complete graph (clique)
        A[s, i, j] = A[s, j, i] = 1.0
    for i in ids:                                 # self-loop
        A[s, i, i] = 1.0
print("Tensor shape:", tuple(A.shape))

# aggregate co-occurrence graph (sanity check)
plt.figure(figsize=(7, 6))
plt.imshow(A.sum(0).numpy(), cmap='viridis')
plt.xticks(range(N), nodes, rotation=90); plt.yticks(range(N), nodes)
plt.colorbar(); plt.title("Co-occurrence (sum of slices)")
plt.tight_layout(); plt.show()

# ---------- 3. Choose CP rank ----------
errs = []
for R in range(1, 9):
    cp = parafac(A, rank=R, init='random', n_iter_max=1000, tol=1e-8,
                 random_state=0, normalize_factors=True)
    e = (torch.norm(A - tl.cp_to_tensor(cp)) / torch.norm(A)).item()
    errs.append(e); print(f"R={R}  rel_err={e:.4f}")
plt.plot(range(1, 9), errs, 'o-'); plt.xlabel("rank"); plt.ylabel("rel. error")
plt.title("CP rank selection (elbow)"); plt.show()

# ---------- 4. CP / PARAFAC ----------
R = 3   # set from elbow
cp = parafac(A, rank=R, init='random', n_iter_max=1000, tol=1e-8,
             random_state=0, normalize_factors=True)
weights, (U_stu, U_d1, U_d2) = cp
print("Weights:", weights.numpy().round(3))

dif_f = pd.DataFrame(U_d1.numpy(), index=nodes,
                     columns=[f"C{r+1}" for r in range(R)])
print(dif_f.round(3))

plt.figure(figsize=(5, 6))
plt.imshow(dif_f.values, cmap='RdBu_r', aspect='auto')
plt.yticks(range(N), nodes); plt.xticks(range(R), dif_f.columns)
plt.colorbar(); plt.title("Difficulty loadings"); plt.tight_layout(); plt.show()

# student profiles: dominant component per student
stu_f = pd.DataFrame(U_stu.numpy(), columns=dif_f.columns)
stu_f['StudentID'] = df['StudentID'].values
stu_f['dominant'] = stu_f[dif_f.columns].abs().idxmax(axis=1)
print(stu_f['dominant'].value_counts())

# ---------- 5. Tucker (alternative) ----------
core, (T_stu, T_d1, T_d2) = tucker(A, rank=[4, 3, 3], init='svd',
                                   n_iter_max=500, tol=1e-8)
t_err = (torch.norm(A - tl.tucker_to_tensor((core, [T_stu, T_d1, T_d2])))
         / torch.norm(A)).item()
print("Tucker rel_err:", round(t_err, 4))
print(pd.DataFrame(T_d1.numpy(), index=nodes).round(3))
