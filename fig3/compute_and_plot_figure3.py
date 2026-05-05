import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.func import functional_call, grad, vmap
from sklearn.cluster import SpectralClustering, KMeans
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WMDD_DIR     = os.environ.get("WMDD_DIR",     f"{os.environ['SCRATCH']}/dntk/wmdd")
TRAIN_DIR    = os.environ.get("TRAIN_DIR",    f"{WMDD_DIR}/imagenette_ipc50")
VAL_DIR      = os.environ.get("VAL_DIR",
                              f"{os.environ['SCRATCH']}/dntk/data/imagenette2-160/val")
TEACHER_PATH = os.environ.get("TEACHER_PATH", f"{WMDD_DIR}/teacher_model/imagenette_resnet18.pth")
OUT_DIR      = os.environ.get("OUT_DIR",      f"{os.environ['SCRATCH']}/dntk/outputs")

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

N_TRAIN_PER_CLASS = 50
N_TRAIN_TOTAL     = 10 * N_TRAIN_PER_CLASS  # m = 500
K_PROJ            = 4096
CHUNK             = 100_000
SEED              = 42
BATCH             = 32
LAMBDA_REG        = 1e-4

# Sample-size sweep matching the paper's Figure 3 x-axis (5..40 gradients)
SAMPLE_SIZES = [5, 6, 8, 10, 13, 17, 22, 28, 35, 40]

# Number of repetitions for stochastic methods (paper averages across runs)
N_REPS = 3

SYNTHETIC_CONFIGS = [
    (tv, tg, H)
    for H  in [5, 10, 15, 20]
    for tv in [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    for tg in [0.70, 0.80, 0.90, 0.95, 0.99]
]

# ---------------------------------------------------------------------------
# Data, model, gradient computation (same logic as Figure 1)
# ---------------------------------------------------------------------------
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tfm = transforms.Compose([
    transforms.Resize(176), transforms.CenterCrop(160),
    transforms.ToTensor(), NORM,
])
train_ds = ImageFolder(TRAIN_DIR, transform=tfm)
val_ds   = ImageFolder(VAL_DIR,   transform=tfm)

train_idx_by_class = defaultdict(list)
for idx, (_, y) in enumerate(train_ds.samples):
    train_idx_by_class[y].append(idx)
train_idx_by_class = {c: train_idx_by_class[c][:N_TRAIN_PER_CLASS] for c in range(10)}
train_indices = sum([train_idx_by_class[c] for c in range(10)], [])
test_indices  = list(range(len(val_ds)))

def load_with_remapping(model, sd):
    try:
        model.load_state_dict(sd)
        return
    except RuntimeError:
        pass
    remapped = {k.replace("module.", "").replace("model.", "")
                 .replace("downsample.conv.", "downsample.0.")
                 .replace("downsample.norm.", "downsample.1."): v
                for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    missing    = [k for k in missing    if "num_batches_tracked" not in k]
    unexpected = [k for k in unexpected if "num_batches_tracked" not in k]
    if missing or unexpected:
        raise RuntimeError(f"unmapped keys: missing={missing}, unexpected={unexpected}")

print(f"Loading teacher from {TEACHER_PATH}")
ckpt = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)
load_with_remapping(model, sd)
model = model.to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)

params  = {k: v.detach().clone() for k, v in model.named_parameters()}
buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
sorted_param_names = sorted(params.keys())

def logit_c(p, b, x, c):
    out = functional_call(model, (p, b), x.unsqueeze(0))
    return out[0, c]

batched_per_sample_grad = vmap(grad(logit_c), in_dims=(None, None, 0, None))

def jl_project_batched(grad_dict):
    B = next(iter(grad_dict.values())).shape[0]
    proj = torch.zeros(B, K_PROJ, device=device)
    chunk_id = 0
    for name in sorted_param_names:
        g = grad_dict[name].detach().reshape(B, -1)
        n = g.shape[1]
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            gen = torch.Generator(device=device)
            gen.manual_seed(SEED * 1_000_003 + chunk_id)
            Q = torch.randn(end - start, K_PROJ, device=device, generator=gen)
            proj.add_(g[:, start:end] @ Q)
            chunk_id += 1
            del Q
    return proj / math.sqrt(K_PROJ)

def compute_phi_and_logits(dataset, indices, name,
                           _model=model, _params=params, _buffers=buffers,
                           _grad_fn=batched_per_sample_grad):
    n = len(indices)
    phi = torch.zeros(10, n, K_PROJ, device=device)
    logits = torch.zeros(n, 10, device=device)
    labels = torch.zeros(n, dtype=torch.long)
    xs = torch.zeros(n, 3, 160, 160, device=device)
    for i, idx in enumerate(indices):
        x, y = dataset[idx]
        xs[i] = x
        labels[i] = y
    with torch.no_grad():
        for start in range(0, n, 64):
            end = min(start + 64, n)
            logits[start:end] = _model(xs[start:end])
    for c in range(10):
        for start in tqdm(range(0, n, BATCH), desc=f"{name} logit {c}"):
            end = min(start + BATCH, n)
            g_dict = _grad_fn(_params, _buffers, xs[start:end], c)
            phi[c, start:end] = jl_project_batched(g_dict)
            del g_dict
    return phi.cpu(), logits.cpu(), labels

print("Computing Phi_train, teacher_logits_train (500 distilled images)...")
phi_train, y_logits_train, _ = compute_phi_and_logits(train_ds, train_indices, "train")
print(f"Computing Phi_test, teacher_logits_test ({len(test_indices)} val images)...")
phi_test, y_logits_test, y_labels_test = compute_phi_and_logits(val_ds, test_indices, "test")

del model, params, buffers
torch.cuda.empty_cache()

teacher_test_pred = y_logits_test.argmax(dim=1).numpy()
teacher_test_acc  = (teacher_test_pred == y_labels_test.numpy()).mean()
print(f"Teacher accuracy on test set: {teacher_test_acc:.4f}")

# ---------------------------------------------------------------------------
# KRR evaluation: given compressed (Phi_hat, y_hat) for each class, predict
# test logits and report fidelity / accuracy / MSE.
# ---------------------------------------------------------------------------
def evaluate_krr(Phi_hat_per_class, y_hat_per_class, lambda_reg=LAMBDA_REG):
    """Phi_hat_per_class: list of 10 tensors, each (s_c, K_PROJ)
       y_hat_per_class:   list of 10 tensors, each (s_c,)
       Returns fidelity, accuracy, mse."""
    n_test = phi_test.shape[1]
    krr_pred = torch.zeros(n_test, 10)
    for c in range(10):
        Phi_h = Phi_hat_per_class[c]                  # (s, K_PROJ)
        y_h   = y_hat_per_class[c]                    # (s,)
        Phi_te = phi_test[c, :, :]                    # (n_test, K_PROJ)
        K_tr = Phi_h @ Phi_h.T                        # (s, s)
        K_te = Phi_h @ Phi_te.T                       # (s, n_test)
        s = K_tr.shape[0]
        scale = K_tr.diagonal().mean().clamp(min=1e-12)
        K_tr_n = K_tr / scale
        K_te_n = K_te / scale
        I = torch.eye(s)
        try:
            alpha = torch.linalg.solve(K_tr_n + lambda_reg * I, y_h)
        except torch._C._LinAlgError:
            alpha = torch.linalg.lstsq(K_tr_n + lambda_reg * I, y_h).solution
        krr_pred[:, c] = K_te_n.T @ alpha
    pred = krr_pred.argmax(dim=1).numpy()
    fid = (pred == teacher_test_pred).mean()
    acc = (pred == y_labels_test.numpy()).mean()
    mse = ((krr_pred - y_logits_test) ** 2).mean().item()
    return fid, acc, mse

# ---------------------------------------------------------------------------
# Sampling methods: FPS, K-means, Leverage, Random
# Each returns 10 lists of selected indices into the m=500 training pool.
# ---------------------------------------------------------------------------
def select_random(s, seed):
    rng = np.random.default_rng(seed)
    return [rng.choice(N_TRAIN_TOTAL, size=s, replace=False).tolist() for _ in range(10)]

def select_fps(s, seed):
    """Farthest Point Sampling per class. Distances are L2 in projected
    gradient feature space."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(10):
        Phi = phi_train[c].numpy()                    # (m, K_PROJ)
        m = Phi.shape[0]
        idx0 = int(rng.integers(m))
        sel = [idx0]
        # min-distance from each point to the current selected set
        d2 = np.sum((Phi - Phi[idx0]) ** 2, axis=1)
        for _ in range(s - 1):
            j = int(np.argmax(d2))
            sel.append(j)
            d_new = np.sum((Phi - Phi[j]) ** 2, axis=1)
            d2 = np.minimum(d2, d_new)
        out.append(sel)
    return out

def select_kmeans_centroids(s, seed):
    """K-means on each class's gradients. Returns the s indices nearest the
    centroids (paper's interpretation: the actual cluster representative,
    not synthetic centroids in feature space, since we want a sampling method)."""
    out = []
    for c in range(10):
        Phi = phi_train[c].numpy()
        km = KMeans(n_clusters=s, random_state=seed, n_init=10).fit(Phi)
        centers = km.cluster_centers_                  # (s, K_PROJ)
        # For each centroid, find the data point closest to it
        sel = []
        for k in range(s):
            d2 = np.sum((Phi - centers[k]) ** 2, axis=1)
            j = int(np.argmin(d2))
            # Avoid duplicates: pick next-nearest if already picked
            order = np.argsort(d2)
            for cand in order:
                if int(cand) not in sel:
                    sel.append(int(cand))
                    break
            else:
                sel.append(j)
        out.append(sel)
    return out

def select_leverage(s, seed, lambda_reg=LAMBDA_REG):
    """Leverage score sampling for KRR. Leverage of point i is the i-th
    diagonal entry of K (K + lambda I)^{-1}, computed via eigendecomp."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(10):
        Phi = phi_train[c]
        K = Phi @ Phi.T                                # (m, m)
        scale = K.diagonal().mean().clamp(min=1e-12)
        K = K / scale
        eigvals, eigvecs = torch.linalg.eigh(K)
        # leverage_i = sum_j (eigvecs[i,j])^2 * eigvals[j] / (eigvals[j] + lambda)
        ratio = eigvals / (eigvals + lambda_reg)        # (m,)
        leverage = ((eigvecs ** 2) * ratio).sum(dim=1)  # (m,)
        leverage = torch.clamp(leverage, min=1e-12).numpy()
        probs = leverage / leverage.sum()
        sel = rng.choice(N_TRAIN_TOTAL, size=s, replace=False, p=probs).tolist()
        out.append(sel)
    return out

def materialize_from_indices(indices_per_class):
    """Convert per-class index selections into (Phi_hat, y_hat) pairs.
    indices_per_class[c] is a list of s indices into [0, m)."""
    Phi_hat_per_class = []
    y_hat_per_class   = []
    for c in range(10):
        idx = indices_per_class[c]
        Phi_hat = phi_train[c, idx, :]                  # (s, K_PROJ)
        y_hat   = y_logits_train[idx, c]                # (s,)
        Phi_hat_per_class.append(Phi_hat)
        y_hat_per_class.append(y_hat)
    return Phi_hat_per_class, y_hat_per_class

# ---------------------------------------------------------------------------
# Synthetic method: Algorithm 1 (local-global gradient distillation)
# ---------------------------------------------------------------------------
def synthesize_gradients(tau_v, tau_g, H):
    """Run Algorithm 1 on the projected gradient features. Returns
    (Phi_hat_per_class, y_hat_per_class, s_actual) where s_actual is the
    number of synthetic gradients produced (varies with config)."""
    # Step 1: per-class kernels and class-averaged kernel
    K_per_class = torch.stack([phi_train[c] @ phi_train[c].T for c in range(10)])  # (10, m, m)
    K_bar = K_per_class.mean(dim=0).numpy()             # (m, m)

    # Step 1b: spectral clustering on K_bar (paper uses it as adjacency)
    # Make symmetric and clip to non-negative for SpectralClustering
    A = np.maximum(K_bar, 0)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    sc = SpectralClustering(n_clusters=H, affinity="precomputed",
                            random_state=SEED, assign_labels="kmeans")
    cluster_labels = sc.fit_predict(A)
    clusters = [np.where(cluster_labels == h)[0] for h in range(H)]

    # Step 2: global eigendecomposition of K_bar
    eigvals_g, eigvecs_g = np.linalg.eigh(K_bar)
    eigvals_g = eigvals_g[::-1]                          # descending
    eigvecs_g = eigvecs_g[:, ::-1]                       # match
    eigvals_g = np.maximum(eigvals_g, 0)

    cumvar = np.cumsum(eigvals_g) / eigvals_g.sum()
    r_g = int(np.searchsorted(cumvar, tau_v) + 1)
    r_g = min(r_g, len(eigvals_g))

    # Step 3: per-cluster eigendecomp + coverage analysis
    coverage = np.zeros(r_g)
    local_eigvecs_per_cluster = []                       # for step 4
    local_ranks = []
    for h, Ih in enumerate(clusters):
        if len(Ih) < 2:
            local_eigvecs_per_cluster.append(np.zeros((len(Ih), 0)))
            local_ranks.append(0)
            continue
        Kh = K_bar[np.ix_(Ih, Ih)]
        eh, vh = np.linalg.eigh(Kh)
        eh = eh[::-1]
        vh = vh[:, ::-1]
        eh = np.maximum(eh, 0)
        rh = int(np.searchsorted(np.cumsum(eh) / max(eh.sum(), 1e-12), tau_v) + 1)
        rh = min(rh, len(eh))
        Uh_top = vh[:, :rh]                              # (|Ih|, rh)
        local_eigvecs_per_cluster.append(Uh_top)
        local_ranks.append(rh)
        # Coverage: how much of each global eigvec lies in span(local top)
        for j in range(r_g):
            u_global = eigvecs_g[Ih, j]                  # restrict to cluster
            norm_u = np.linalg.norm(u_global)
            if norm_u < 1e-12:
                continue
            proj = Uh_top @ (Uh_top.T @ u_global)
            coverage_jh = (np.linalg.norm(proj) / norm_u) ** 2
            coverage[j] = max(coverage[j], coverage_jh)

    gap_directions = np.where(coverage < tau_g)[0]

    # Steps 4-5: synthesize gradients from local eigvecs and gap directions
    Phi_hat_per_class = [[] for _ in range(10)]
    y_hat_per_class   = [[] for _ in range(10)]
    full_eigvecs      = []  # (synthesizer eigenvector lifted to full m-dim space)

    # Local representatives: one per (cluster, local eigenvector)
    for h, Ih in enumerate(clusters):
        Uh_top = local_eigvecs_per_cluster[h]
        for j in range(local_ranks[h]):
            u_local = Uh_top[:, j]
            u_full  = np.zeros(N_TRAIN_TOTAL)
            u_full[Ih] = u_local
            for c in range(10):
                Phi_hat_per_class[c].append(phi_train[c].numpy().T @ u_full)
                y_hat_per_class[c].append(y_logits_train[:, c].numpy() @ u_full)
            full_eigvecs.append(u_full)

    # Gap representatives: full global eigenvectors
    for j in gap_directions:
        u_full = eigvecs_g[:, j]
        for c in range(10):
            Phi_hat_per_class[c].append(phi_train[c].numpy().T @ u_full)
            y_hat_per_class[c].append(y_logits_train[:, c].numpy() @ u_full)
        full_eigvecs.append(u_full)

    # Step 6: QR-based redundancy removal on the eigenvector set
    if len(full_eigvecs) == 0:
        return None, None, 0
    U = np.stack(full_eigvecs, axis=1)                    # (m, total_synth)
    Q, R = np.linalg.qr(U)
    diag_R = np.abs(np.diag(R))
    if diag_R.size == 0:
        return None, None, 0
    threshold = 1e-6 * diag_R.max()
    keep = diag_R > threshold
    keep_idx = np.where(keep)[0].tolist()

    Phi_hat_final = []
    y_hat_final   = []
    for c in range(10):
        Phi_hat_final.append(torch.tensor(np.array(Phi_hat_per_class[c])[keep_idx],
                                          dtype=torch.float32))
        y_hat_final.append(torch.tensor(np.array(y_hat_per_class[c])[keep_idx],
                                        dtype=torch.float32))
    return Phi_hat_final, y_hat_final, int(keep.sum())

# ---------------------------------------------------------------------------
# Run all methods and collect results
# ---------------------------------------------------------------------------
results = {
    "FPS":       {"s": [], "fid": [], "acc": [], "mse": []},
    "K-means":   {"s": [], "fid": [], "acc": [], "mse": []},
    "Leverage":  {"s": [], "fid": [], "acc": [], "mse": []},
    "Random":    {"s": [], "fid": [], "acc": [], "mse": []},
    "Synthetic": {"s": [], "fid": [], "acc": [], "mse": []},
}

print("\nRunning sampling methods (averaged over %d reps where stochastic)..." % N_REPS)
for s in SAMPLE_SIZES:
    for method, select_fn in [
        ("FPS",      select_fps),
        ("K-means",  select_kmeans_centroids),
        ("Leverage", select_leverage),
        ("Random",   select_random),
    ]:
        fids, accs, mses = [], [], []
        for rep in range(N_REPS):
            sel = select_fn(s, seed=SEED + rep)
            Phi_hat, y_hat = materialize_from_indices(sel)
            fid, acc, mse = evaluate_krr(Phi_hat, y_hat)
            fids.append(fid); accs.append(acc); mses.append(mse)
        results[method]["s"].append(s)
        results[method]["fid"].append(np.mean(fids))
        results[method]["acc"].append(np.mean(accs))
        results[method]["mse"].append(np.mean(mses))
        print(f"  s={s:3d}  {method:<10s}  fid={np.mean(fids):.3f}  "
              f"acc={np.mean(accs):.3f}  mse={np.mean(mses):.3f}")

print(f"\nRunning Synthetic (Algorithm 1) over {len(SYNTHETIC_CONFIGS)} configs...")
synthetic_runs = []
for tv, tg, H in tqdm(SYNTHETIC_CONFIGS, desc="synthetic configs"):
    Phi_hat, y_hat, s_actual = synthesize_gradients(tv, tg, H)
    if s_actual == 0 or s_actual > N_TRAIN_TOTAL:
        continue
    fid, acc, mse = evaluate_krr(Phi_hat, y_hat)
    synthetic_runs.append({"s": s_actual, "fid": fid, "acc": acc, "mse": mse,
                           "tv": tv, "tg": tg, "H": H})

# Pareto frontier in (s, accuracy) space: for each unique s, keep best accuracy
synth_by_s = defaultdict(list)
for r in synthetic_runs:
    synth_by_s[r["s"]].append(r)
synth_pareto = []
for s_val, runs in sorted(synth_by_s.items()):
    best = max(runs, key=lambda r: r["acc"])
    synth_pareto.append(best)

# Filter Synthetic results to roughly the SAMPLE_SIZES range and downsample for plotting
synth_pareto = [r for r in synth_pareto
                if SAMPLE_SIZES[0] <= r["s"] <= SAMPLE_SIZES[-1] * 2]
results["Synthetic"]["s"]   = [r["s"]   for r in synth_pareto]
results["Synthetic"]["fid"] = [r["fid"] for r in synth_pareto]
results["Synthetic"]["acc"] = [r["acc"] for r in synth_pareto]
results["Synthetic"]["mse"] = [r["mse"] for r in synth_pareto]
print(f"Synthetic: {len(synth_pareto)} Pareto-optimal configs in plotted range")

# Save raw results
torch.save({
    "results": results,
    "synthetic_runs": synthetic_runs,
    "teacher_test_acc": teacher_test_acc,
    "lambda_reg": LAMBDA_REG,
    "k_proj": K_PROJ,
}, f"{OUT_DIR}/figure3_results.pt")

# ---------------------------------------------------------------------------
# Plot four panels matching Figure 3
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 9))

method_styles = {
    "FPS":       {"color": "C0", "marker": "o", "ls": "--"},
    "K-means":   {"color": "C1", "marker": "s", "ls": "--"},
    "Leverage":  {"color": "C2", "marker": "^", "ls": "--"},
    "Random":    {"color": "C3", "marker": "v", "ls": "--"},
    "Synthetic": {"color": "C4", "marker": "D", "ls": "-",  "lw": 2},
}

# Panel 1: Fidelity vs Sample Size
ax = axes[0, 0]
for method, r in results.items():
    style = method_styles[method]
    ax.plot(r["s"], 100 * np.array(r["fid"]),
            label=method, **{k: v for k, v in style.items()})
ax.set_xscale("log")
ax.set_xlabel("Number of Gradients")
ax.set_ylabel("Fidelity (%)")
ax.set_title("Fidelity vs Sample Size")
ax.grid(alpha=0.3)
ax.legend()

# Panel 2: Accuracy vs Sample Size
ax = axes[0, 1]
for method, r in results.items():
    style = method_styles[method]
    ax.plot(r["s"], 100 * np.array(r["acc"]),
            label=method, **{k: v for k, v in style.items()})
ax.axhline(100 * teacher_test_acc, color="k", ls=":", lw=0.8,
           label=f"Baseline ({teacher_test_acc*100:.1f}%)")
ax.set_xscale("log")
ax.set_xlabel("Number of Gradients")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy vs Sample Size")
ax.grid(alpha=0.3)
ax.legend()

# Panel 3: MSE vs Sample Size
ax = axes[1, 0]
for method, r in results.items():
    style = method_styles[method]
    ax.semilogy(r["s"], r["mse"],
                label=method, **{k: v for k, v in style.items()})
ax.set_xscale("log")
ax.set_xlabel("Number of Gradients")
ax.set_ylabel("MSE")
ax.set_title("MSE vs Sample Size")
ax.grid(alpha=0.3)
ax.legend()

# Panel 4: Accuracy vs Compression Ratio (m / s)
ax = axes[1, 1]
for method, r in results.items():
    style = method_styles[method]
    cr = N_TRAIN_TOTAL / np.array(r["s"])
    ax.plot(cr, 100 * np.array(r["acc"]),
            label=method, **{k: v for k, v in style.items()})
ax.axhline(100 * teacher_test_acc, color="k", ls=":", lw=0.8,
           label=f"Baseline ({teacher_test_acc*100:.1f}%)")
ax.set_xlabel("Compression Ratio  (m / s)")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy vs Compression Ratio")
ax.grid(alpha=0.3)
ax.legend()

plt.suptitle(
    "Figure 3 reproduction: per-class NTK KRR with sampling vs synthesis methods, "
    "ResNet-18 / WMDD-distilled ImageNette",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/figure3_reproduction.png", dpi=150)
plt.savefig(f"{OUT_DIR}/figure3_reproduction.pdf")
print(f"\nSaved figure: {OUT_DIR}/figure3_reproduction.png")