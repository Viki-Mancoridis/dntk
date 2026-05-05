import os, math, torch, numpy as np
from collections import defaultdict
from torchvision import transforms, models, datasets
from torch.utils.data import Subset
from torch.func import functional_call, grad
from tqdm import tqdm

DATA_DIR = os.environ.get("DATA_DIR", f"{os.environ['SCRATCH']}/dntk/data/imagenette2-160")
OUT_DIR  = os.environ.get("OUT_DIR",  f"{os.environ['SCRATCH']}/dntk/outputs")
device   = "cuda" if torch.cuda.is_available() else "cpu"

N_PER_CLASS = 50    # n = 500 total
K_PROJ      = 4096  # 2048 # JL target dimension
CHUNK       = 100_000
SEED        = 42

# --- model
ckpt = torch.load(f"{OUT_DIR}/resnet18_imagenette.pt", map_location=device)
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)  # we'll feed params through functional_call instead

# --- 50 samples per class from the val set (deterministic)
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
val_tfm = transforms.Compose([
    transforms.Resize(176), transforms.CenterCrop(160),
    transforms.ToTensor(), NORM,
])
val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_tfm)
by_class = defaultdict(list)
for i, (_, y) in enumerate(val_ds.samples):
    by_class[y].append(i)

rng = np.random.default_rng(SEED)
selected = []
for c in sorted(by_class):
    selected += rng.choice(by_class[c], size=N_PER_CLASS, replace=False).tolist()
subset = Subset(val_ds, selected)

# --- functional_call setup (per-sample gradients via torch.func)
params  = {k: v.detach() for k, v in model.named_parameters()}
buffers = {k: v.detach() for k, v in model.named_buffers()}

def logit_c(p, b, x, c):
    out = functional_call(model, (p, b), x.unsqueeze(0))
    return out[0, c]

per_sample_grad = grad(logit_c)  # gradient w.r.t. argument 0 = params dict

# --- chunked, seeded JL projection: ϕ̃ = Q^T ϕ / sqrt(k)
# Q is implicit; we redraw chunks deterministically so the SAME Q is used per sample.
def jl_project(grad_dict, k=K_PROJ, base_seed=SEED, chunk=CHUNK):
    proj = torch.zeros(k, device=device)
    chunk_id = 0
    for name in sorted(grad_dict.keys()):
        g = grad_dict[name].detach().flatten()
        n = g.numel()
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            gen = torch.Generator(device=device)
            gen.manual_seed(base_seed * 1_000_003 + chunk_id)
            Q = torch.randn(end - start, k, device=device, generator=gen)
            proj.add_(g[start:end] @ Q)
            chunk_id += 1
    return proj / math.sqrt(k)

# --- compute Φ^c for c = 0..9
N = len(subset)
xs = torch.stack([subset[i][0] for i in range(N)]).to(device)

phi = torch.zeros(10, N, K_PROJ, device=device)
for c in range(10):
    for i in tqdm(range(N), desc=f"logit {c}"):
        g_dict = per_sample_grad(params, buffers, xs[i], c)
        phi[c, i] = jl_project(g_dict)
        del g_dict

# --- per-class kernels: K^c = Φ^c (Φ^c)^T ∈ R^{N×N}
K_per_class = torch.einsum("cnk,cmk->cnm", phi, phi).cpu()

torch.save({
    "K_per_class": K_per_class,
    "selected_indices": selected,
    "n_per_class": N_PER_CLASS,
    "k_proj": K_PROJ,
    "class_to_idx": ckpt["class_to_idx"],
}, f"{OUT_DIR}/per_class_kernels.pt")
print(f"saved K_per_class with shape {tuple(K_per_class.shape)}")