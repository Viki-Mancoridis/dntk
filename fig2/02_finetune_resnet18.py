import os, csv, torch
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader

DATA_DIR = os.environ.get("DATA_DIR", f"{os.environ['SCRATCH']}/dntk/data/imagenette2-160")
OUT_DIR  = os.environ.get("OUT_DIR",  f"{os.environ['SCRATCH']}/dntk/outputs")
device   = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)

NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(160), transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), NORM,
])
val_tfm = transforms.Compose([
    transforms.Resize(176), transforms.CenterCrop(160),
    transforms.ToTensor(), NORM,
])

train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_tfm)
val_ds   = datasets.ImageFolder(f"{DATA_DIR}/val",   transform=val_tfm)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model = model.to(device)

opt     = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()
EPOCHS  = 8

history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

# track best-val checkpoint (selected by val loss, the cleaner signal of the two)
best_val_loss  = float("inf")
best_epoch     = -1
best_val_acc   = None
best_state_dict = None

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum, train_n = 0.0, 0
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        # sample-weighted running loss so the average isn't biased by a small last batch
        bs = y.numel()
        train_loss_sum += loss.item() * bs
        train_n        += bs
    train_loss = train_loss_sum / train_n

    model.eval()
    val_loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss_sum += loss_fn(logits, y).item() * y.numel()
            correct      += (logits.argmax(1) == y).sum().item()
            total        += y.numel()
    val_loss = val_loss_sum / total
    val_acc  = correct / total

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # snapshot best-val state on CPU (so we don't accumulate GPU memory)
    if val_loss < best_val_loss:
        best_val_loss   = val_loss
        best_val_acc    = val_acc
        best_epoch      = epoch
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        marker = "  <-- best so far"
    else:
        marker = ""
    print(f"epoch {epoch}: train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{marker}", flush=True)

print(f"best epoch: {best_epoch} (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

# save the best-val checkpoint as the main one consumed by step 3
torch.save({"state_dict": best_state_dict,
            "class_to_idx": train_ds.class_to_idx,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc},
           f"{OUT_DIR}/resnet18_imagenette.pt")

# also keep the final-epoch checkpoint for comparison
torch.save({"state_dict": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "history": history},
           f"{OUT_DIR}/resnet18_imagenette_final_epoch.pt")

# --- save metrics as CSV for later replotting without rerunning training
with open(f"{OUT_DIR}/finetune_history.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
    for e, tl, vl, va in zip(history["epoch"], history["train_loss"],
                             history["val_loss"], history["val_acc"]):
        w.writerow([e, tl, vl, va])

# --- training curves: losses on the left, val accuracy on the right
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

ax_loss.plot(history["epoch"], history["train_loss"], "o-", label="train")
ax_loss.plot(history["epoch"], history["val_loss"],   "o-", label="val")
ax_loss.axvline(best_epoch, color="k", ls=":", lw=0.8, alpha=0.6,
                label=f"best (epoch {best_epoch})")
ax_loss.set_xlabel("epoch")
ax_loss.set_ylabel("cross-entropy loss")
ax_loss.set_title("Loss")
ax_loss.legend()
ax_loss.grid(alpha=0.3)

ax_acc.plot(history["epoch"], history["val_acc"], "o-", color="C2")
ax_acc.axvline(best_epoch, color="k", ls=":", lw=0.8, alpha=0.6)
ax_acc.set_xlabel("epoch")
ax_acc.set_ylabel("val accuracy")
ax_acc.set_title("Val accuracy")
ax_acc.set_ylim(0, 1)
ax_acc.grid(alpha=0.3)
# annotate the saved (best-val) point
ax_acc.annotate(f"saved: {best_val_acc:.3f}",
                xy=(best_epoch, best_val_acc),
                xytext=(8, -15), textcoords="offset points", fontsize=9)

plt.suptitle(f"ResNet-18 fine-tuning on ImageNette  ({EPOCHS} epochs, AdamW lr=3e-4)",
             fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/finetune_curves.png", dpi=150)
plt.savefig(f"{OUT_DIR}/finetune_curves.pdf")
print(f"saved curves to {OUT_DIR}/finetune_curves.{{png,pdf}}")