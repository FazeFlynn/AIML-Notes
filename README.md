# AIML-Note


### Saving Checkpoints to Resume Training the other day:
```python
# =========================
# CELL 1: SETUP + CONFIG + AUTO-RESUME BOOTSTRAP
# =========================

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from google.colab import drive

# -------------------------
# 1) Mount persistent storage
# -------------------------
drive.mount('/content/drive')

# -------------------------
# 2) Reproducibility + device
# -------------------------
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# -------------------------
# 3) Config (industry style)
# -------------------------
CONFIG = {
    'input_dim': 128,
    'hidden_dim': 256,
    'output_dim': 10,
    'batch_size': 64,
    'lr': 3e-4,
    'epochs': 10,
    'save_every_steps': 200,
    'checkpoint_dir': '/content/drive/MyDrive/training_checkpoints',
    'checkpoint_name': 'latest_checkpoint.pt',
}

# -------------------------
# 4) Model definition
# -------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# 5) Dummy dataset (replace with real one)
# -------------------------
X = torch.randn(5000, CONFIG['input_dim'])
y = torch.randint(0, CONFIG['output_dim'], (5000,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

# -------------------------
# 6) Build training objects
# -------------------------
model = SimpleNet(
    CONFIG['input_dim'],
    CONFIG['hidden_dim'],
    CONFIG['output_dim']
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'])
criterion = nn.CrossEntropyLoss()

# optional but industry standard
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CONFIG['epochs'] * len(loader)
)

# -------------------------
# 7) Checkpoint paths
# -------------------------
checkpoint_dir = Path(CONFIG['checkpoint_dir'])
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = checkpoint_dir / CONFIG['checkpoint_name']

# -------------------------
# 8) Auto-resume state
# -------------------------
start_epoch = 0
global_step = 0
best_loss = float('inf')

if checkpoint_path.exists():
    print('Checkpoint found. Resuming training...')
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])

    start_epoch = ckpt['epoch'] + 1
    global_step = ckpt['global_step']
    best_loss = ckpt['best_loss']

    print(f'Resumed from epoch={start_epoch}, step={global_step}')
else:
    print('No checkpoint found. Starting fresh training.')


# =========================
# CELL 2: TRAINING LOOP + SAFE CHECKPOINTING
# =========================

for epoch in range(start_epoch, CONFIG['epochs']):
    model.train()
    running_loss = 0.0
    epoch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        global_step += 1

        # -------------------------
        # periodic checkpoint save
        # -------------------------
        if global_step % CONFIG['save_every_steps'] == 0:
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'best_loss': best_loss,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'config': CONFIG,
            }

            torch.save(checkpoint, checkpoint_path)
            print(f'[Checkpoint Saved] step={global_step}')

    avg_loss = running_loss / len(loader)
    epoch_time = time.time() - epoch_start

    print(
        f'Epoch [{epoch + 1}/{CONFIG["epochs"]}] | '
        f'Loss: {avg_loss:.4f} | '
        f'Time: {epoch_time:.2f}s'
    )

    # -------------------------
    # best-model style checkpointing
    # -------------------------
    if avg_loss < best_loss:
        best_loss = avg_loss

        best_checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'best_loss': best_loss,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'config': CONFIG,
        }

        torch.save(best_checkpoint, checkpoint_dir / 'best_model.pt')
        print(f'[Best Model Updated] loss={best_loss:.4f}')

# -------------------------
# final save (always do this)
# -------------------------
final_checkpoint = {
    'epoch': CONFIG['epochs'] - 1,
    'global_step': global_step,
    'best_loss': best_loss,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'config': CONFIG,
}

torch.save(final_checkpoint, checkpoint_dir / 'final_model.pt')
print('Training complete. Final model saved.')

```
