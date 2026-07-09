import math
import torch
import torch.nn.functional as F

MAX_STEPS = 


def train_epoch(model, dataloader, optimizer, scheduler, device, load_balance_weight=0.01, max_steps=None):
    model.train()
    total_loss = 0
    for i, (inputs, targets) in enumerate(dataloader):
        if max_steps is not None and i >= max_steps:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits, aux_loss = model(inputs, training=True)
        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss = ce_loss + load_balance_weight * aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += ce_loss.item()
    steps = min(max_steps, len(dataloader)) if max_steps else len(dataloader)
    return total_loss / steps


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits, _ = model(inputs, training=False)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


def train_model(model, train_loader, val_loader, device, num_epochs=20,
                lr=1e-3, load_balance_weight=0.01, model_name="model", max_steps=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
 
    total_steps = num_epochs * len(train_loader)
    warmup_steps = len(train_loader)  # 1-epoch linear warmup
 
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}
    best_ppl = float('inf')
 
    patience = 3
    no_improve = 0
 
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, load_balance_weight, max_steps=max_steps)
        val_loss, val_ppl = evaluate(model, val_loader, device)
 
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
 
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            no_improve = 0
            torch.save(model.state_dict(), f'best_{model_name}.pt')
        else:
            no_improve += 1
 
        print(f"  [{model_name}] Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val PPL: {val_ppl:.2f}")
 
        if no_improve >= patience:
            print(f"  [{model_name}] Early stop at epoch {epoch+1} "
                  f"(no improvement for {patience} epochs)")
            break
 
    model.load_state_dict(torch.load(f'best_{model_name}.pt', map_location=device))
    print(f"  [{model_name}] Best Val PPL: {best_ppl:.2f}")
    return history, best_ppl
