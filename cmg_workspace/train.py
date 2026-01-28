import torch
import time
import os
from module.cmg import CMG
from cmg_trainer import CMGTrainer
from dataloader.dataloader import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta


#TODO check normalization of motion range.
#TODO 

# Config
DATA_PATH = "dataloader/cmg_training_data.pt"
BATCH_SIZE = 256
NUM_EPOCHS = 400
SAVE_INTERVAL = 10
LR = 3e-4

# TensorBoard setup
log_dir = f"runs/cmg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(log_dir, exist_ok=True)  # 创建目录
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs: {log_dir}")

# DataLoader
dataloader, stats = get_dataloader(DATA_PATH, batch_size=BATCH_SIZE)
print(f"Batches per epoch: {len(dataloader)}")

# Model
motion_dim = stats["motion_dim"]
command_dim = stats["command_dim"]

model = CMG(
    motion_dim=motion_dim,
    command_dim=command_dim,
    hidden_dim=512,
    num_experts=4,
    num_layers=3,
)
trainer = CMGTrainer(model, lr=LR)

# 加入自适应学习率调度器
# mode='min': 监控指标（Loss）越小越好
# factor=0.5: 触发时学习率减半
# patience=10: 如果连续10个epoch没有改进，则触发
# min_lr: 学习率下限
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    trainer.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

# Training
best_loss = float('inf')
start_time = time.time()
print("Starting training...")

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    loss = trainer.train_epoch(dataloader)
    scheduler.step(loss)  
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    remaining = (elapsed / (epoch + 1)) * (NUM_EPOCHS - epoch - 1)
    
    current_lr = trainer.optimizer.param_groups[0]['lr']
    # TensorBoard    
    writer.add_scalar('metrics/Learning Rate', current_lr, epoch)
    writer.add_scalar('metrics/MSE Loss', loss, epoch)
    writer.add_scalar('metrics/Teacher Prob', trainer.teacher_prob, epoch)
    # writer.add_scalar('Time/epoch_seconds', epoch_time, epoch)
    
    remaining_h = int(remaining // 3600)
    remaining_m = int((remaining % 3600) // 60)
    remaining_s = int(remaining % 60)
    
    # Console
    print(f"[{epoch+1:4d}/{NUM_EPOCHS}] "
          f"Loss: {loss:.4f} | "
          f"Teacher: {trainer.teacher_prob:.3f} | "
          f"Time: {epoch_time:.1f}s | "
          f"ETA: {remaining_h:02d}:{remaining_m:02d}:{remaining_s:02d}")
    
    # Save best
    if loss < best_loss:
        best_loss = loss
        torch.save(model.state_dict(), f"{log_dir}/cmg_best.pt")
        print(f"  -> New best: {best_loss:.4f}")
    
    # Checkpoint
    if (epoch + 1) % SAVE_INTERVAL == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss': loss,
            'stats': stats,
        }, f"{log_dir}/cmg_ckpt_{epoch+1}.pt")

# Final
torch.save({
    'model_state_dict': model.state_dict(),
    'stats': stats,
}, f"{log_dir}/cmg_final.pt")

writer.close()
print(f"\nDone! Total: {(time.time()-start_time)/60:.1f}min, Best: {best_loss:.4f}")
print(f"Models saved to: {log_dir}/")