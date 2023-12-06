from datasets.dataload import create_dataloader
from model.ImagePromptGenerator import GridWithTransformer

train_loader, val_loader, test_loader = create_dataloader(batch_size = 16, num_workers=8)

model = GridWithTransformer()

for i, (imgs, caps, caplens) in enumerate(train_loader):
    log_vars, loss = model.train_step(imgs, caps, caplens)
    print(log_vars)
    break