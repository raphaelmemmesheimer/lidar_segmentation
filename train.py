# Train a unet segmentation model 
import torch
from tqdm import tqdm
from person_segmentation import PersonSegmentationModelUNet
from semantic_kitti_dataset import SemanticKittiDataset

print('Creating model and dataset...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PersonSegmentationModelUNet(in_channels=5, out_channels=2).to(device)
dataset = SemanticKittiDataset('.', transform=None, split='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

print('Creating loss and optimizer...')
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('Training model...')
for epoch in tqdm(range(10)):
    for x, y in tqdm(dataloader):
        # print(x.shape, y.shape)
        # print(x.dtype, y.dtype)
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            y_pred = model(x)
            # print(type(y_pred))
            loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # show loss in tqdm
        #tqdm.set_postfix(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
        #tqdm.write("WOrking")
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
