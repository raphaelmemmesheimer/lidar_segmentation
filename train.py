# Train a unet segmentation model 
import torch
from tqdm import tqdm
from utils.person_segmentation import PersonSegmentationModelUNet
from utils.semantic_kitti_dataset import SemanticKittiDataset
from model.resnetunet import ResNetUNet

print('Creating model and dataset...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = PersonSegmentationModelUNet(in_channels=5, out_channels=1).to(device)
model = ResNetUNet(in_channels=5, out_channels=22).to(device)
dataset = SemanticKittiDataset('.', transform=None, split='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

#dataset.show(0)

print('Creating loss and optimizer...')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('Training model...')
for epoch in tqdm(range(10)):
    for x, y in tqdm(dataloader):
        #print(x.shape, y.shape)
        #print(x.dtype, y.dtype)
        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast():
            y_pred = model(x)
            #print(type(y_pred), y_pred.shape)
            loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # show loss in tqdm
        #tqdm.set_postfix(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
        #tqdm.write("WOrking")
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
