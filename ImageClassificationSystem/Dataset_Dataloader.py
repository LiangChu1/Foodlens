import cv2
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

TOTAL_CATEGORIES = 101
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class DataPrep(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for idx, folder in enumerate(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                self.images.append(image_path)
                self.labels.append(idx)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((600, 600), transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataPath = r'.\Food_Data'
dataset = DataPrep(dataPath, transform=data_transforms)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

class FoodModel(nn.Module):
    def __init__(self, num_classes):
        super(FoodModel, self).__init__()
        self.resnet = resnet50()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

model = FoodModel(TOTAL_CATEGORIES).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += (loss.item() * images.size(0))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    scheduler.step()
    epoch_loss = running_loss / len(dataset)
    epoch_accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    if (epoch % 2 == 0) or (epoch == 1):
        torch.save(model.state_dict(), 'CheckpointFoodModel.pt')

torch.save(model.state_dict(), 'FoodModel.pt')
print('Model Training Sucessfuly Completed')