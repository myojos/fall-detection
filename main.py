import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from .UCF_dataset import UCFDataset
from .transformers import ToTensor, RandomHorizontalFlip
from .two_stream_fall_net import TwoStreamFallNet


# Define train test functions
def train(device, model, train_loader, optimizer, loss_function, epoch, writer):
    model = model.to(device)
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        rgb = data['rgb'].to(device)
        flow = data['flow'].to(device)
        target = data['y'].to(device)

        optimizer.zero_grad()
        output = model(rgb, flow)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    writer.add_scalar("train_loss", train_loss, global_step=epoch)


def test(device, model, test_loader, loss_function, epoch, writer):
    model = model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            rgb = data['rgb'].to(device)
            flow = data['flow'].to(device)
            target = data['y'].to(device)
            output = model(rgb, flow)

            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    writer.add_scalar("test_loss", test_loss, global_step=epoch)
    writer.add_scalar("accuracy", accuracy, global_step=epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))


def main():
    train_transforms = transforms.Compose([ToTensor(), RandomHorizontalFlip()])
    # Create dataloaders
    train_dataset = UCFDataset('data_list.csv', root_dir='.', train=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_dataset = UCFDataset('data_list.csv', root_dir='.', train=False, transform=ToTensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)

    device = torch.device("cuda:0")
    loss_function = nn.CrossEntropyLoss()
    model = TwoStreamFallNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 12, 0.1, -1)
    writer = SummaryWriter("logs/normalized")

    # Training loop
    for epoch in range(1, 40):
        try:
            train(device, model, train_loader, optimizer, loss_function, epoch, writer)
            test(device, model, test_loader, loss_function, epoch, writer)
            scheduler.step()
        except KeyboardInterrupt:
            break
    writer.close()


if __name__ == '__main__':
    main()
