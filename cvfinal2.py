import torch
import torchvision
from transformers import AutoImageProcessor, AutoModelForImageClassification
import matplotlib.pyplot as plt


def cal_accuracy(trained_model, test_data):
    with torch.no_grad():
        accuracy, num_examples = 0, 0
        for idx, data in enumerate(test_data, 0):
            images, labels = data
            outputs = trained_model(images)
            _, predicted = torch.max(outputs, 1)
            equal = predicted.eq(labels.view_as(predicted))
            accuracy += torch.sum(equal)
            num_examples += labels.size(0)

    return accuracy / num_examples


if __name__ == '__main__':

    batch_size = 16

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        torchvision.transforms.v2.CutMix(num_classes=10)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = torch.hub.load('pytorch/vision', 'convnext_tiny', pretrained=True)
    model1.classifier[2] = torch.nn.Linear(in_features=model1.classifier.in_features, out_features=10, bias=True)

    processor = AutoImageProcessor.from_pretrained("swin_transformer")
    model2 = AutoModelForImageClassification.from_pretrained("swin_transformer")
    model2.classifier = torch.nn.Linear(in_features=model2.classifier.in_features, out_features=10, bias=True)

    model1.to(device)
    model2.to(device)

    num_epochs = 200
    criterion = torch.nn.CrossEntropyLoss()

    base_params1 = [p for p in model1.parameters() if id(p) not in [id(param) for param in model1.classifier[2].parameters()]]
    base_params2 = [p for p in model2.parameters() if id(p) not in [id(param) for param in model2.classifier.parameters()]]

    optimizer1 = torch.optim.Adam([
        {'params': base_params1},
        {'params': model1.classifier[2].parameters(), 'lr': 5e-4}
    ], lr=1e-4)

    optimizer2 = torch.optim.Adam([
        {'params': base_params2},
        {'params': model2.classifier.parameters(), 'lr': 5e-4}
    ], lr=1e-4)

    loss_list1 = []
    loss_list2 = []

    accuracy_list1 = []
    accuracy_list2 = []

    for epoch in range(num_epochs):
        model1.train()
        model2.train()

        epoch_loss1 = 0
        epoch_loss2 = 0

        for batch_idx, batch in enumerate(trainloader, 0):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            outputs1 = model1(images)
            outputs2 = model2(images)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss1 = criterion(outputs1, labels)
            loss2 = criterion(outputs2, labels)

            loss1.backward()
            loss2.backward()

            optimizer1.step()
            optimizer2.step()

            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch_idx}, CNN Loss: {loss1.item()}, Transformer Loss: {loss2.item()}')

        model1.eval()
        model2.eval()

        loss_list1.append(epoch_loss1/len(trainloader))
        loss_list2.append(epoch_loss2/len(trainloader))

        accuracy_list1.append(cal_accuracy(model1, testloader))
        accuracy_list2.append(cal_accuracy(model2, testloader))

        print(f'CNN test accuracy: {accuracy_list1[-1]}, Transformer test accuracy: {accuracy_list2[-1]}')

    torch.save(model1, f'trained_model//convnext//Adam_CEL_Epoch{num_epochs}.pth')
    torch.save(model2, f'trained_model//swim_transformer//Adam_CEL_Epoch{num_epochs}.pth')

    fig, ax1 = plt.subplots()
    ax1.plot(range(num_epochs), loss_list1, color='blue', label='CNN loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('CNN Loss')

    ax2 = ax1.twinx()
    ax2.plot(range(num_epochs),loss_list2, color='red', label='Transformer Loss')
    ax2.set_ylabel('Transformer Loss')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Loss of different model')
    plt.savefig(f'figure//Loss_Adam_CEL_epoch{num_epochs}.png', dpi=300)
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(range(num_epochs), accuracy_list1, color='blue', label='CNN accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('CNN accuracy')

    ax2 = ax1.twinx()
    ax2.plot(range(num_epochs), accuracy_list2, color='red', label='Transformer accuracy')
    ax2.set_ylabel('Transformer accuracy')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('testing accuracy of different model')
    plt.savefig(f'figure//Accuracy_Adam_CEL_epoch{num_epochs}.png', dpi=300)
    plt.show()

