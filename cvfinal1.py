import cv2
import torch
import torchvision
import matplotlib.pyplot as plt


class RotationDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, transform):
        if is_train:
            dataset = torchvision.datasets.CIFAR10(root='data/', train=True, transform=transform, download=True)
        else:
            dataset = torchvision.datasets.CIFAR10(root='data/', train=False, transform=transform, download=True)

        self.length = len(dataset)
        self.images = []
        self.labels = [i % 4 for i in range(self.length * 4)]
        for image, _ in dataset:
            img = image.permute(1, 2, 0).detach().numpy()
            img_90 = cv2.flip(cv2.transpose(img.copy()), 1)
            img_180 = cv2.flip(cv2.transpose(img_90.copy()), 1)
            img_270 = cv2.flip(cv2.transpose(img_180.copy()), 1)
            self.images += [torch.tensor(img).permute(2, 0, 1), torch.tensor(img_90).permute(2, 0, 1),
                            torch.tensor(img_180).permute(2, 0, 1), torch.tensor(img_270).permute(2, 0, 1)]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.length


def RotationDataLoader(batch_size, transform):
    train_iter = torchvision.utils.data.DataLoader(RotationDataset(is_train=True, transform=transform), batch_size=batch_size, shuffle=True)
    test_iter = torchvision.utils.data.DataLoader(RotationDataset(is_train=False, transform=transform), batch_size=batch_size)
    return train_iter, test_iter


def SuperviseDataLoader(batch_size, transform):
    train_dataset = torchvision.datasets.CIFAR10(root='data/', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='data/', train=False, transform=transform, download=True)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_iter, test_iter


def Focal_loss(pred, target,  alpha=0.5, gamma=2):
    logpt = -torch.nn.CrossEntropyLoss(reduction='none')(pred, target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, 1)

    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / len(y)


def train(net, train_iter, test_iter, start, num_epochs, lr, device, threshold, save_path):
    net = net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(start, num_epochs):
        net.train()
        train_loss = 0.0
        train_acc = 0.0
        data_num = 0

        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()
            train_acc += accuracy(y_hat.detach(), y.detach())
            data_num += 1

        history['train_loss'].append(float(train_loss / data_num))
        history['train_acc'].append(float(train_acc / data_num))

        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        data_num = 0

        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            with torch.no_grad():
                loss = criterion(y_hat, y)
                test_loss += loss.detach()
                test_acc += accuracy(y_hat.detach(), y.detach())

                data_num += 1

        history['test_loss'].append(float(test_loss / data_num))
        history['test_acc'].append(float(test_acc / data_num))
        if history['test_acc'][-1] > threshold:
            print("early stop")
            break

    torch.save(net, save_path)
    return history


def plot_figure(x, x_label, y1, y1_label, y2, y2_label, title, save_path):
    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color='blue', label=y1_label)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color='red', label=y2_label)
    ax2.set_ylabel(y2_label)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    batch_size = 32
    num_classes = 10
    num_rotation_epochs = 100
    num_supervise_epochs = 100
    lr = 1e-4
    threshold = 0.95

    resnet18 = torchvision.models.resnet18(pretrained=True)
    resnet18.fc = torch.nn.Linear(in_features=resnet18.fc.in_features, out_features=num_classes, bias=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size = None
    is_freeze = False
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_iter, test_iter = RotationDataLoader(batch_size=batch_size, transform=transform)
    history1 = train(resnet18, train_iter, test_iter, 0, num_rotation_epochs, lr, device, threshold, '')

    if is_freeze:
        for param in resnet18.parameters():
            param[1].requires_grad = False

    new_resnet18 = resnet18[:-2]
    new_resnet18.add_module("new Adapt", torch.nn.AdaptiveAvgPool2d((1, 1)))
    new_resnet18.add_module("new Flatten", torch.nn.Flatten())
    new_resnet18.add_module("new linear", torch.nn.Linear(in_features=resnet18.fc.in_features, out_features=num_classes))

    lr = 1e-4
    train_iter, test_iter = SuperviseDataLoader(batch_size=batch_size, transform=transform)
    history2 = train(new_resnet18, train_iter, test_iter, num_rotation_epochs, num_rotation_epochs + num_supervise_epochs, lr, device, threshold, '')






