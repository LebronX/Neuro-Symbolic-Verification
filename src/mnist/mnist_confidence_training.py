from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize = 14


class ConfidenceNet(nn.Module):
    def __init__(self):
        super(ConfidenceNet, self).__init__()
        self.fc_input = nn.Linear(resize * resize, 10)
        # self.fc1 = nn.Linear(10, 10)  # 3 * 10 for hidden layers, 0.9 accuracy
        # self.fc2 = nn.Linear(10, 10)
        # self.fc3 = nn.Linear(10, 10)
        self.fc_output = nn.Linear(10, 10)

        self.fc_input_prop = nn.Linear(resize * resize, 10)
        # self.fc1_prop = nn.Linear(10, 10)  # 35 * 5 and 5 * 35 for hidden layers
        # self.fc2_prop = nn.Linear(10, 20)
        self.fc_output_prop = nn.Linear(10, resize * resize)

    def forward(self, x):
        x_prop = x

        # For network under verified
        x = self.fc_input(x)
        x = F.relu(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        x = self.fc_output(x)
        output = x

        # for property network
        x_prop = self.fc_input_prop(x_prop)
        x_prop = F.relu(x_prop)
        # x_prop = self.fc1_prop(x_prop)
        # x_prop = F.relu(x_prop)
        # x_prop = self.fc2_prop(x_prop)
        # x_prop = F.relu(x_prop)
        x_prop = self.fc_output_prop(x_prop)
        output_prop = x_prop

        return output, output_prop


def train(args, model, device, train_loader, optimizer, epoch):

    model.train()
    criterion_nuv = nn.CrossEntropyLoss()
    criterion_prop = nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.reshape(-1, resize * resize).to(device), target.to(
            device
        )  # for fc input layer
        # data, target = data.to(device), target.to(device) # for conv
        optimizer.zero_grad()
        output, output_prop = model(data)
        loss_nuv = criterion_nuv(output, target)

        # For property network
        loss_prop = criterion_prop(output_prop, data)

        loss = loss_nuv + loss_prop  # total sum of loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, device, test_loader):

    model.eval()
    test_loss = 0
    correct = 0
    test_loss_prop = 0
    sm = nn.Softmax(dim=1)
    criterion = nn.CrossEntropyLoss(reduction="sum").to(device)
    criterion_prop = nn.MSELoss(reduction="mean").to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.reshape(-1, resize * resize).to(device), target.to(
                device
            )  # for fc input layer
            output, output_prop = model(data)

            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # For ae robustness property
            test_loss_prop += criterion_prop(
                output_prop, data
            ).item()  # sum up property nn loss

            # For finding counter example
            # if torch.norm(output_prop - data, float("inf")).item() < 1:
            #     print("haha")
            #     if pred.eq(target.view_as(pred)).item():
            #         if torch.max(sm(output)).item() < 0.8:
            #             print("low confidence counterexample: ")
            #             print(data.reshape(-1, resize*resize))
            #             plt.imshow(data.reshape(resize,resize), cmap="gray")
            #             plt.show()
            #     else:
            #         print("wrong counterexample: ")
            #         print(data.reshape(-1, resize*resize))

    test_loss /= len(test_loader.dataset)
    test_loss_prop /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    print(
        "\nTest set: Average property loss: {:.4f}\n".format(
            test_loss_prop,
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.5,
        metavar="LR",
        help="learning rate (default: 0.5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = ConfidenceNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # images, labels = next(iter(train_loader))
    # print(images[1])
    # plt.imshow(images[1].reshape(resize,resize), cmap="gray")
    # plt.show()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        save_net = "../../networks/mnist/mnist_confidence_exp" + str(args.seed)
        torch.save(model.state_dict(), save_net + ".pt")

        # Model for marabou
        dummy_input = torch.randn(resize * resize)
        torch.onnx.export(
            model,
            dummy_input,
            save_net + ".onnx",
            output_names=["output_nuv", "output_prop"],
            export_params=True,
        )


if __name__ == "__main__":
    main()
