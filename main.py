##
# CSE 666: Assignment 1
#
# @author vedharis
# Ved Harish Valsangkar
# Person number: 50290388
#
# References used:
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
# https://gist.github.com/bveliqi/a847f955f2ec13d74d22c088c6f771b4
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#
# For Dropout:
# https://stats.stackexchange.com/questions/240305/where-should-i-place-dropout-layers-in-a-neural-network
#
##

import torch
import torch.cuda as cuda
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import model_def as mod

import time


def main():
    device = torch.device("cuda:0" if cuda.is_available() else "cpu")

    num_epochs = 5
    batch_size = 100

    print("CSE666: Biometrics\nAssignment 1\n\nStarting training\n")
    start = time.time()

    # Transform example
    # https://discuss.pytorch.org/t/changing-transformation-applied-to-data-during-training/15671

    # data_transform = transforms.Compose([transforms.RandomSizedCrop(224),
    data_transform = transforms.Compose([transforms.RandomHorizontalFlip(0.2),
                                         transforms.RandomVerticalFlip(0.2),
                                         # transforms.RandomResizedCrop(60),
                                         transforms.ToTensor()])

    data_dir = '/home/ved/PycharmProjects/CSE666_ass_1/tiny-imagenet-200/'
    op_dir = '/home/ved/PycharmProjects/CSE666_ass_1/op/'

    train_dataset = datasets.ImageFolder(data_dir + 'train', transform=data_transform)
    test_dataset = datasets.ImageFolder(data_dir + 'val', transform=transforms.ToTensor())
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    # model, criterion, optimizer = mod.get_model(device, num_classes=200, layer1ch=32, layer2ch=64, hidden_size=512,
    #                                             k_size=3, padding=1, lamb=1, do_rate=0.1, learning_rate=0.001)

    model, criterion, optimizer = mod.get_model(device)

    total_len = len(train_loader)
    running_loss = 0.0

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            # Change variable type to match GPU requirements
            inp = images.to(device)
            lab = labels.to(device)

            # Reset gradients before processing
            optimizer.zero_grad()

            # Get model output
            out = model(inp)

            # Calculate loss
            loss = criterion(out, lab)

            # Update weights
            loss.backward()
            optimizer.step()

            # running_loss += loss.data[0]

            if i % 100 == 0:
                print("Epoch " + str(epoch) + " Step " + str(i+1) + "/" + str(total_len), end="\t")
                print("Running Loss data: ", loss.data)
                # print("\nRunning Loss (avg): ", running_loss/100)
                # running_loss = 0.0

    # Set to eval mode
    model.eval()
    lap = time.time()
    print("Training completed in {0}secs/{1}hrs".format(lap-start, (lap-start)/3600))

    # torch.no_grad() used to reduce gradient calculation which is not needed for testing.
    with torch.no_grad():
        correct = 0
        total = 0

        test_len = len(test_loader)
        print("\n\nRunning test\n")

        t = True

        for i, (images, labels) in enumerate(test_loader):  # images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted, labels)
            print("----------------")
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if i % 20 == 0:
                print("Step " + str(i + 1) + "/" + str(test_len))

    accu = (100 * correct) / total
    print('Accuracy of the model on the test images: {:.2f} %'.format(accu))
    print('Correct: {0}\nTotal: {1}'.format(correct, total))

    filename = op_dir + 'A1_A:{:.2f}_'.format(accu) + str(time.strftime("%Y%m%d_%H%M%S", time.gmtime())) + '.ckpt'
    torch.save(model.state_dict(), filename)
    end = time.time()
    print("Model saved. Program completed in {0}secs/{1}hrs".format(end-start, (end-start)/3600))


if __name__ == "__main__":
    main()
