import torch
import torch.cuda as cuda
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import model_def as mod

data_dir = '/home/ved/PycharmProjects/CSE666_ass_1/tiny-imagenet-200/'
op_dir = '/home/ved/PycharmProjects/CSE666_ass_1/op/'

device = torch.device("cuda:0" if cuda.is_available() else "cpu")

batch_size = 100

test_dataset = datasets.ImageFolder(data_dir + 'val', transform=transforms.ToTensor())
test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

model, _, _ = mod.get_model(device)

m = torch.load("/home/ved/PycharmProjects/CSE666_ass_1/op/A1_A:0.00_20190223_231257.ckpt")

model.load_state_dict(m)
model.eval()

print("M = ", m)

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

        # print(predicted, labels)
        # print("--------", i, "--------")

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i % 20 == 0:
            print("Step " + str(i + 1) + "/" + str(test_len))

accu = (100 * correct) / total
print('Accuracy of the model on the test images: {:.2f} %'.format(accu))
print('Correct: {0}\nTotal: {1}'.format(correct, total))

