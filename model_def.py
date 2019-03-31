##
#
# The model used below was discussed in this paper:
# http://cs231n.stanford.edu/reports/2016/pdfs/419_Report.pdf
#
# It follows the following pattern. Note: conv<A>-<B> -> A x A filter with B channels
# conv3-32 - relu - conv3-32 - relu - maxpool - dropout - conv3-64 - relu - conv3-64 -
# relu - maxpool - dropout - fc(512) - relu - dropout - softmax
##


import torch.nn as nn
import torch.optim as optim


class Model666(nn.Module):
    def __init__(self, num_classes=200, layer1ch=32, layer2ch=64, hidden_size=512, k_size=3, padding=1, do_rate=0.1):
        super(Model666, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=layer1ch,
                      kernel_size=k_size,
                      stride=1,
                      padding=padding),
            nn.ReLU(),
        )
        # output =
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=layer1ch,
                      out_channels=layer1ch,
                      kernel_size=k_size,
                      stride=1,
                      padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Dropout(do_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=layer1ch,
                      out_channels=layer2ch,
                      kernel_size=k_size+2,
                      stride=1,
                      padding=padding+1),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=layer2ch,
                      out_channels=layer2ch,
                      kernel_size=k_size+2,
                      stride=1,
                      padding=padding+1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.Dropout(do_rate)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 16 * layer2ch, hidden_size),
            nn.ReLU(),
            nn.Dropout(do_rate), )

        # self.fc = nn.Linear(hidden_size, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.Softmax())
        # nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


def get_model(device, opt='Adam', num_classes=200, layer1ch=32, layer2ch=64, hidden_size=512, k_size=3, padding=1,
              lamb=0.01, do_rate=0.1, learning_rate=0.01):
    model = Model666(num_classes=num_classes,
                     layer1ch=layer1ch,
                     layer2ch=layer2ch,
                     hidden_size=hidden_size,
                     k_size=k_size,
                     padding=padding,
                     do_rate=do_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamb)
    elif opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lamb)
    else:
        print("Unknown value. Choosing Adam instead.")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamb)

    return model, criterion, optimizer

# class Model666(nn.Module):
#     def __init__(self, num_classes=200):
#         super(Model666, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=3,
#                       out_channels=layer1ch,
#                       kernel_size=k_size,
#                       stride=1,
#                       padding=padding),
#             # nn.BatchNorm2d(1),
#             nn.ReLU(),
#             nn.Dropout(do_rate),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2))
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=layer1ch,
#                       out_channels=layer2ch,
#                       kernel_size=k_size,
#                       stride=1,
#                       padding=padding),
#             # nn.BatchNorm2d(1),
#             nn.ReLU(),
#             nn.Dropout(do_rate),
#             nn.MaxPool2d(kernel_size=2,
#                          stride=2))
#
#         # self.layer1 = nn.Sequential(
#         #     nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
#         #     # nn.BatchNorm2d(1),
#         #     nn.ReLU(),
#         #     nn.Dropout(do_rate),
#         #     nn.MaxPool2d(kernel_size=2, stride=2))
#         # self.layer2 = nn.Sequential(
#         #     nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
#         #     # nn.BatchNorm2d(1),
#         #     nn.ReLU(),
#         #     nn.Dropout(do_rate),
#         #     nn.MaxPool2d(kernel_size=2, stride=2))
#         # self.layer3 = nn.Sequential(
#         #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#         #     # nn.BatchNorm2d(1),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=2, stride=2))
#         # self.layer4 = nn.Sequential(
#         #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
#         #     # nn.BatchNorm2d(32),
#         #     nn.ReLU(),
#         #     nn.MaxPool2d(kernel_size=2, stride=2))
#         # self.fc1 = nn.Linear(32*3*3, hidden_size)
#         # self.fc1 = nn.Linear(512, hidden_size)
#         self.fc1 = nn.Sequential(
#             nn.Linear(16 * 16 * layer2ch, hidden_size),
#             nn.ReLU())
#         # self.fc = nn.Linear(hidden_size, num_classes)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, num_classes),
#             # nn.Linear(hidden_size, num_classes),
#             nn.Softmax())
#         # nn.ReLU())
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         # out = self.layer3(out)
#         # out = self.layer4(out)
#         out = out.reshape(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.fc(out)
#         return out
