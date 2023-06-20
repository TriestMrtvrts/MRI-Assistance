import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from captum.attr import IntegratedGradients
import __main__


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # размер исходной картинки 180x180

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(8, 16, 8, padding=1)
        self.dropout2 = nn.Dropout(0.25)

        self.batchnorm2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(16, 32, 2, padding=1)
        self.dropout3 = nn.Dropout(0.25)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(32, 16, 16, padding=1)
        self.dropout4 = nn.Dropout(0.25)
        self.batchnorm4 = nn.BatchNorm2d(16)

        # flatten
        self.flatten = nn.Flatten()

        self.fc_2_1 = nn.Linear(28224, 512)
        self.fc_2_2 = nn.Linear(512, 4)

        # linear 1
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool1(x)

        x = func.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.batchnorm2(x)
        x = self.pool2(x)

        x_1 = func.relu(self.conv3(x))
        x_1 = self.dropout3(x_1)
        x_1 = self.batchnorm3(x_1)
        x_1 = self.pool3(x_1)

        x_1 = func.relu(self.conv4(x_1))
        x_1 = self.dropout4(x_1)
        x_1 = self.batchnorm4(x_1)

        x_1 = self.flatten(x_1)
        x_1 = func.relu(self.fc1(x_1))
        x_1 = self.fc2(x_1)

        x_2 = self.flatten(x)
        x_2 = func.relu(self.fc_2_1(x_2))
        x_2 = self.fc_2_2(x_2)

        return x_1 + x_2


setattr(__main__, "ConvNet", ConvNet)

device = 'cpu'
model_ = torch.load(os.path.join('model',
                                 'https://drive.google.com/file/d/1IwBFus08Nf2tpG06It9KD1FHW-g-yYEc/view?usp=sharing'
                                 # download latest version of model from here
                                 ))
model_.eval()


def get_class_of_demension(idx):
    classes = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    return classes[idx]


def get_segmented_map(image_attr: np.array,
                      color_map: str = 'positive',
                      borders: tuple = (20, 20)) -> np.array:
    """arg: color_map: [positive, all]"""
    if color_map != 'all':
        for i in range(len(image_attr)):
            for j in range(len(image_attr[i])):
                flag_zero = False
                if color_map == 'positive':
                    if max(image_attr[i][j]) != image_attr[i][j][1]:
                        flag_zero = True
                    else:
                        if sum(image_attr[i][j]) - max(image_attr[i][j]) > borders[1]:
                            flag_zero = True
                elif color_map == 'negative':
                    if max(image_attr[i][j]) == image_attr[i][j][1] or max(image_attr[i][j]) == image_attr[i][j][2]:
                        flag_zero = True
                    else:
                        if sum(image_attr[i][j]) - max(image_attr[i][j]) > borders[0]:
                            flag_zero = True
                if flag_zero:
                    image_attr[i][j] = [0, 0, 0]
    return image_attr


def show_pack_of_images(images, labels):
    f, axes = plt.subplots(1, len(images), figsize=(30, 5))
    for i, axis in enumerate(axes):
        img = images[i]
        axes[i].imshow(img)
        axes[i].set_title(labels[i])
    plt.show()


def create_color_map_igrad(net, img_path: str) -> tuple:
    integrated_gradients = IntegratedGradients(net)
    img = cv2.cvtColor(cv2.resize(cv2.imread(img_path, 0), (180, 180)), cv2.COLOR_GRAY2RGB)
    img_tensor = torch.from_numpy(np.array(img).astype(np.float32)).to('cpu')
    img_tensor = img_tensor.permute(2, 0, 1) / 255
    img_tensor = img_tensor.unsqueeze(0)

    output = model_(img_tensor)
    prob = func.softmax(output, dim=0)
    probability = float(np.max(prob.detach().numpy()))
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()
    predicted_label = pred_label_idx.item()

    attributions_ig = integrated_gradients.attribute(img_tensor, target=pred_label_idx, n_steps=200)

    imgs = [(img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8),
            (np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)) * 255).astype(np.uint8)]
    imgs.extend([get_segmented_map(imgs[1].copy(), 'negative'), get_segmented_map(imgs[1].copy(), 'positive')])
    labels = [get_class_of_demension(predicted_label), 'all', 'negative', 'positive']

    return imgs, labels, probability


def get_results_model(image_path, model):
    images, labels, probability = create_color_map_igrad(model, image_path)

    img = images[3].copy()
    original = images[0].copy()

    result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    result = cv2.blur(result, (5, 5));

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    ret, result = cv2.threshold(result, 0.3 * max_val, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for element in contours:
        if 150 > len(element) > 35:
            color = (255, 0, 0)
            x, y, w, h = cv2.boundingRect(element)
            cv2.rectangle(original, (x - 2, y - 2), (x + w + 1, y + h + 1), color, 1)

    return original, labels[0], probability
