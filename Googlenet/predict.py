import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    # img_path = "./rose.jpg"
    path = "/home/fa/data/flower_photos/daisy"
    file_name_list = os.listdir(path=path)
    for k in file_name_list:
        img_path = "/home/fa/data/flower_photos/daisy/" + str(k)
        # print(image_path)
        # img_path = "C:\\Users\\86182\\Downloads\\data\\flower_photos\\tulips\\10791227_7168491604.jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = GoogLeNet(num_classes=5, aux_logits=False).to(device)

        # load model weights
        weights_path = "./googleNet.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                              strict=False)

        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        plt.title(print_res)
        for i in range(len(predict)):
            print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                      predict[i].numpy()))
        # plt.show()


if __name__ == '__main__':
    main()
