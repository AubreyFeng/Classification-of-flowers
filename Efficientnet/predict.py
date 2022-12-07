from concurrent.futures import ThreadPoolExecutor
import json
import os

import matplotlib.pyplot as plt
import torch
from model import efficientnetv2_s as create_model
from PIL import Image
from torchvision import transforms

# create model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_model(num_classes=5).to(device)
# load model weights
model_weight_path = "weights/model-29.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


# read class_indict
json_path = 'class_indices.json'
assert os.path.exists(
    json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)


# read img_dir
path = "/home/fa/data/flower_photos"  # 文件夹目录
dirs = os.listdir(path)  # 得到文件夹下的所有文件名称
imgs = []
for dir in dirs:  # 遍历文件夹
    # print(dir)  # 打印结果
    path_flower = os.path.join(path, dir)
    if os.path.isdir(path_flower):
        # print(path_f)
        files = os.listdir(path_flower)
        for file in files:
            img_path = os.path.join(path_flower, file)
            # print(img_path)
            imgs.append(img_path)

img_size = {"s": [300, 384],  # train_size, val_size
            "m": [384, 480],
            "l": [384, 480]}
num_model = "s"

data_transform = transforms.Compose(
    [transforms.Resize(img_size[num_model][1]),
     transforms.CenterCrop(img_size[num_model][1]),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def main(img_path):
    # load image
    assert os.path.exists(
        img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(
        class_indict[str(predict_cla)], predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    with ThreadPoolExecutor(10) as t:
        for img in imgs:
            t.submit(main, img_path=img)
