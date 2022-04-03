import cv2
import numpy as np
import os
import sys
import torch
import argparse
import torchvision.transforms as T

from utils.getter import *
from utils.logger import setup_logger
from torch.backends import cudnn
from PIL import Image

sys.path.append('.')

def visualizer(test_img, camid, top_k=10, img_size=[128, 128]):
    figure = np.asarray(query_img.resize((img_size[1], img_size[0])))
    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        print('Top %d: %s'  % (k +1, img_path[indices[0][k]]))
        img = np.asarray(Image.open(img_path[indices[0][k]]).resize(
            (img_size[1], img_size[0])))
        figure = np.hstack((figure, img))
        title = name
    figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
    result_path = os.path.join(config.log_dir, "results")

    if not os.path.exists(result_path):
        print('Create a new folder named results in {}'.format(config.log_dir))
        os.makedirs(result_path)

    cv2.imwrite(os.path.join(
        result_path, "{}-cam{}.png".format(test_img, camid)), figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str)
    parser.add_argument(
        "--query_image", default="", help="path to query image", type=str)

    args = parser.parse_args()

    config_path = os.path.join('configs', f'{args.config_file}.yaml')
    config = Config(config_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    cudnn.benchmark = True

    model = Backbone(num_classes=751, model_name=config.model_name, training=False)
    model.load_param(config.test_weight)

    model = model.to(config.device)

    transform = T.Compose([
        T.Resize(config.image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    log_dir = config.log_dir
    logger = setup_logger(
        'person-reid-baseline.test', log_dir, 0)
    model.eval()

    #test all image from query dir

    path_to_img = args.query_image
    test_img = path_to_img.split("/")[-1]
    logger.info('Finding ID {} ...'.format(test_img))

    gallery_feats = torch.load(os.path.join(config.log_dir, config.gfeats))
    img_path = np.load(os.path.join(config.log_dir, config.img_path))

    query_img = Image.open(path_to_img)
    input = torch.unsqueeze(transform(query_img), 0)
    input = input.to(config.device)
    with torch.no_grad():
        query_feat = model(input)

    dist_mat = cosine_similarity(query_feat, gallery_feats)
    indices = np.argsort(dist_mat, axis=1)
    visualizer(test_img, camid='mixed', top_k=10,
                img_size=config.image_size)
        
        
