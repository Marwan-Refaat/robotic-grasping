import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from PIL import Image

from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.visualisation.plot import plot_results, save_results

logging.basicConfig(level=logging.INFO)

#Function that adds command-line arguments, self-explanatory
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate network')
    parser.add_argument('--network', type=str,
                        help='Path to saved network to evaluate')
    parser.add_argument('--rgb_path', type=str, default='cornell/08/pcd0845r.png',
                        help='RGB Image path')
    parser.add_argument('--depth_path', type=str, default='cornell/08/pcd0845d.tiff',
                        help='Depth Image path')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--save', type=int, default=0,
                        help='Save the results')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    
    # Load image (Uses args to get rgb and depth path)
    logging.info('Loading image...')
    pic = Image.open(args.rgb_path, 'r')
    rgb = np.array(pic)
    pic = Image.open(args.depth_path, 'r')
    depth = np.expand_dims(np.array(pic), axis=2) #What does this do?

    # Load Network (loads the trained model file)
    logging.info('Loading model...')
    net = torch.load(args.network)
    logging.info('Done')

    # Get the compute device
    device = get_device(args.force_cpu) #Custom file (basic function that chooses GPU or CPU as processing device)

    img_data = CameraData(include_depth=args.use_depth, include_rgb=args.use_rgb) #Custom class that parses image

    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth) #Get transformed/processed images and Pytorch tensor

    with torch.no_grad():
        xc = x.to(device) #Pytorch magic? converts tensor to tensor with device?
        pred = net.predict(xc) #Where the magic happens, runs the network on the tensor

        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width']) #Custom class, gets the three images

        if args.save:
            save_results(
                rgb_img=img_data.get_rgb(rgb, False), #Gets non-normalized RGB image
                depth_img=np.squeeze(img_data.get_depth(depth)), #Gets depth image, why squeeze?
                grasp_q_img=q_img,
                grasp_angle_img=ang_img,
                no_grasps=args.n_grasps,
                grasp_width_img=width_img
            )
        else:
            fig = plt.figure(figsize=(10, 10))
            plot_results(fig=fig,
                         rgb_img=img_data.get_rgb(rgb, False),
                         grasp_q_img=q_img,
                         grasp_angle_img=ang_img,
                         no_grasps=args.n_grasps,
                         grasp_width_img=width_img)
            fig.savefig('img_result.pdf')
