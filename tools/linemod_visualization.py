import _init_paths
import argparse
import os
import random
import numpy as np
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from torch_cluster import knn

import matplotlib.pyplot as plt
import cv2
import sys

def project_points(points_3d, rmat, tvec, cam_fx, cam_fy, cam_cx, cam_cy):
    """
    Project 3D points onto the 2D image plane using camera intrinsics.

    Args:
        points_3d (np.ndarray): 3D points of shape (N, 3).
        rmat (np.ndarray): Rotation matrix of shape (3, 3).
        tvec (np.ndarray): Translation vector of shape (3,).
        cam_fx (float): Focal length in x-axis.
        cam_fy (float): Focal length in y-axis.
        cam_cx (float): Principal point x-coordinate.
        cam_cy (float): Principal point y-coordinate.

    Returns:
        np.ndarray: 2D projected points of shape (N, 2).
    """
    # Apply rotation and translation
    points_cam = np.dot(rmat, points_3d.T).T + tvec
    # Avoid division by zero
    z = points_cam[:, 2]
    z[z == 0] = 1e-6
    # Perspective division
    x = (points_cam[:, 0] * cam_fx) / z + cam_cx
    y = (points_cam[:, 1] * cam_fy) / z + cam_cy
    points_2d = np.stack((x, y), axis=1)
    return points_2d

def draw_model(image, model_points, rmat, tvec, cam_fx, cam_fy, cam_cx, cam_cy):
    """
    Draw the 3D model and coordinate axes on the image.

    Args:
        image (np.ndarray): Original image in BGR format.
        model_points (np.ndarray): 3D model points of shape (N, 3).
        rmat (np.ndarray): Rotation matrix of shape (3, 3).
        tvec (np.ndarray): Translation vector of shape (3,).
        cam_fx (float): Focal length in x-axis.
        cam_fy (float): Focal length in y-axis.
        cam_cx (float): Principal point x-coordinate.
        cam_cy (float): Principal point y-coordinate.

    Returns:
        np.ndarray: Image with the 3D model and axes drawn.
    """
    # Project model points to 2D
    projected_points = project_points(model_points, rmat, tvec, cam_fx, cam_fy, cam_cx, cam_cy).astype(int)

    # Draw the model points
    for point in projected_points:
        if 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]:
            cv2.circle(image, tuple(point), 2, (0, 255, 255), -1)  # Yellow dots

    # Draw coordinate axes
    axis_length = 0.1  # Adjust based on your model's scale
    axes_3d = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ])
    projected_axes = project_points(axes_3d, rmat, tvec, cam_fx, cam_fy, cam_cx, cam_cy).astype(int)

    origin = tuple(projected_axes[0])
    x_axis = tuple(projected_axes[1])
    y_axis = tuple(projected_axes[2])
    z_axis = tuple(projected_axes[3])

    # Draw axes
    cv2.line(image, origin, x_axis, (0, 0, 255), 2)  # X-axis in red
    cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y-axis in green
    cv2.line(image, origin, z_axis, (255, 0, 0), 2)  # Z-axis in blue

    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--model', type=str, required=True, help='Path to PoseNet model')
    parser.add_argument('--refine_model', type=str, required=True, help='Path to PoseRefineNet model')
    parser.add_argument('--output_result_dir', type=str, default='experiments/eval_result/linemod', help='Directory to save evaluation results')
    opt = parser.parse_args()

    num_objects = 13
    objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    num_points = 500
    iteration = 4
    bs = 1
    dataset_config_dir = os.path.abspath(os.path.join(opt.dataset_root, '../dataset_config'))  # Corrected Path
    output_result_dir = opt.output_result_dir

    # Verify if dataset_config_dir exists
    models_info_path = os.path.join(dataset_config_dir, 'models_info.yml')
    if not os.path.isfile(models_info_path):
        print(f"Error: 'models_info.yml' not found at {models_info_path}")
        print("Please ensure that 'models_info.yml' exists in the dataset_config directory.")
        sys.exit(1)

    # Initialize models
    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    refiner.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()

    # Initialize dataset and dataloader
    testdataset = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

    # Retrieve camera parameters from the dataset
    cam_fx = testdataset.cam_fx
    cam_fy = testdataset.cam_fy
    cam_cx = testdataset.cam_cx
    cam_cy = testdataset.cam_cy

    sym_list = testdataset.get_sym_list()
    num_points_mesh = testdataset.get_num_points_mesh()
    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    # Load object diameters
    diameter = []
    with open(models_info_path, 'r') as meta_file:
        meta = yaml.load(meta_file, Loader=yaml.FullLoader)  # Added Loader parameter for safety
        for obj in objlist:
            if obj in meta:
                diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)  # Convert mm to meters and apply scaling
            else:
                print(f"Warning: Object {obj} not found in models_info.yml. Using default diameter.")
                diameter.append(0.1)  # Default diameter (adjust as necessary)
    print("Object diameters:", diameter)

    success_count = [0 for _ in range(num_objects)]
    num_count = [0 for _ in range(num_objects)]
    fw = open(os.path.join(output_result_dir, 'eval_result_logs.txt'), 'w')

    for i, data in enumerate(testdataloader, 0):
        points, choose, img, target, model_points, idx = data
        if len(points.size()) == 2:
            print(f'No.{i} NOT Pass! Lost detection!')
            fw.write(f'No.{i} NOT Pass! Lost detection!\n')
            continue

        # Move data to GPU
        points, choose, img, target, model_points, idx = (
            Variable(points).cuda(),
            Variable(choose).cuda(),
            Variable(img).cuda(),
            Variable(target).cuda(),
            Variable(model_points).cuda(),
            Variable(idx).cuda()
        )

        # Forward pass through PoseNet
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2, keepdim=True)
        pred_c = pred_c.view(bs, num_points)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(bs * num_points, 1, 3)

        # Extract the best prediction
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
        my_pred = np.append(my_r, my_t)

        # Refinement iterations
        for ite in range(iteration):
            T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
            my_mat = quaternion_matrix(my_r)
            R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
            my_mat[0:3, 3] = my_t

            new_points = torch.bmm((points - T), R).contiguous()
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / torch.norm(pred_r, dim=2, keepdim=True)
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r_final = copy.deepcopy(my_mat_final)
            my_r_final[0:3, 3] = 0
            my_r_final = quaternion_from_matrix(my_r_final, True)
            my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

            my_pred = np.append(my_r_final, my_t_final)
            my_r = my_r_final
            my_t = my_t_final

        # Compute predicted and target point clouds
        model_points_np = model_points[0].cpu().detach().numpy()
        my_r_matrix = quaternion_matrix(my_r)[:3, :3]
        pred = np.dot(model_points_np, my_r_matrix.T) + my_t
        target_np = target[0].cpu().detach().numpy()

        # Compute distance
        if idx[0].item() in sym_list:
            # Symmetric objects: use nearest neighbor distance
            pred_tensor = torch.from_numpy(pred.astype(np.float32)).cuda()
            target_tensor = torch.from_numpy(target_np.astype(np.float32)).cuda()

            # Reshape for torch_cluster knn
            pred_tensor = pred_tensor.contiguous()  # Shape: (N, 3)
            target_tensor = target_tensor.contiguous()  # Shape: (M, 3)

            # Get nearest neighbors using torch_cluster knn
            assign_index = knn(target_tensor, pred_tensor, k=1)[1]  # Returns (row, col), we want col indices

            # Reorder target using obtained indices
            target_tensor = target_tensor[assign_index]

            # Calculate distance
            dis = torch.mean(torch.norm((pred_tensor - target_tensor), dim=1)).item()
        else:
            # Asymmetric objects: use mean distance
            dis = np.mean(np.linalg.norm(pred - target_np, axis=1))

        # Determine success
        if dis < diameter[idx[0].item()]:
            success_count[idx[0].item()] += 1
            result = 'Pass'
        else:
            result = 'NOT Pass'

        print(f'No.{i} {result}! Distance: {dis}')
        fw.write(f'No.{i} {result}! Distance: {dis}\n')
        num_count[idx[0].item()] += 1

        # Visualization Part
        try:
            # Load the original image
            img_np = img[0].cpu().numpy().transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img_np = (img_np * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Retrieve the object ID
            obj_id = idx[0].item()

            # Draw the estimated pose on the image
            img_with_pred = copy.deepcopy(img_bgr)
            img_with_pred = draw_model(img_with_pred, model_points_np, my_r_matrix, my_t, cam_fx, cam_fy, cam_cx, cam_cy)

            # Display the image with overlay
            cv2.imshow('Pose Estimation', img_with_pred)
            key = cv2.waitKey(500) & 0xFF  # Display each image for 1 ms

            if key == ord('q'):
                print("Visualization interrupted by user.")
                break

        except Exception as e:
            print(f"Visualization failed for sample {i}: {e}")
            continue

    # After processing all samples, print success rates
    for i in range(num_objects):
        success_rate = float(success_count[i]) / num_count[i] if num_count[i] > 0 else 0.0
        print(f'Object {objlist[i]} success rate: {success_rate}')
        fw.write(f'Object {objlist[i]} success rate: {success_rate}\n')
    overall_success = float(sum(success_count)) / sum(num_count) if sum(num_count) > 0 else 0.0
    print(f'ALL success rate: {overall_success}')
    fw.write(f'ALL success rate: {overall_success}\n')
    fw.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()