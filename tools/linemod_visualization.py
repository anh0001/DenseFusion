import _init_paths
import argparse
import os
import numpy as np
import torch
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix
import cv2

def draw_axis(img, R, t, K):
    # Camera internals
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    
    # Define axis points in 3D space (larger size for better visibility)
    axis_length = 0.3  # 30cm
    points = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
    
    # Project 3D points to image plane
    points_camera = np.dot(R, points.T).T + t
    points_2d = np.zeros((4,2))
    
    for i in range(4):
        points_2d[i,0] = fx * points_camera[i,0]/points_camera[i,2] + cx
        points_2d[i,1] = fy * points_camera[i,1]/points_camera[i,2] + cy
    
    # Draw the axes
    points_2d = points_2d.astype(int)
    img = cv2.line(img, tuple(points_2d[0]), tuple(points_2d[1]), (0,0,255), 3) # X-axis (Red)
    img = cv2.line(img, tuple(points_2d[0]), tuple(points_2d[2]), (0,255,0), 3) # Y-axis (Green)
    img = cv2.line(img, tuple(points_2d[0]), tuple(points_2d[3]), (255,0,0), 3) # Z-axis (Blue)
    
    return img

def draw_model_points(img, model_points, R, t, K, color=(0, 255, 255)):
    # Project model points to image plane
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    
    # Transform points to camera frame
    points_camera = np.dot(R, model_points.T).T + t
    
    # Project to image plane
    points_2d = np.zeros((len(model_points), 2))
    mask = points_camera[:, 2] > 0  # Only draw points in front of camera
    
    for i in range(len(model_points)):
        if mask[i]:
            points_2d[i,0] = fx * points_camera[i,0]/points_camera[i,2] + cx
            points_2d[i,1] = fy * points_camera[i,1]/points_camera[i,2] + cy
    
    # Draw points
    points_2d = points_2d[mask].astype(int)
    for point in points_2d:
        cv2.circle(img, tuple(point), 2, color, -1)
    
    return img

def draw_bounding_box(img, model_points, R, t, K, color=(0, 255, 0)):
    # Get 3D bounding box
    min_x, max_x = np.min(model_points[:,0]), np.max(model_points[:,0])
    min_y, max_y = np.min(model_points[:,1]), np.max(model_points[:,1])
    min_z, max_z = np.min(model_points[:,2]), np.max(model_points[:,2])
    
    corners = np.array([
        [min_x, min_y, min_z], [max_x, min_y, min_z], [max_x, max_y, min_z], [min_x, max_y, min_z],
        [min_x, min_y, max_z], [max_x, min_y, max_z], [max_x, max_y, max_z], [min_x, max_y, max_z]
    ])
    
    # Project corners to image plane
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    
    corners_camera = np.dot(R, corners.T).T + t
    corners_2d = np.zeros((8,2))
    
    for i in range(8):
        corners_2d[i,0] = fx * corners_camera[i,0]/corners_camera[i,2] + cx
        corners_2d[i,1] = fy * corners_camera[i,1]/corners_camera[i,2] + cy
    
    # Draw bounding box
    corners_2d = corners_2d.astype(int)
    
    # Draw bottom rectangle
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, 2)
    # Draw top rectangle
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i+4]), tuple(corners_2d[((i+1)%4)+4]), color, 2)
    # Draw vertical lines
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, 2)
    
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, required=True, help='dataset root dir')
    parser.add_argument('--model', type=str, required=True, help='resume PoseNet model')
    parser.add_argument('--refine_model', type=str, required=True, help='resume PoseRefineNet model')
    opt = parser.parse_args()

    # Initialize models and dataset
    estimator = PoseNet(num_points=500, num_obj=13)
    estimator.cuda()
    refiner = PoseRefineNet(num_points=500, num_obj=13)
    refiner.cuda()
    estimator.load_state_dict(torch.load(opt.model))
    refiner.load_state_dict(torch.load(opt.refine_model))
    estimator.eval()
    refiner.eval()

    testdataset = PoseDataset_linemod('eval', 500, False, opt.dataset_root, 0.0, True)
    testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=10)

    # Camera matrix
    K = np.array([
        [testdataset.cam_fx, 0, testdataset.cam_cx],
        [0, testdataset.cam_fy, testdataset.cam_cy],
        [0, 0, 1]
    ])

    # Create window with specific size
    cv2.namedWindow('Pose Estimation', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Pose Estimation', 960, 720)

    for i, data in enumerate(testdataloader, 0):
        points, choose, img, target, model_points, idx = data
        if len(points.size()) == 2:
            print('No.{0} NOT Pass! Lost detection!'.format(i))
            continue

        # Move data to GPU
        points = points.cuda()
        choose = choose.cuda()
        img = img.cuda()
        target = target.cuda()
        model_points = model_points.cuda()
        idx = idx.cuda()

        # Get prediction and refine
        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, pred_r.size()[1], 1)
        pred_c = pred_c.view(1, -1)
        how_max, which_max = torch.max(pred_c, 1)
        pred_t = pred_t.view(-1, 1, 3)

        # Get the best prediction
        my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
        my_t = (points.view(-1, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()

        # Refinement iterations
        for _ in range(4):
            my_mat = quaternion_matrix(my_r)
            my_mat[0:3, 3] = my_t
            
            # Move tensor to GPU and handle the transformation
            my_t_cuda = torch.from_numpy(my_t).cuda().float().reshape(1, 1, 3)
            R_cuda = torch.from_numpy(my_mat[:3, :3]).cuda().float()
            new_points = torch.bmm((points - my_t_cuda), R_cuda.view(1, 3, 3)).contiguous()
            
            pred_r, pred_t = refiner(new_points, emb, idx)
            pred_r = pred_r.view(1, 1, -1)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, 1, 1)
            my_r_2 = pred_r.view(-1).cpu().data.numpy()
            my_t_2 = pred_t.view(-1).cpu().data.numpy()
            my_mat_2 = quaternion_matrix(my_r_2)
            my_mat_2[0:3, 3] = my_t_2

            my_mat_final = np.dot(my_mat, my_mat_2)
            my_r = quaternion_from_matrix(my_mat_final, True)
            my_t = my_mat_final[0:3, 3]

        # Prepare image for visualization
        # Convert image from tensor to numpy and denormalize
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = ((img_np * std + mean) * 255).astype(np.uint8)
        img_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Get rotation matrix from quaternion
        R = quaternion_matrix(my_r)[:3, :3]

        # Draw visualizations
        img_vis = draw_axis(img_vis, R, my_t, K)
        
        # Convert model points to the correct scale and draw
        model_points_np = model_points[0].cpu().numpy()
        img_vis = draw_model_points(img_vis, model_points_np, R, my_t, K)
        img_vis = draw_bounding_box(img_vis, model_points_np, R, my_t, K)

        # Add text with object ID
        cv2.putText(img_vis, f'Object {testdataset.objlist[idx[0].item()]}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display visualization
        cv2.imshow('Pose Estimation', img_vis)
        
        key = cv2.waitKey(1000)  # Wait for 10ms
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()