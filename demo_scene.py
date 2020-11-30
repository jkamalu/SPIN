import os
import json
import argparse

import cv2
import numpy as np

import torch
from torchvision.transforms import Normalize

from models import hmr, SMPL
from utils.imutils import crop
from utils.renderer import Renderer
import config
import constants


DIMENSION = 224
ORIG_W = 3760
ORIG_H = 480


def bbox_(bbox):
    bbox = np.array(bbox).astype(np.float32)
    center = bbox[:2] + 0.5 * bbox[2:]
    scale = max(bbox[2:]) / 200.0
    return center, scale


def convert_crop_cam_to_orig_img(cam, bbox, orig_w, orig_h):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    bbox_center_x, bbox_center_y, bbox_scale = bbox[:,0], bbox[:,1], bbox[:,2]
    
    orig_center_x, orig_center_y = orig_w / 2., orig_h / 2.
        
    crop_camera_sx, crop_camera_sy, crop_camera_tx, crop_camera_ty = cam[:, 0], cam[:, 1], cam[:, 2], cam[:, 3]
    
    sx = crop_camera_sx * (bbox_scale / orig_w) * 200.
    
    sy = crop_camera_sy * (bbox_scale / orig_h) * 200.
    
    tx = ((bbox_center_x - orig_center_x) / orig_center_x / sx) + crop_camera_tx
    
    ty = ((bbox_center_y - orig_center_y) / orig_center_y / sy) + crop_camera_ty
    
    orig_cam = np.stack([sx, sy, tx, ty]).T
    
    return orig_cam


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--imgdir', type=str, required=True)
    parser.add_argument('--detections', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    
    args = parser.parse_args()
    
    os.system(f"mkdir -p {args.logdir}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        assert torch.cuda.is_available(), "torch.cuda.is_available evaluates to False"
    
    # Load pretrained model
    model = hmr(config.SMPL_MEAN_PARAMS).to(device)
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)

    # Load SMPL model
    smpl = SMPL(config.SMPL_MODEL_DIR,
                batch_size=1,
                create_transl=False).to(device)
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer(smpl.faces, resolution=(ORIG_W, ORIG_H), orig_img=True, wireframe=True)

    with open(args.detections, "rb") as reader:
        detections = json.load(reader)

    normalizer = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
    for fname in detections:
        
        image = cv2.imread(os.path.join(args.imgdir, fname))[:, :, ::-1].copy()
        
        image_copy = image.copy()

        for center, scale in map(bbox_, detections[fname]):
                        
            image_cropped = crop(image, center, scale, (DIMENSION, DIMENSION)).astype(np.float32) / 255.
            image_cropped = torch.from_numpy(image_cropped).permute(2, 0, 1)
            image_cropped_normalized = normalizer(image_cropped.clone()).unsqueeze(0)
            
            cv2.imwrite(os.path.join(args.logdir, "test.jpg"), 255 * image_cropped_normalized.cpu().numpy())
        
            # Preprocess input image and generate predictions
            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = model(image_cropped_normalized.to(device))
                pred_output = smpl(
                    betas=pred_betas, 
                    body_pose=pred_rotmat[:, 1:], 
                    global_orient=pred_rotmat[:, 0].unsqueeze(1), 
                    pose2rot=False
                )
                pred_vertices = pred_output.vertices
                                
                s, tx, ty = pred_camera.cpu().numpy()[0]
                
                camera = convert_crop_cam_to_orig_img(
                    cam=np.array([[s, s, tx, ty]]),
                    bbox=np.array([*center, scale])[None, :],
                    orig_w=ORIG_W,
                    orig_h=ORIG_H
                )
                                
                camera = camera[0]
                pred_vertices = pred_vertices[0].cpu().numpy()
                
                # Render parametric shape
                image_copy = renderer.render(
                    image_copy,
                    pred_vertices,
                    cam=camera
                )
            
        # Save reconstructions
        cv2.imwrite(os.path.join(args.logdir, fname), image_copy[:,:,::-1])
