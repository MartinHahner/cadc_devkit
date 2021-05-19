import cv2
import json
import socket
import numpy as np
import load_calibration

from pathlib import Path
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

RED, GREEN, BLUE, YELLOW = [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]

COLORMAP = {'Car': GREEN,
            'Cyclist': RED,
            'Pedestrian': YELLOW}

date = '2019_02_27'
sequence = '0010'
camera = '0'
frame = 26

hostname = socket.gethostname()

if hostname == 'beast':

  BASE = Path(f'/scratch_net/beast_second/mhahner/datasets/CADCD/{date}')

else:

  BASE = Path(f'/srv/beegfs02/scratch/tracezuerich/data/datasets/CADCD/{date}')

########################################################################################################################

image_path = str(BASE / sequence / "labeled" / f"image_0{camera}" / "data" / f"{format(frame, '010')}.png")
lidar_path = str(BASE / sequence / "labeled" / "lidar_points" / "data" / f"{format(frame, '010')}.bin")
calib_path = str(BASE / "calib")

annotations_file = BASE / sequence / "3d_ann.json"

# Load 3d annotations
with open(annotations_file) as f:
    annotations_data = json.load(f)

calib = load_calibration.load_calibration(calib_path)

# Projection matrix from camera to image frame
T_IMG_CAM = np.eye(4)
T_IMG_CAM[0:3,0:3] = np.array(calib['CAM0' + camera]['camera_matrix']['data']).reshape(-1, 3)
T_IMG_CAM = T_IMG_CAM[0:3, :]                                                                 # remove last row

T_CAM_LIDAR = np.linalg.inv(np.array(calib['extrinsics']['T_LIDAR_CAM0' + camera]))
T_IMG_LIDAR = np.matmul(T_IMG_CAM, T_CAM_LIDAR)

img = cv2.imread(image_path)
img_h, img_w = img.shape[:2]

# Add each cuboid to image
for cuboid in annotations_data[frame]['cuboids']:

  T_LIDAR_CUBOID = np.eye(4)
  T_LIDAR_CUBOID[0:3, 0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_dcm()
  T_LIDAR_CUBOID[0][3] = cuboid['position']['x']
  T_LIDAR_CUBOID[1][3] = cuboid['position']['y']
  T_LIDAR_CUBOID[2][3] = cuboid['position']['z']

  #T_Lidar_Cuboid[0][3] = -T_Lidar_Cuboid[0][3]
  # print(cuboid['yaw'])
  # print(T_Lidar_Cuboid)

  label = cuboid['label']
  color = COLORMAP.get(label, BLUE)

  w, l, h = cuboid['dimensions']['x'], cuboid['dimensions']['y'], cuboid['dimensions']['z']

  #########
  # front #
  #########

  front_bottom_right = np.array([[1, 0, 0, l / 2], [0, 1, 0, -w / 2], [0, 0, 1, -h / 2], [0, 0, 0, 1]])
  front_bottom_left = np.array([[1, 0, 0, l / 2], [0, 1, 0, w / 2], [0, 0, 1, -h / 2], [0, 0, 0, 1]])
  front_top_right = np.array([[1, 0, 0, l / 2], [0, 1, 0, -w / 2], [0, 0, 1, h / 2], [0, 0, 0, 1]])
  front_top_left = np.array([[1, 0, 0, l / 2], [0, 1, 0, w / 2], [0, 0, 1, h / 2], [0, 0, 0, 1]])

  f_b_r_ldr = np.matmul(T_LIDAR_CUBOID, front_bottom_right)
  f_b_r_cam = np.matmul(T_CAM_LIDAR, f_b_r_ldr)
  f_b_r_img = np.matmul(T_IMG_CAM, f_b_r_cam)
  f_b_r_crd = (int(f_b_r_img[0][3] / f_b_r_img[2][3]), int(f_b_r_img[1][3] / f_b_r_img[2][3]))

  f_b_l_ldr = np.matmul(T_LIDAR_CUBOID, front_bottom_left)
  f_b_l_cam = np.matmul(T_CAM_LIDAR, f_b_l_ldr)
  f_b_l_img = np.matmul(T_IMG_CAM, f_b_l_cam)
  f_b_l_crd = (int(f_b_l_img[0][3] / f_b_l_img[2][3]), int(f_b_l_img[1][3] / f_b_l_img[2][3]))

  f_t_r_ldr = np.matmul(T_LIDAR_CUBOID, front_top_right)
  f_t_r_cam = np.matmul(T_CAM_LIDAR, f_t_r_ldr)
  f_t_r_img = np.matmul(T_IMG_CAM, f_t_r_cam)
  f_t_r_crd = (int(f_t_r_img[0][3] / f_t_r_img[2][3]), int(f_t_r_img[1][3] / f_t_r_img[2][3]))

  f_t_l_ldr = np.matmul(T_LIDAR_CUBOID, front_top_left)
  f_t_l_cam = np.matmul(T_CAM_LIDAR, f_t_l_ldr)
  f_t_l_img = np.matmul(T_IMG_CAM, f_t_l_cam)
  f_t_l_crd = (int(f_t_l_img[0][3] / f_t_l_img[2][3]), int(f_t_l_img[1][3] / f_t_l_img[2][3]))

  ########
  # back #
  ########

  back_bottom_right = np.array([[1, 0, 0, -l / 2], [0, 1, 0, -w / 2], [0, 0, 1, -h / 2], [0, 0, 0, 1]])
  back_bottom_left = np.array([[1, 0, 0, -l / 2], [0, 1, 0, w / 2], [0, 0, 1, -h / 2], [0, 0, 0, 1]])
  back_top_right = np.array([[1, 0, 0, -l / 2], [0, 1, 0, -w / 2], [0, 0, 1, h / 2], [0, 0, 0, 1]])
  back_top_left = np.array([[1, 0, 0, -l / 2], [0, 1, 0, w / 2], [0, 0, 1, h / 2], [0, 0, 0, 1]])

  b_b_r_ldr = np.matmul(T_LIDAR_CUBOID, back_bottom_right)
  b_b_r_cam = np.matmul(T_CAM_LIDAR, b_b_r_ldr)
  b_b_r_img = np.matmul(T_IMG_CAM, b_b_r_cam)
  b_b_r_crd = (int(b_b_r_img[0][3] / b_b_r_img[2][3]), int(b_b_r_img[1][3] / b_b_r_img[2][3]))

  b_b_l_ldr = np.matmul(T_LIDAR_CUBOID, back_bottom_left)
  b_b_l_cam = np.matmul(T_CAM_LIDAR, b_b_l_ldr)
  b_b_l_img = np.matmul(T_IMG_CAM, b_b_l_cam)
  b_b_l_crd = (int(b_b_l_img[0][3] / b_b_l_img[2][3]), int(b_b_l_img[1][3] / b_b_l_img[2][3]))

  b_t_r_ldr = np.matmul(T_LIDAR_CUBOID, back_top_right)
  b_t_r_cam = np.matmul(T_CAM_LIDAR, b_t_r_ldr)
  b_t_r_img = np.matmul(T_IMG_CAM, b_t_r_cam)
  b_t_r_crd = (int(b_t_r_img[0][3] / b_t_r_img[2][3]), int(b_t_r_img[1][3] / b_t_r_img[2][3]))

  b_t_l_ldr = np.matmul(T_LIDAR_CUBOID, back_top_left)
  b_t_l_cam = np.matmul(T_CAM_LIDAR, b_t_l_ldr)
  b_t_l_img = np.matmul(T_IMG_CAM, b_t_l_cam)
  b_t_l_crd = (int(b_t_l_img[0][3] / b_t_l_img[2][3]), int(b_t_l_img[1][3] / b_t_l_img[2][3]))

  # perform checks
  valid = True

  cam_points = [f_b_r_cam, f_b_l_cam, f_t_r_cam, f_t_l_cam,
                b_b_r_cam, b_b_l_cam, b_t_r_cam, b_t_l_cam]

  crd_points = [f_b_r_crd, f_b_l_crd, f_t_r_crd, f_t_l_crd,
                b_b_r_crd, b_b_l_crd, b_t_r_crd, b_t_l_crd]

  # check if behind camera
  for cam_point in cam_points:
    if cam_point[2][3] < 0:
      valid = False

  # check if outside image
  for crd_point in crd_points:
    if crd_point[0] < 0 or crd_point[0] > img_w or crd_point[1] < 0 or crd_point[1] > img_h:
      valid = False

  if not valid:
    continue

  # draw cuboid center
  ctr_cam = np.matmul(T_CAM_LIDAR, T_LIDAR_CUBOID)
  ctr_img = np.matmul(T_IMG_CAM, ctr_cam)
  x = int(ctr_img[0][3]/ctr_img[2][3])
  y = int(ctr_img[1][3]/ctr_img[2][3])
  cv2.circle(img, (x,y), radius=3, color=color, thickness=2, lineType=8, shift=0)

  # draw 12 individual lines

  # front
  cv2.line(img, f_b_r_crd, f_t_r_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, f_b_r_crd, f_b_l_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, f_b_l_crd, f_t_l_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, f_t_l_crd, f_t_r_crd, GREEN, thickness=2, lineType=8, shift=0)
  # back
  cv2.line(img, b_b_r_crd, b_t_r_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, b_b_r_crd, b_b_l_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, b_b_l_crd, b_t_l_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, b_t_l_crd, b_t_r_crd, GREEN, thickness=2, lineType=8, shift=0)
  # connect front to back
  cv2.line(img, f_b_r_crd, b_b_r_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, f_t_r_crd, b_t_r_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, f_b_l_crd, b_b_l_crd, GREEN, thickness=2, lineType=8, shift=0)
  cv2.line(img, f_t_l_crd, b_t_l_crd, GREEN, thickness=2, lineType=8, shift=0)


plt.title('/'.join(image_path.split('/')[-6:]))
plt.imshow(img)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
