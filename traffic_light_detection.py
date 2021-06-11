#!/usr/bin/env python3

import numpy as np
from math import cos, sin, pi,tan
from cutils import CUtils
import logging

#   Required to import carla library
import os
import sys
sys.path.append(os.path.abspath(sys.path[0] + '/traffic_light_detection_module'))
from yolo import YOLO

# Utils : X - Rotation
def rotate_x(angle):
    R = np.mat([[ 1,         0,           0],
                 [ 0, cos(angle), -sin(angle) ],
                 [ 0, sin(angle),  cos(angle) ]])
    return R

# Utils : Y - Rotation
def rotate_y(angle):
    R = np.mat([[ cos(angle), 0,  sin(angle) ],
                 [ 0,         1,          0 ],
                 [-sin(angle), 0,  cos(angle) ]])
    return R

# Utils : Z - Rotation
def rotate_z(angle):
    R = np.mat([[ cos(angle), -sin(angle), 0 ],
                 [ sin(angle),  cos(angle), 0 ],
                 [         0,          0, 1 ]])
    return R

# Utils : Rotation - XYZ
def to_rot(r):
    Rx = np.mat([[ 1,         0,           0],
                 [ 0, cos(r[0]), -sin(r[0]) ],
                 [ 0, sin(r[0]),  cos(r[0]) ]])

    Ry = np.mat([[ cos(r[1]), 0,  sin(r[1]) ],
                 [ 0,         1,          0 ],
                 [-sin(r[1]), 0,  cos(r[1]) ]])

    Rz = np.mat([[ cos(r[2]), -sin(r[2]), 0 ],
                 [ sin(r[2]),  cos(r[2]), 0 ],
                 [         0,          0, 1 ]])

    return Rz*Ry*Rx

class TrafficLightDetection:
    def __init__(self, camera_parameters, config):
        # Constants
        self.NUM_SEMAPHORE_CHECKS = 13
        self.SCORE_THRESHOLD = 0.20
        
        #Detector
        self.traffic_light_detector = YOLO(config)

        # Detection parameters
        self.prev_semaphore_box = None
        self.count_semaphore_detections = 0
        self.count_missdetection = 0
        self._num_go = 0
        self._num_stop = 0
        self.th = int(0.10 * camera_parameters["width"])

        # Camera parameters
        self.cam_height = camera_parameters['z']
        self.cam_x_pos = camera_parameters['x']
        self.cam_y_pos = camera_parameters['y']
        
        self.camera_width = camera_parameters['width']
        self.camera_height = camera_parameters['height']

        self.camera_fov = camera_parameters['fov']

        # Calculate Intrinsic Matrix
        f = self.camera_width /(2 * tan(self.camera_fov * pi / 360))
        Center_X = self.camera_width / 2.0
        Center_Y = self.camera_height / 2.0

        intrinsic_matrix = np.array([[f, 0, Center_X],
                                     [0, f, Center_Y],
                                     [0, 0, 1]])
                                      
        self.inv_intrinsic_matrix = np.linalg.inv(intrinsic_matrix)

        # Rotation matrix to align image frame to camera frame
        rotation_image_camera_frame = np.dot(rotate_z(-90 * pi /180),rotate_x(-90 * pi /180))

        image_camera_frame = np.zeros((4,4))
        image_camera_frame[:3,:3] = rotation_image_camera_frame
        image_camera_frame[:, -1] = [0, 0, 0, 1]

        # Lambda Function for transformation of image frame in camera frame 
        self.image_to_camera_frame = lambda object_camera_frame: np.dot(image_camera_frame, object_camera_frame)

    
    def detect(self, image):
        boxes = self.traffic_light_detector.predict(image)
        if len(boxes) > 0:
            current_box = boxes[0]

            if current_box.get_score() > self.SCORE_THRESHOLD:
                self.count_missdetection = 0
                # First time semaphore is detected
                if self.prev_semaphore_box == None:
                    self.prev_semaphore_box = current_box
                    self.count_semaphore_detections = 1
                    if current_box.get_label() == 0:
                        self._num_go = 1
                    else:
                        self._num_stop =1
                else:
                    # Check if the boxes refer to the same object
                    xmin_diff = abs(self.camera_width * current_box.xmin - self.camera_width * self.prev_semaphore_box.xmin)
                    xmax_diff = abs(self.camera_width * current_box.xmax - self.camera_width * self.prev_semaphore_box.xmax)
                    ymin_diff = abs(self.camera_width * current_box.ymin - self.camera_width * self.prev_semaphore_box.ymin)
                    ymax_diff = abs(self.camera_width * current_box.ymax - self.camera_width * self.prev_semaphore_box.ymax)

                    if xmin_diff < self.th and xmax_diff < self.th and ymin_diff < self.th and ymax_diff < self.th:
                        # the two boxes refer to the same object
                        self.count_semaphore_detections += 1
                        if current_box.get_label() == 0:
                            self._num_go += 1
                        else:
                            self._num_stop +=1
                        self.prev_semaphore_box = current_box
                    else:

                        logging.debug("Threshold violata")
                        self.prev_semaphore_box = None
                        self.count_semaphore_detections = 0
                        self._num_go = 0
                        self._num_stop = 0
            else:
                self.count_missdetection += 1
        elif self.prev_semaphore_box is not None:
            self.count_missdetection += 1

        if self.count_missdetection == int(0.3*self.NUM_SEMAPHORE_CHECKS):
                
                logging.debug("Missdetection")
                self.prev_semaphore_box = None
                self.count_semaphore_detections = 0      
                self.count_missdetection = 0
                self._num_go = 0
                self._num_stop = 0
                return False, boxes, True

        if self.count_semaphore_detections == self.NUM_SEMAPHORE_CHECKS or self._num_go >= int(self.NUM_SEMAPHORE_CHECKS/2)+1 or self._num_stop >= int(self.NUM_SEMAPHORE_CHECKS/2)+1:
            is_green = self._num_go > self._num_stop
            self.count_semaphore_detections = 0
            self.count_missdetection = 0
            self._num_go = 0
            self._num_stop = 0
            return True, boxes, is_green
        return False, boxes, False
           

    def get_traffic_light_fences(self, depth_data, current_x, current_y, current_yaw):
        traffic_light_fences = []     # [x0, y0, x1, y1]
        if self.prev_semaphore_box is not None:
            xmin = self.camera_width*self.prev_semaphore_box.xmin
            ymin = self.camera_height*self.prev_semaphore_box.ymin
            xmax = self.camera_width*self.prev_semaphore_box.xmax
            ymax = self.camera_height*self.prev_semaphore_box.ymax
            prop = self.prev_semaphore_box.xmin
            
            # 45:1=alfa:prop   -> alfa = prop*90 - 45 

            w = (xmax-xmin)
            xmin = xmin - 4*w
            xmax = xmax + 4*w
            
            ymin = ymin - 2*w
            ymax = ymax - 2*w

            # From pixel to waypoint
            depth = 1000 #Distance of the sky
            for i in range(int(xmin), int(xmax+1)):
                for j in range(int(ymin), int(ymax+1)):
                    if j < self.camera_height and i < self.camera_width:
                        if depth > depth_data[j][i]:
                            # Projection Pixel to Image Frame
                            y = j
                            x = i
                            depth = depth_data[y][x]# Consider depth in meters


            # From pixel to waypoint
            pixel = [x , y, 1]
            pixel = np.reshape(pixel, (3,1))
            
            logging.debug("PROP %s", str(prop))
            # Projection Pixel to Image Frame
            depth = depth_data[y][x] * 1000  # Consider depth in meters  

            logging.debug("Depth prima %s", str(depth))
            if depth != 1000.0 and prop > 0.5:
                alpha = prop * 90 - 45
                alpha = alpha / 180 * pi
                depth = depth * cos(alpha)
                
                logging.debug("Depth dopo %s", str(depth))
                image_frame_vect = np.dot(self.inv_intrinsic_matrix, pixel) * depth
                
                # Create extended vector
                image_frame_vect_extended = np.zeros((4,1))
                image_frame_vect_extended[:3] = image_frame_vect 
                image_frame_vect_extended[-1] = 1
                
                # Projection Camera to Vehicle Frame
                camera_frame = self.image_to_camera_frame(image_frame_vect_extended)
                camera_frame = camera_frame[:3]
                camera_frame = np.asarray(np.reshape(camera_frame, (1,3)))

                camera_frame_extended = np.zeros((4,1))
                camera_frame_extended[:3] = camera_frame.T 
                camera_frame_extended[-1] = 1

                camera_to_vehicle_frame = np.zeros((4,4))
                camera_to_vehicle_frame[:3,:3] = to_rot([0, 0, 0])
                camera_to_vehicle_frame[:,-1] = [self.cam_x_pos, self.cam_y_pos, self.cam_height, 1]

                vehicle_frame = np.dot(camera_to_vehicle_frame, camera_frame_extended)
                vehicle_frame = vehicle_frame[:3]
                vehicle_frame = np.asarray(np.reshape(vehicle_frame, (1,3)))

                stopsign_data = CUtils()
                if (int(round(abs(cos(current_yaw))))):
                    stopsign_data.create_var('x', vehicle_frame[0][0]-self.cam_x_pos)
                    stopsign_data.create_var('y', vehicle_frame[0][1])
                else:
                    stopsign_data.create_var('x', vehicle_frame[0][1])
                    stopsign_data.create_var('y', vehicle_frame[0][0]-self.cam_x_pos)
                stopsign_data.create_var('z', vehicle_frame[0][2])

                # obtain stop sign fence points for LP
                x = stopsign_data.x
                y = stopsign_data.y

                spos = np.array([
                        [current_x-5*int(round(abs(sin(current_yaw)))), current_x+5*int(round(abs(sin(current_yaw))))],
                        [current_y-5*int(round(abs(cos(current_yaw)))), current_y+5*int(round(abs(cos(current_yaw))))]])
                spos_shift = np.array([
                        [x, x],
                        [y, y]])
                before = np.array([
                        [5*int(round(abs(cos(current_yaw)))),5*int(round(abs(cos(current_yaw))))],
                        [5*int(round(abs(sin(current_yaw)))),5*int(round(abs(sin(current_yaw))))]])

                if np.sign(round(np.cos(current_yaw))) > 0: #mi sto muovendo lungo le x positive, verso destra
                    spos = np.add(spos, spos_shift)
                    
                elif np.sign(round(np.cos(current_yaw))) < 0: #mi sto muovendo lungo le x negative, verso sinistra
                    spos = np.subtract(spos, spos_shift)
                    
                else:
                    if np.sign(round(np.sin(current_yaw))) > 0: #mi sto muovendo lungo le y positive, verso il basso
                        spos = np.add(spos, spos_shift)
                    else:
                        spos = np.subtract(spos, spos_shift)

                traffic_light_fences.append([spos[0,0], spos[1,0], spos[0,1], spos[1,1]])
                self.prev_semaphore_box = None

        logging.debug("Fence calcolata: %s", str(traffic_light_fences))
        return traffic_light_fences
