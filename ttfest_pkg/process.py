import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import calib, undistort
from threshold import get_combined_gradients, get_combined_hls, combine_grad_hls
from line import Line, get_perspective_transform, get_lane_lines_img

import time
from pid_controller import PIDController


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#       Select desired input name/type          #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#input_type = 'image'
#input_type = 'video' 
input_type = 'frame_by_frame'

input_name = 'drive.mp4'
#input_name = 'test_images/calibration1.jpg'
#input_name = 'project_video.mp4' 
#input_name = 'challenge_video.mp4'
#input_name = 'harder_challenge_video.mp4'

# If input_type is `image`, select whether you'd like to save intermediate images or not. 
save_img = True

left_line = Line()
right_line = Line()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Tune Parameters for different inputs        #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
th_sobelx, th_sobely, th_mag, th_dir = (35, 100), (30, 255), (30, 255), (0.7, 1.3)
th_h, th_l, th_s = (10, 100), (0, 60), (85, 255)

# camera matrix & distortion coefficient
mtx, dist = calib()
ptime = 0
ctime = 0

Kp = 0.3
Ki = 0.2
Kd = 0.1

center_car = 320
setpoint = center_car

def pipeline(frame):
    # Correcting for Distortion
    undist_img = undistort(frame, mtx, dist)
    
    # resize video
    undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
    rows, cols = undist_img.shape[:2]

    combined_gradient = get_combined_gradients(undist_img, th_sobelx, th_sobely, th_mag, th_dir)

    combined_hls = get_combined_hls(undist_img, th_h, th_l, th_s)

    combined_result = combine_grad_hls(combined_gradient, combined_hls)

    c_rows, c_cols = combined_result.shape[:2]
    s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
    s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

    warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))

    searching_img = get_lane_lines_img(warp_img, left_line, right_line)

if __name__ == '__main__':

    # For debugging Frame by Frame, using cv2.imshow()
    if input_type == 'frame_by_frame':
        cap = cv2.VideoCapture(input_name)
        
        frame_num = -1 

        while (cap.isOpened()):
            ctime = time.time()
            _, frame = cap.read()
            
            frame_num += 1   # increment frame_num, used for naming saved images 
            pid_controller = PIDController(Kp, Ki, Kd, setpoint)
            # Correcting for Distortion
            undist_img = undistort(frame, mtx, dist)
            # resize video
            undist_img = cv2.resize(undist_img, None, fx=1 / 2, fy=1 / 2, interpolation=cv2.INTER_AREA)
            rows, cols = undist_img.shape[:2]

            combined_gradient = get_combined_gradients(undist_img, th_sobelx, th_sobely, th_mag, th_dir)

            combined_hls = get_combined_hls(undist_img, th_h, th_l, th_s)

            combined_result = combine_grad_hls(combined_gradient, combined_hls)

            c_rows, c_cols = combined_result.shape[:2]
            s_LTop2, s_RTop2 = [c_cols / 2 - 24, 5], [c_cols / 2 + 24, 5]
            s_LBot2, s_RBot2 = [110, c_rows], [c_cols - 110, c_rows]

            src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
            dst = np.float32([(170, 720), (170, 0), (550, 0), (550, 720)])

            warp_img, M, Minv = get_perspective_transform(combined_result, src, dst, (720, 720))

            searching_img = get_lane_lines_img(warp_img, left_line, right_line)


            

            # Drawing the lines back down onto the road
            
            lane_color = np.zeros_like(undist_img)
            

            # Combine the result with the original image
            result = cv2.addWeighted(undist_img, 1, lane_color, 0.3, 0)
            
            current_position = (left_line.startx + right_line.startx) / 2

            error = setpoint - current_position
            
            steering_angle = pid_controller.update(error)
            print(error,steering_angle)

            # print(undist_img.shape)

            cv2.line(result,(int(setpoint),3*result.shape[0]//4),(int(setpoint-error),3*result.shape[0]//4),(0,255,0),5)
            cv2.line(result,(int(setpoint),3*result.shape[0]//4),(int(setpoint-steering_angle),3*result.shape[0]//4),(0,0,255),5)
            # if error > 38:
            #    print("Sola gidiyor")
            # elif error < 20:
            #     print("Sağa gidiyor")
            # elif error > 20 and error < 38:
            #     print("Düz gidiyor")

            fps=1/(ctime-ptime)
            ptime=ctime
            fps= int(fps)
            fps = str(fps)
            cv2.putText(result, fps, (7,70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100,255,0), 3, cv2.LINE_AA)
            
            cv2.imshow('road info', result)
      
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # If working with video mode, use moviepy and process each frame and save the video.
    