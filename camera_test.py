import numpy as np
import cv2, time, math, serial
from collections import defaultdict, deque
from operator import itemgetter

print("Initialised")

#Groups lines based on angle with k-means.
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    rho1 = line1[0]
    theta1 = line1[1]
    rho2 = line2[0]
    theta2 = line2[1]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

from camera import PiVideoStream

cap = PiVideoStream().start()
print("Capture started")
arduino = serial.Serial("/dev/ttyUSB0", 9600, timeout=.1)
print("Serial started")

time.sleep(0.5)

cv2.namedWindow('raw',cv2.WINDOW_NORMAL)
cv2.resizeWindow('raw', 320,240)

arduino.write(("<" + str(0) + "," + str(0) + ">").encode())
print("Motor: All stop")

fps = deque(maxlen=3)
tavg_intersection_x = deque(maxlen=10)
tavg_intersection_y = deque(maxlen=10)
first_frame = True

time.sleep(0.5)
input("Press Enter to continue...")

try:
    while True:
        start = time.time()

        capture = cap.read()
        feed = cv2.pyrDown(capture)

        gray = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY) #for Haar-cascade
        
        rows,cols = gray.shape

        roi_v = int(rows/2) - 2
        raw = feed[roi_v:rows, 0:cols]
        
        frame = cv2.Canny(raw, 35, 75, apertureSize = 3)

        try:
            lines = cv2.HoughLines(frame,1,np.pi/180,35)

            indices = []

            for i, line in enumerate(lines):
                rho,theta = line[0]
                if ( (-0.1<theta<0.1) or (3.1<theta<3.2)or (1.4<theta<1.75) ):
                    indices.append(i)

            lines = np.delete(lines, indices, 0)
            segmented = segment_by_angle_kmeans(lines)

            segmented = segment_by_angle_kmeans(lines)

            try:
                segmented_1 = np.array(segmented[0])
                segmented_1_rho_avg = np.mean(segmented_1[:,0,0])
                segmented_1_theta_avg = np.mean(segmented_1[:,0,1])
                
                segmented_2 = np.array(segmented[1])
                segmented_2_rho_avg = np.mean(segmented_2[:,0,0])
                segmented_2_theta_avg = np.mean(segmented_2[:,0,1])

            except Exception as e:
                print("failed to generate segmented_avg")
                print(e)
            
            intersections = intersection([segmented_1_rho_avg, segmented_1_theta_avg],
                                         [segmented_2_rho_avg, segmented_2_theta_avg])

            a = np.cos(segmented_1_theta_avg)
            b = np.sin(segmented_1_theta_avg)
            x0 = a*segmented_1_rho_avg
            y0 = b*segmented_1_rho_avg
            x1 = int(x0 + 200*(-b))
            y1 = int(y0 + 200*(a))
            x2 = int(x0 - 200*(-b))
            y2 = int(y0 - 200*(a))
            cv2.line(feed,(x1,y1+roi_v),(x2,y2+roi_v),(0,255,0),1)

            a = np.cos(segmented_2_theta_avg)
            b = np.sin(segmented_2_theta_avg)
            x0 = a*segmented_2_rho_avg
            y0 = b*segmented_2_rho_avg
            x1 = int(x0 + 200*(-b))
            y1 = int(y0 + 200*(a))
            x2 = int(x0 - 200*(-b))
            y2 = int(y0 - 200*(a))
            cv2.line(feed,(x1,y1+roi_v),(x2,y2+roi_v),(255,255,0),1)

            tavg_intersection_x.append(intersections[0][0])
            tavg_intersection_y.append(intersections[0][1])

            x_i = int(np.mean(tavg_intersection_x))
            y_i = int(np.mean(tavg_intersection_y))

            cv2.circle(feed, (x_i, y_i+roi_v), 3, (0,0,255), thickness=2)

            origin_x = int(cols/2)
            gradient = (rows-y_i)/(origin_x-x_i)
            angle = math.degrees(math.atan(gradient))

            img_center = np.array([int(cols/2), int(rows/2)])
            cv2.circle(feed, (img_center[0], img_center[1]), 3, (255,0,255), thickness=2)
            
            if angle < 0:
                steer = int(90+angle)
                print("Steer right: " + str(steer))
                if steer>0:
                    if steer>15:
                        if steer>30:
                            arduino.write(("<" + str(6) + "," + str(2) + ">").encode())
                        else:
                            arduino.write(("<" + str(5) + "," + str(2) + ">").encode())
                    else:
                        arduino.write(("<" + str(4) + "," + str(2) + ">").encode())
                else:
                    arduino.write(("<" + str(3) + "," + str(3) + ">").encode())
            else:
                steer = int(90-angle)
                print("Steer left: " + str(steer))
                if steer>2:
                    if steer>15:
                        if steer>30:
                            arduino.write(("<" + str(2) + "," + str(6) + ">").encode())
                        else:
                            arduino.write(("<" + str(2) + "," + str(5) + ">").encode())
                    else:
                        arduino.write(("<" + str(2) + "," + str(4) + ">").encode())
                else:
                    arduino.write(("<" + str(3) + "," + str(3) + ">").encode())

        except KeyboardInterrupt:
            arduino.write(("<" + str(0) + "," + str(0) + ">").encode())
            print("All stop")
            break;

        except Exception as e:
            print("No valid lines found: " + str(e))
            arduino.write(("<" + str(1) + "," + str(1) + ">").encode())

        end = time.time()
        #print("FPS: " + str(int(1/(end-start))))
        fps.append(int(1/(end-start)))
        print("avg_fps: " + str(int(np.mean(fps))))
        
        cv2.imshow('raw',feed)
        #cv2.imshow('int',frame)
        cv2.waitKey(1)

except KeyboardInterrupt:
            arduino.write(("<" + str(0) + "," + str(0) + ">").encode())
            print("All stop")

except Exception as e:
    arduino.write(("<" + str(0) + "," + str(0) + ">").encode())
    print(e)

arduino.write(("<" + str(0) + "," + str(0) + ">").encode())
cv2.destroyAllWindows()
cap.stop()
