import numpy as np
import cv2, time, math, serial
from collections import defaultdict
from operator import itemgetter

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

'''#Finds the intersection of two lines given in Hesse normal form.
def intersection(line1, line2):
    print("finding individual intersection")
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    print("finding group intersection")
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections'''

def intersection(line1, line2):
    print("finding individual intersection")
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
arduino = serial.Serial("/dev/ttyUSB0", 9600, timeout=.1)

time.sleep(1)

cv2.namedWindow('raw',cv2.WINDOW_NORMAL)
cv2.resizeWindow('raw', 320,130)

print("all stop")
arduino.write(("<" + str(0) + "," + str(0) + ">").encode())

fps = []

time.sleep(0.5)
input("Press Enter to continue...")

try:
    while True:
        start = time.time()
        
        feed = cap.read()
        raw = cv2.pyrDown(feed)
        
        rows,cols, bgr = raw.shape
        #M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
        #raw = cv2.warpAffine(raw,M,(cols,rows))

        roi_v = int(rows/2) - 2
        raw = raw[roi_v:rows, 0:cols]
        
        frame = cv2.Canny(raw, 30, 75, apertureSize = 3)

        try:
            lines = cv2.HoughLines(frame,1,np.pi/180,55)

            indices = []

            for i, line in enumerate(lines):
                rho,theta = line[0]
                if ( theta == 0 or theta == math.radians(180) or (1.4<theta<1.7) ):
                    indices.append(i)

            lines = np.delete(lines, indices, 0)
            segmented = segment_by_angle_kmeans(lines)

            '''for line in segmented[0]:
                rho,theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 200*(-b))
                y1 = int(y0 + 200*(a))
                x2 = int(x0 - 200*(-b))
                y2 = int(y0 - 200*(a))
                cv2.line(raw,(x1,y1),(x2,y2),(0,255,0),1)

            for line in segmented[1]:
                rho,theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 200*(-b))
                y1 = int(y0 + 200*(a))
                x2 = int(x0 - 200*(-b))
                y2 = int(y0 - 200*(a))
                cv2.line(raw,(x1,y1),(x2,y2),(255,255,0),1)'''

            segmented = segment_by_angle_kmeans(lines)

            try:
                segmented_1 = np.array(segmented[0])
                segmented_1_rho_avg = np.mean(segmented_1[:,0,0])
                segmented_1_theta_avg = np.mean(segmented_1[:,0,1])
                #segmented_1_avg = np.array([segmented_1_rho_avg, segmented_1_theta_avg])
                
                segmented_2 = np.array(segmented[1])
                segmented_2_rho_avg = np.mean(segmented_2[:,0,0])
                segmented_2_theta_avg = np.mean(segmented_2[:,0,1])
                #segmented_2_avg = np.array([segmented_2_rho_avg, segmented_2_theta_avg])

                segmented_avg = np.array([[segmented_1_rho_avg, segmented_1_theta_avg],
                                          [segmented_2_rho_avg, segmented_2_theta_avg]])
            except Exception as e:
                print("failed to generate segmented_avg")
                print(e)

            #print(segmented_avg)

            #avg_segmented_1 = np.array([np.average(segmented[0][0][0]), np.average(segmented[0][0][1])]
            #avg_segmented_2 = np.array([np.average(segmented[0][1][0]), np.average(segmented[0][1][1])]

            #avg_segmented = np.array([  , [ , ] ])

            #intersections = segmented_intersections(segmented)

            print("attempting intersections")
            #intersections = segmented_intersections(segmented_avg)
            
            intersections = intersection([segmented_1_rho_avg, segmented_1_theta_avg],
                                         [segmented_2_rho_avg, segmented_2_theta_avg])

            

            print(intersections)
            
            #x_i = int(np.average(intersections[0][0][0]))
            #y_i = int(np.average(intersections[0][0][1]))

            x_i = intersections[0][0]
            y_i = intersections[0][1]

            print(str((x_i, y_i)))

            cv2.circle(raw, (x_i, y_i), 3, (0,0,255), thickness=2)

            origin_x = int(cols/2)
            gradient = (rows-y_i)/(origin_x-x_i)
            angle = math.degrees(math.atan(gradient))
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

        except Exception as e:
            print("No valid lines found: " + str(e))
            arduino.write(("<" + str(2) + "," + str(2) + ">").encode())

        end = time.time()
        #print("FPS: " + str(int(1/(end-start))))
        fps.append(int(1/(end-start)))
        print("avg_fps: " + str(int(np.sum(fps) / len(fps))))
        
        cv2.imshow('raw',raw)
        cv2.waitKey(1)

except:
    arduino.write(("<" + str(0) + "," + str(0) + ">").encode())

arduino.write(("<" + str(0) + "," + str(0) + ">").encode())
cv2.destroyAllWindows()
cap.stop()
