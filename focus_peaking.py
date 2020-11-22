import cv2
import numpy as np
import numexpr as ne

def main():
    cap = cv2.VideoCapture(2)
    cap.set(3,1920)
    cap.set(4,1080)

    frame_pre, src_pre = cap.read()
     
    scale_percent = 20 # percent of original size
    scaled_width = int(src_pre.shape[1] * scale_percent / 100)
    scaled_height = int(src_pre.shape[0] * scale_percent / 100)
    scaled_dim = (scaled_width, scaled_height) 
    orig_dim = (src_pre.shape[1], src_pre.shape[0])

    while(True):
        frame, src_raw = cap.read()
        src = src_raw
        #src = cv2.resize(src_raw, scaled_dim, interpolation = cv2.INTER_AREA)
        #gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        #edges = cv2.Canny(gray,150,250)
        
        #edges = cv2.resize(edges, orig_dim, cv2.INTER_AREA)
        #edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        #edges_bgr[edges_bgr==0] = 1
        #img = cv2.multiply(edges_bgr, src_raw)
        
        cv2.imshow("end", src_raw)
        #cv2.imshow("gray", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
        main()
