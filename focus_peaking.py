import cv2
import numpy as np
import numexpr as ne

def main():

    #Use Video4linux backend 
    #(default is gstreamer which does not support fourcc settings)
    cap = cv2.VideoCapture(2, cv2.CAP_V4L)
    
    #set four character codec to MJPEG 
    #for cheap (~10gbp hdmi capture cards) default is YUYV at max. 5 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    #set device capture resolution
    cap.set(3,1920)
    cap.set(4,1080)
    
    #kernel for edge dilation (make thicker)
    kernel = np.ones((5,5),np.uint8)

    while(True):   
        #read frame from device
        frame, src = cap.read()
        
        #==== Focus peak detection ====#
        #convert frame to grayscale
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        #equalize histogram
        gray = cv2.equalizeHist(gray) 

        #blur grayscale image
        blur = cv2.GaussianBlur(gray,(7,7),0)
        
        #run edge detection
        edges = cv2.Canny(gray,50,250)

        #dilate detected edges to make them bigger
        edges = cv2.dilate(edges,kernel,iterations = 1)
        
        #==== Edge overlay on original image ====#
        #Create mask from detected edges
        ret, mask = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)

        #cut out the edges from their background
        fg = cv2.bitwise_or(edges, edges, mask=mask)

        #convert edges foreground to color
        fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)

        #invert edge mask
        mask_inv = cv2.bitwise_not(mask)

        #remove edges foreground from original image
        bg = cv2.bitwise_or(src_raw, src_raw, mask=mask_inv)

        #combine foreground edges with background image
        result = cv2.bitwise_or(fg,bg)
        
        cv2.imshow("end", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
        main()
