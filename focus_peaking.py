import cv2
import numpy as np

#TODO:
cfg_overlay_rot = True; #enable drawing of the rule of thirds grid

#resolution of video capture
cfg_capture_width = 1600
cfg_capture_height = 900

#resolution of drawing
cfg_draw_width = 1600
cfg_draw_height = 900

def main():

    #Use Video4linux backend 
    #(default is gstreamer which does not support fourcc settings)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)
    
    #set four character codec to MJPEG 
    #for cheap (~10gbp hdmi capture cards) default is YUYV at max. 5 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    #set device capture resolution
    cap.set(3,cfg_capture_width)
    cap.set(4,cfg_capture_height)
    

    #kernel for edge dilation (make thicker)
    kernel = np.ones((5,5),np.uint8)

    target_dims = (cfg_draw_width, cfg_draw_height)

    while(True):   
        #read frame from device
        frame, src = cap.read()
        
        #==== Focus peak detection ====#
        #convert frame to grayscale
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        #equalize histogram
        gray = cv2.equalizeHist(gray) 

        #blur grayscale image
        blur = cv2.GaussianBlur(gray,(3,3),0)
        
        #run edge detection
        edges = cv2.Canny(gray,200,300)

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
        bg = cv2.bitwise_or(src, src, mask=mask_inv)

        #combine foreground edges with background image
        result = cv2.bitwise_or(fg,bg)
        
        result = cv2.resize(result, target_dims)
        cv2.imshow("end", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
        main()
