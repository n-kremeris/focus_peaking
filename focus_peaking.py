import cv2
import numpy as np

# Grid configuration
cfg_grid_enable = True; #enable drawing of grid
cfg_grid_div = 3 # grid divisions

# Resolution of video capture
cfg_capture_width = 1600
cfg_capture_height = 900

# Resolution of drawing
cfg_draw_width = 1600
cfg_draw_height = 900

# Misc config
cfg_mirror = True # Mirror the image
cfg_fullscreen_enable = True # Fullscreen display of result


class Grid:
    lines = [] 
    x_div = 0
    y_div = 0
    
    # Precalculate required lines in advance, requires target dimensions
    def __init__(self, target_dims, divisor):
        self.target_dims = target_dims
        self.divisor = divisor

        # x and y division length is width and heigth divided by the grid divisor
        # truncate to integer value
        self.x_div = int(target_dims[0] / divisor)
        self.y_div = int(target_dims[1] / divisor)

        #add vertical lines
        for i in range(1, divisor):
            x = self.x_div * i;
            y0 = 0;
            y1 = target_dims[1]
            p1 = (x, y0)
            p2 = (x, y1)
            self.lines.append( (p1, p2))

        #add horizontal lines
        for i in range(1, divisor):
            y = self.y_div * i;
            x0 = 0;
            x1 = target_dims[0]
            p1 = (x0, y)
            p2 = (x1, y)
            self.lines.append( (p1, p2))

    def Draw(self, image):
        for line in self.lines:
            image = cv2.line(image, line[0], line[1], (0, 0, 255), 1, 1)

        return image


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
    
    # resizing detection
    capture_dims = (cfg_capture_width, cfg_capture_height)
    target_dims = (cfg_draw_width, cfg_draw_height)
    resize_f = True

    current_grid = Grid(target_dims, cfg_grid_div)

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
        
        #resize if target dimensions dont match capture dimensions
        if (resize_f):
            result = cv2.resize(result, target_dims)

        #flip image horisontaly if mirroring enabled
        if (cfg_mirror):
            result = cv2.flip(result, 1)
            
        # draw grid
        if (cfg_grid_enable):
            result = current_grid.Draw(result)

        #set result window properties to fullscreen if enabled
        if (cfg_fullscreen_enable):
            cv2.namedWindow("result", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("result",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        # Display result and wait for "q" to quit
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
        main()
