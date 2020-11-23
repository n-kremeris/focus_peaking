import cv2
import numpy as np

# focus peaking configuration
cfg_focuspeaking_enable = True

# Grid configuration
cfg_grid_enable = True; #enable drawing of grid
cfg_grid_div = 3 # grid divisions
cfg_grid_thiccness = 3 # grid line width

# Resolution of video capture
cfg_capture_width = 1600
cfg_capture_height = 900

# Resolution of drawing
cfg_draw_width = 1600
cfg_draw_height = 900

# Misc config
cfg_mirror_enable = True # Mirror the image
cfg_fullscreen_enable = True # Fullscreen display of image


#==== Class to draw a grid on the output frame ====#
class Grid:
    lines = []
    x_div = 0
    y_div = 0

    # Precalculate required lines in advance, requires target dimensions
    def __init__(self, target_dims, divisor, thickness):
        self.target_dims = target_dims
        self.divisor = divisor
        self.thickness = thickness

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
            image = cv2.line(image, line[0], line[1], (0, 0, 255), self.thickness, 1)

        return image


#==== Focus peak detection ====#
class FocusPeakDetector:
    def __init__(self, th1, th2, blur_size, dilate_size, dilate_iter):
        self.th1 = th1
        self.th2 = th2
        self.blur_kernel = (blur_size, blur_size)
        self.dilate_kernel = np.ones((dilate_size, dilate_size),np.uint8) #kernel for dilate size (makes lines thicker)
        self.dilate_iter = dilate_iter #number of iterations to run dilation

    def Detect(self, image):
        #convert frame to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #equalize histogram (contrast stretch)
        gray = cv2.equalizeHist(gray)

        #blur grayscale image
        blur = cv2.GaussianBlur(gray,self.blur_kernel,0)

        #run edge detection
        edges = cv2.Canny(gray, self.th1, self.th2)

        #dilate detected edges to make them bigger
        edges = cv2.dilate(edges, self.dilate_kernel, iterations = self.dilate_iter)

        #create binary mask of detected edges
        ret, mask = cv2.threshold(edges, 10, 255, cv2.THRESH_BINARY)

        return (mask, edges)


#==== Overlay fb over bg with given mask ====#
def overlay(fg, bg, mask):

    #cut out the edges from their background
    fg = cv2.bitwise_or(fg, fg, mask=mask)

    #convert edges foreground to color
    fg = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)

    #invert edge mask
    mask_inv = cv2.bitwise_not(mask)

    #remove foreground from background
    bg = cv2.bitwise_or(bg, bg, mask=mask_inv)

    #combine foreground edges with background image
    return cv2.bitwise_or(fg,bg)


def main():
    global cfg_focuspeaking_enable
    global cfg_mirror_enable

    #Use Video4linux backend
    #(default is gstreamer which does not support fourcc settings)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L)

    #set four character codec to MJPEG
    #for cheap (~10gbp hdmi capture cards) default is YUYV at max. 5 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    #set device capture resolution
    cap.set(3,cfg_capture_width)
    cap.set(4,cfg_capture_height)

    # resizing detection
    capture_dims = (cfg_capture_width, cfg_capture_height)
    target_dims = (cfg_draw_width, cfg_draw_height)
    resize_f = (capture_dims == target_dims)

    grid = Grid(target_dims, cfg_grid_div, cfg_grid_thiccness)
    fpdetector = FocusPeakDetector(200, 300, 3, 3, 1)

    while(True):
        # Read frame from device
        frame, image = cap.read()

        if (cfg_focuspeaking_enable):
            # Detect focus peaks
            mask, peaks = fpdetector.Detect(image)
            # Overlay focus peaks on original image
            image = overlay(peaks, image, mask)

        # Resize if target dimensions dont match capture dimensions
        if (resize_f):
            image = cv2.resize(image, target_dims)

        # Flip image horisontaly if mirroring enabled
        if (cfg_mirror_enable):
            image = cv2.flip(image, 1)

        # Draw grid
        if (cfg_grid_enable):
            image = grid.Draw(image)

        # Set image window properties to fullscreen if enabled
        if (cfg_fullscreen_enable):
            cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        # Display image and wait for "q" to quit, or other options
        cv2.imshow("image", image)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

        if key & 0xff == ord('f'):
            cfg_focuspeaking_enable = not cfg_focuspeaking_enable

        if key & 0xff == ord('m'):
            cfg_mirror_enable = not cfg_mirror_enable


if __name__ == "__main__":
        main()
