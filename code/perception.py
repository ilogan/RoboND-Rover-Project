import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))

    return warped, mask

# Function to create mask for yellow rocks
def find_rocks(img, hue=96):
    # convert BGR to HSV
    frame = img
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define upper and lower bound for identifying yellow color
    lower_yellow = np.array([hue-10,130,130])
    upper_yellow = np.array([hue+10,255,255])

    # threshold for only yellow colors
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #convert to binary image
    rockpix1 = (mask>0)
    mask[rockpix1] = 1

    return mask


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    img = Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 142], [302 ,142],[200, 96], [120, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                      [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                      [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                      ])
    # 2) Apply perspective transform
    warped, mask = perspect_transform(img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navi_thresh = color_thresh(warped) # Navigable Terrain
    rock_thresh = find_rocks(warped) # Rock Samples
    obs_thresh = np.absolute(np.float32(navi_thresh)-1) * mask # Obstacles
                                                               # (reverse warp_thresh values and
                                                               # multiply by mask to identify obstacles
                                                               #in view of rover camera)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,2] = navi_thresh * 255
    Rover.vision_image[:,:,0] = obs_thresh * 255

    # 5) Convert map image pixel values to rover-centric coords
    navi_xpix, navi_ypix = rover_coords(navi_thresh)
    obs_xpix, obs_ypix = rover_coords(obs_thresh)

    # 6) Convert rover-centric pixel values to world coordinates
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size

    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw

    navi_xworld, navi_yworld = pix_to_world(navi_xpix, navi_ypix, xpos, ypos, yaw, world_size, scale)
    obs_xworld, obs_yworld = pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[navi_yworld, navi_xworld, 2] += 10
    Rover.worldmap[obs_yworld, obs_xworld, 0] += 1

    if rock_thresh.any():
        rock_xpix, rock_ypix = rover_coords(rock_thresh)
        rock_dist, rock_ang = to_polar_coords(rock_xpix, rock_ypix)
        rock_index = np.argmin(rock_dist)
        rock_xworld, rock_yworld = pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
        rock_xcen = rock_xworld[rock_index]
        rock_ycen = rock_yworld[rock_index]

        Rover.worldmap[rock_ycen, rock_xcen, 1] = 255

        Rover.vision_image[:,:,1] = rock_thresh * 255
    else:
        Rover.vision_image[:,:,1] = 0

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    dist, angles = to_polar_coords(navi_xpix, navi_ypix)
    Rover.nav_dists  = dist
    Rover.nav_angles = angles

    return Rover
