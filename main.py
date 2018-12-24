import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import pickle
import argparse
# docstring based on this https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html


def get_points(image, n=4):
    """Uses ginput to take points as input.
    Args:
        image (numpy.ndarray) : image to take input from.
        n (int) : number of points to take.

    Returns:
        `list` of `tuples` : The `n` points sampled from the image.
    """

    plt.imshow(image)
    points = plt.ginput(n, timeout=-100)
    plt.show()
    plt.close()
    return points

def compute_points_mat(src_points, target_points):
    """
        Transforms a list of tuples - where src_point[i] and
        target_points[i] correspond to the same feature viewed 
        in 2 images - into a format suitable for solving for the 
        homography matrix.

        Args:
            src_points(list of tuples) : points from the first image.
            target_points(list of tuples) : points from the second image. 

        Returns:
            numpy.ndarray: matrix of shape (2 * number of points, 8)
    """


    A = np.empty((2*len(src_points), 8))
    row=0

    for (x_s,y_s),(x_t,y_t) in zip(src_points, target_points):
        
        A[row,0] = A[row+1,3] = x_s
        A[row,1] = A[row+1,4] = y_s
        A[row,2] = A[row+1,5] = 1
        A[row,3:6] = A[row+1,0:3] = 0
        A[row, 6] = -x_s*x_t
        A[row, 7] = -y_s*x_t
        A[row+1, 6] = -x_s*y_t
        A[row+1, 7] = -y_s*y_t
        row+=2
    
    return A 



def compute_homography_mat(src_points, target_points):
    """
        Uses the points in src_points and their corresponding points
        target_points to compute the homography matrix between the 2 images 
        from which the 2 lists where obtained.

        Args:
            src_points(list of tuples) : points from the first image.
            target_points(list of tuples) : points from the second image. 

        Returns:
            numpy.ndarray : matrix of shape (3,3) which is the homgraphy matrix.
    """

    A = compute_points_mat(src_points, target_points)
    b = np.ndarray.flatten(np.array(target_points))
    
    H = np.linalg.lstsq(A, b,None)[0]
    H = np.concatenate([H,[1]])
    H = H.reshape(3,3)
    return H


def transform_points(points, H):
    """
        Given a matrix of points with wach row (u, v) transforms 
        the points using H (the homography matrix).
        Args:
            points (numpy.ndarray) : matrix with each row being a point (u,v)
            H : (numpy.ndarray) :  matrix with shape (3,3) describing a projective transformation in 2D.

        Returns:
            numpy.ndarray : matrix of same shape as points, where mapped_points[i] 
                            is the mapping of points[i] using H.
    
    """


    ones = np.ones((points.shape[0], 1))
    points = np.concatenate([points, ones], 1)

    mapped_points = np.dot(points, H.T)
    mapped_points[:,:-1] /= np.expand_dims(mapped_points[:,-1],1)
    mapped_points = mapped_points[:,:-1]

    return mapped_points


def get_inliers(src_points, target_points, d=1, s=4, N=5000, T=None):
    """
        Uses RANSAC to find out which pairs of points from src_points 
        and target points is an inlier.

        Args:
            src_points(list of tuples) : points from the first image.
            target_points(list of tuples) : points from the second image. 
            d (float) : max distance an inlier can be at relative to the deduced transformation.
            s (int) : number of points to sample at random to solve for H at each iteration.
            N (int) : number of times to run the RANSAC loop.
            T (int) : min number of inliers that need to exist so that we re-solve for the 
                    transformation using the sampled points and the inliers.

        Returns:
            (tuple of numpy.ndarray) : (inliers in image one, their corresponding points in image two)

        Note:
            the number of returned inliers >= s. 

    """
    assert len(src_points) == len(target_points)
    src_points = np.concatenate([np.expand_dims(np.array(p),0) for p in src_points if type(p) != np.ndarray],0)
    target_points = np.concatenate([np.expand_dims(np.array(p),0) for p in target_points if type(p) != np.ndarray],0)
    

    T  = min(len(src_points), 10)

    samples_indices_history = []

    num_inliers_history = []

    def _get_inlier_indices(indices):
        
        H = compute_homography_mat(src_points[indices], target_points[indices])

        
        mapped_points = transform_points(src_points, H)

        dist = np.linalg.norm(mapped_points - target_points, axis=1)
        inlier_indices = np.where(dist <= d)[0]
        return inlier_indices
    

    for _ in range(N):
        indices = np.random.choice(len(src_points), s, replace=False)
        
        inlier_indices = _get_inlier_indices(indices)
        samples_indices_history.append(inlier_indices)
        num_inliers_history.append(len(inlier_indices))

        if num_inliers_history[-1] >= T:
            inlier_indices = _get_inlier_indices(inlier_indices)
            samples_indices_history.append(inlier_indices)
            num_inliers_history.append(len(inlier_indices))
            

    samples_indices_history = np.array(samples_indices_history)
    num_inliers_history = np.array(num_inliers_history)
    best_indices = samples_indices_history[np.argmax(num_inliers_history)]
    return src_points[best_indices], target_points[best_indices]



def transform_grid(u_range, v_range, H):
    """
        Generates a grid of with u values belonging to u_range and v values 
        belonging to v_range and transforms it using H.
        
        Args:
            u_range (numpy.ndarray): vector of length (N)
            v_range (numpy.ndarray): vector of length (M)
            H (numpy.ndarray) : matrix of shape (3,3) used to project the grid of points.
        
        Returns:
            (tuple of numpy.ndarray) : 
                first element: points of the grid layed out in a matrix with each row representing
                                a point (u, v)
                second element: a matrix of the same shape as the first element where each row 
                                represents the corresponding mapped point (u',v') using H.
    """



    grid_u, grid_v = np.meshgrid( u_range, v_range )

    u_flat = np.expand_dims(np.ndarray.flatten(grid_u), 1)
    v_flat = np.expand_dims(np.ndarray.flatten(grid_v), 1)
    points = np.concatenate([u_flat, v_flat],1)
    
    return points, transform_points(points, H)


def warp_image(image, H):

    """Warps an image using the homography matrix H.

    Args:
        image (numpy.ndarray): image to be warpped.
        H (numpy.ndarray) : Homography matrix used to warp the image.
    
    Returns:
        (tuple of numpy.ndarray, int, int):
            first element: the warpped images.
            second element: the minimum u corrdinate in corrdinate space not image space
                            this means this could be a negative number, in other words 
                            this is the amount of translation in the u diraction.

            third element: minimum v corrdcinate i.e. the translation in v direction.
    """

    H_inv = np.linalg.inv(H) 
    H_inv = H_inv / H_inv[2,2]
    # u == x
    # v == y
    

    orig_u_range = np.arange(image.shape[1])
    orig_v_range = np.arange(image.shape[0])

    _, transformed_image, = transform_grid(orig_u_range, orig_v_range, H)
    
    min_u=int(np.min(transformed_image[:,0]))
    max_u=int(np.max(transformed_image[:,0]))
    min_v=int(np.min(transformed_image[:,1]))
    max_v=int(np.max(transformed_image[:,1]))

    mapped_u_range = np.arange(min_u, max_u)
    mapped_v_range = np.arange(min_v, max_v)
    
    

    target_image = np.zeros((max_v-min_v, max_u-min_u,3))


    transformed_points, inv_transformed_image = transform_grid(mapped_u_range, mapped_v_range, H_inv)

    def fill_channel(target, channel, batch_size=64):
        I_cont = RectBivariateSpline(orig_v_range, orig_u_range, image[:,:,channel])

        n_iters =int( len(inv_transformed_image) / batch_size )
        
        for i in range(n_iters + 1):
            start = i * batch_size
            end = (i+1) * batch_size
            
            mapped_u_batch = inv_transformed_image[start:end, 0].ravel()
            mapped_v_batch = inv_transformed_image[start:end, 1].ravel()
            
            u_batch = transformed_points[start:end, 0].ravel()
            v_batch = transformed_points[start:end, 1].ravel()

            target[v_batch-min_v, u_batch-min_u, channel] = I_cont(mapped_v_batch, mapped_u_batch, grid=False)

    fill_channel(target_image, 0)
    fill_channel(target_image, 1)
    fill_channel(target_image, 2)

    return target_image, min_u, min_v
                
def read_image(path):
    img = cv.imread(path,1)
    return img


def automatic_intrest_points_detector(image1, image2, N=75):
    """
    This function get key point from two images instead of doing it manullay


    Args:
        image1: first image to get key points from
        image2: second images to get key points from
        N: Number of points to be detected 

    Returns:
        Two lists of detected interest points from the two images
    """
    # ORB: An efficient alternative to SIFT or SURF
    orb = cv.ORB_create() 
    
    # get key points and descriptors from the 2 images
    kps1, descs1 = orb.detectAndCompute(image1, None)
    kps2, descs2 = orb.detectAndCompute(image2, None)
    
    # brute force matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    matches = sorted(matches, key=lambda x:x.distance)
    list_kps1 = [kps1[mat.queryIdx].pt for mat in matches[:N]] 
    list_kps2 = [kps2[mat.trainIdx].pt for mat in matches[:N]]

    return list_kps1, list_kps2


def stitch_2_images(path_1, path_2, correspondance_points=75, save=False, load=False, SIFT=True, ransac=True):
    """
    stitches 2 images.

    path_1(str) : path to first image
    path_2(str) : path to second image
    correspondance points(int) : number of points to take from each image for pixel matching.
    save (bool) : weather to save the sampled points in a pickel file.
    load (boo) : weather to load the points from a pickel file with the same name as the images.
    SIFT (bool) : weather to use SIFT features to match interest points or input the points manually.
    ransac(bool) : weather to use ransac or least square
    
    Returns:
        (numpy.ndarray) : matrix of shape ((warpped_image_1.shape[0] + image_2.shape[0],
                                            warpped_image_1.shape[1] + image_2.shape[1], 3))
                         which is the stitched images.
    """


    image_1 = read_image(path_1)
    image_2 = read_image(path_2)

    image_1_name = path_1.split('/')[-1].split('.')[0]
    image_2_name = path_2.split('/')[-1].split('.')[0]
    
    if load:
        with open(f'{image_1_name}.pkl', 'rb') as f:
            image_1_points = pickle.load(f)

        with open(f'{image_2_name}.pkl', 'rb') as f:
            image_2_points = pickle.load(f)

    else:
        if SIFT:
            image_1_points, image_2_points = automatic_intrest_points_detector(image_1, image_2, correspondance_points)
        else:
            image_1_points = get_points(image_1, correspondance_points)
            image_2_points = get_points(image_2, correspondance_points)
    
    if save:
        with open(f'{image_1_name}.pkl', 'wb+') as f:
            pickle.dump(image_1_points,f)

        with open(f'{image_2_name}.pkl', 'wb+') as f:
            pickle.dump(image_2_points, f)

    # convert bgr to rgb then scale image to range [0,1)
    image_1 = image_1[...,::-1]/255
    image_2 = image_2[...,::-1]/255


    if ransac:
        inlier_src, inlier_target = get_inliers(image_1_points, image_2_points)
    else:
        inlier_src, inlier_target = image_1_points, image_2_points
    
    H = compute_homography_mat(inlier_src, inlier_target)



    warpped_image_1, min_u, min_v = warp_image(image_1, H)

    res = np.zeros((warpped_image_1.shape[0] + image_2.shape[0],
                    warpped_image_1.shape[1] + image_2.shape[1], 3))

    shift_u_1 = min_u if min_u>0 else 0
    shift_v_1 = min_v if min_v>0 else 0

    res[shift_v_1:warpped_image_1.shape[0]+shift_v_1, shift_u_1:warpped_image_1.shape[1]+shift_u_1, :] = warpped_image_1

    shift_u_2 = -min_u if min_u<0 else 0
    shift_v_2 = -min_v if min_v<0 else 0
    res[shift_v_2:image_2.shape[0] + shift_v_2, shift_u_2:image_2.shape[1] + shift_u_2, :] = image_2

    res = res[0:np.maximum(image_1.shape[0], image_2.shape[0])+200,
              0:np.maximum(image_1.shape[1], image_2.shape[1])+700]
    return res

def stitch_N_images(paths, load=False, save=False, SIFT=True, ransac=True, correspondance_points=75):
    """
    stitches N images.

    paths : paths to N images with know order
    
    Returns:
        (numpy.ndarray) : matrix of shape ((warpped_image_1.shape[0] + image_2.shape[0],
                                            warpped_image_1.shape[1] + image_2.shape[1], 3))
                         which is the stitched images.
    """
    N = len(paths)
    swap = paths[0]
    for i in range(1,N):
        res = stitch_2_images(paths[i], swap, correspondance_points=correspondance_points, load=load, save=save, SIFT=SIFT, ransac=True)
        swap = 'swap.png'
        plt.imsave(swap, res)
    return res


parser = argparse.ArgumentParser()
parser.add_argument( "images", nargs=2, type=str,
                        help="list of images to stitch",
                        default=None)
parser.add_argument('--no_sift', action='store_false', help='weather to use sift or select points manualy.', default=True)

parser.add_argument('--no_ransac', action='store_false', help='weather to use ransac to eleminate outliers.', default=True)

parser.add_argument('--num_points', type=int, help='number of points to take from each image.', default=200)

args = parser.parse_args()


res = stitch_2_images(args.images[0], args.images[1], correspondance_points=args.num_points, ransac=args.no_ransac, SIFT=args.no_sift)

plt.imshow(res)
plt.show()




