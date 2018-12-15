import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import pickle


def get_points(image, n=4):
    plt.imshow(image)
    points = plt.ginput(n, timeout=-100)
    plt.show()
    plt.close()
    return points

def compute_points_mat(src_points, target_points):
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
    A = compute_points_mat(src_points, target_points)
    b = np.ndarray.flatten(np.array(target_points))
    
    H = np.linalg.lstsq(A, b,None)[0]
    H = np.concatenate([H,[1]])
    H = H.reshape(3,3)
    return H


def transform_points(points, H):
    ones = np.ones((points.shape[0], 1))
    points = np.concatenate([points, ones], 1)

    mapped_points = np.dot(points, H.T)
    mapped_points[:,:-1] /= np.expand_dims(mapped_points[:,-1],1)
    mapped_points = mapped_points[:,:-1]

    return mapped_points


def get_inliers(src_points, target_points):
    """
    returns inliers    
    """
    assert len(src_points) == len(target_points)
    src_points = np.concatenate([np.expand_dims(np.array(p),0) for p in src_points if type(p) != np.ndarray],0)
    target_points = np.concatenate([np.expand_dims(np.array(p),0) for p in target_points if type(p) != np.ndarray],0)
    
    s = 4
    #N = int(np.log10(1-0.99)/np.log10((1-(1-0.4)**s)))
    N = 2000
    T  = min(len(src_points), 10)
    d = 5

    samples_indices_history = []
    num_inliers_history = []

    def _get_inlier_indices(indices):
        
        H = compute_homography_mat(src_points[indices], target_points[indices])

        # outside_indices = np.array([i for i in range(len(src_points)) if i not in set(indices)])
        
        mapped_points = transform_points(src_points, H)

        dist = np.linalg.norm(mapped_points - target_points, axis=1)
        print(dist)
        inlier_indices = np.where(dist <= d)[0]
        print(inlier_indices)
        return inlier_indices
    
    print('N:', N)

    for _ in range(N):
        indices = np.random.choice(len(src_points), s, replace=False)
        #samples_indices_history.append(indices)
        
        inlier_indices = _get_inlier_indices(indices)
        samples_indices_history.append(inlier_indices)
        num_inliers_history.append(len(inlier_indices))

        if num_inliers_history[-1] >= T:
            inlier_indices = _get_inlier_indices(inlier_indices)
            samples_indices_history.append(inlier_indices)
            num_inliers_history.append(len(inlier_indices))
            

    samples_indices_history = np.array(samples_indices_history)
    print(num_inliers_history)
    num_inliers_history = np.array(num_inliers_history)
    best_indices = samples_indices_history[np.argmax(num_inliers_history)]
    print(num_inliers_history)
    return src_points[best_indices], target_points[best_indices]



def transform_point(point, H):
    if type(point) == list or type(point) == tuple:
        point = np.array(point)

    point_hom = np.concatenate([point, [1]])
    scaler = 1/(H[2,0]*point[0] + H[2,1]*point[1] + 1)
    mapped_point = scaler * np.dot(H[0:2,:], point_hom)

    return mapped_point


def transform_image(u_range, v_range, H):
    grid_u, grid_v = np.meshgrid( u_range, v_range )

    u_flat = np.expand_dims(np.ndarray.flatten(grid_u), 1)
    v_flat = np.expand_dims(np.ndarray.flatten(grid_v), 1)
    points = np.concatenate([u_flat, v_flat],1)
    
    return points, transform_points(points, H)


def warp_image(image, H):
    H_inv = np.linalg.inv(H) 
    H_inv = H_inv / H_inv[2,2]
    # u == x
    # v == y
    

    orig_u_range = np.arange(image.shape[1])
    orig_v_range = np.arange(image.shape[0])

    orig_points, transformed_image, = transform_image(orig_u_range, orig_v_range, H)
    
    min_u=int(np.min(transformed_image[:,0]))
    max_u=int(np.max(transformed_image[:,0]))
    min_v=int(np.min(transformed_image[:,1]))
    max_v=int(np.max(transformed_image[:,1]))

    mapped_u_range = np.arange(min_u, max_u)
    mapped_v_range = np.arange(min_v, max_v)
    print(max_v-min_v, max_u-min_u)
    

    target_image = np.zeros((max_v-min_v, max_u-min_u,3))


    transformed_points, inv_transformed_image = transform_image(mapped_u_range, mapped_v_range, H_inv)

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
    img = plt.imread(path)
    print(img.shape)
    return img[:,:,:-1]

def automatic_intrest_points_detector(image1, image2, N):
    ###
    # image1 and image2 are the 2 images that have commen points
    # N is number of points reqired to be detected
    ###
    # ORB: An efficient alternative to SIFT or SURF
    orb = cv.ORB_create()

    # get key points and descriptors from the 2 images
    kps1, descs1 = orb.detectAndCompute(image1, None)
    kps2, descs2 = orb.detectAndCompute(image2, None)
    print(kps1)
    # brute force matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    matches = sorted(matches, key=lambda x:x.distance)
    list_kps1 = [kps1[mat.queryIdx].pt for mat in matches[:N]]
    list_kps2 = [kps2[mat.trainIdx].pt for mat in matches[:N]]

    return list_kps1, list_kps2

def stitch_images(path_1, path_2, correspondance_points=5, auto=True ,save=True, load=True):
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
        if auto:
            image_1_points, image_2_points = automatic_intrest_points_detector(image_1, image_2, correspondance_points)
        else:
             image_1_points = get_points(image_1, correspondance_points)
             image_2_points = get_points(image_2, correspondance_points)
    
    if save:
        with open(f'{image_1_name}.pkl', 'wb+') as f:
            pickle.dump(image_1_points,f)

        with open(f'{image_2_name}.pkl', 'wb+') as f:
            pickle.dump(image_2_points, f)

    inlier_src, inlier_target = get_inliers(image_1_points, image_2_points)

    print(inlier_src[:,0])
    print(inlier_target[:,0])

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(image_1)
    axs[0].scatter(inlier_src[:,0], inlier_src[:,1])
    axs[1].imshow(image_2)
    axs[1].scatter(inlier_target[:,0], inlier_target[:,1])
    plt.show()



    H = compute_homography_mat(inlier_src, inlier_target)



    warpped_image_1, min_u, min_v = warp_image(image_1, H)

    res = np.zeros((warpped_image_1.shape[0] + image_2.shape[0],
                    warpped_image_1.shape[1] + image_2.shape[1], 3))



    res[:warpped_image_1.shape[0], :warpped_image_1.shape[1], :] = warpped_image_1
    print(min_u, min_v)

    shift_u = -min_u if min_u<0 else 0
    shift_v = -min_v if min_v<0 else 0
    res[shift_v:image_2.shape[0] + shift_v, shift_u:image_2.shape[1] + shift_u, :] = image_2


    return res


res = stitch_images('b1_copy.png','b2_copy.png' , 30, load=False)

plt.imshow(res)
plt.show()





# def main():
#     building_1 = read_image('stitching_1.png')
#     building_2 = read_image('stitching_2.png')


#     try:
#         with open('b1.pkl', 'rb') as f:
#             building_1_points = pickle.load(f)
        
#         with open('b2.pkl', 'rb') as f:
#             building_2_points = pickle.load(f)
        
#     except:
#         building_1_points = get_points(building_1,5)
#         building_2_points = get_points(building_2,5)


#         with open('b1.pkl','wb+') as f:
#             pickle.dump(building_1_points,f)

#         with open('b2.pkl','wb+') as f:
#             pickle.dump(building_2_points,f)
            

   
#     H = compute_homography_mat(building_1_points, building_2_points)
#     # H = np.eye(3) * 1.5
#     # H[2,2]=1


#     plt.imshow(building_2)
#     mapped_points = []
#     for point in building_1_points:
#         mapped_point1 = transform_point(point, H)
#         mapped_point2 = transform_point([point[0] + 500, point[1] + 500], H)
#         mapped_point3 = transform_point([point[0] + 100, point[1] + 100], H)

#         mapped_points.append(mapped_points)
        
#         plt.scatter(mapped_point1[0], mapped_point1[1], c='red')
#         plt.scatter(mapped_point2[0], mapped_point2[1], c='blue')
#         plt.scatter(mapped_point3[0], mapped_point3[1], c='yellow')

    
#     plt.show()


#     # plt.imshow(building_1)
#     # H_inv = np.linalg.inv(H)

#     # b1_points = []

#     # for point in mapped_points + [[0,0]]:
#     #     b1_point = transform_point(point, H_inv/H_inv[2,2])
#     #     b1_points.append(b1_point)
#     #     plt.scatter(b1_point[0], b1_point[1], c='yellow')

#     # plt.show()

#     try:
#         with open('warpped.pkl', 'rb') as f:
#             warpped_building_1, min_u, min_v = pickle.load(f)
    
#     except:
#         warpped_building_1, min_u, min_v = warp_image(building_1, H)
#         with open('warpped.pkl', 'wb+') as f:
#             pickle.dump((warpped_building_1, min_u, min_v), f)
        

#     res = np.zeros((warpped_building_1.shape[0] + building_2.shape[0],
#                     warpped_building_1.shape[1] + building_2.shape[1], 3))



#     res[:warpped_building_1.shape[0], :warpped_building_1.shape[1], :] = warpped_building_1

#     res[-min_v:-min_v + building_2.shape[0], -min_u:-min_u + building_2.shape[1], :] = building_2
    
#     # res[:, int(pivot[0]):, :]  = building_2[:, mapped_points[0][0]:, :]



#     plt.imshow( res )

#     plt.show()


    
    





# main()

# H = np.array([[ 1.96266782e-01 , -1.98196535e+00 , 4.81034471e+02],
#  [ 1.07510522e-01 , -1.23659214e+00 , 3.14197533e+02],
#  [ 3.24089376e-04 , -3.88242171e-03 , 1.00000000e+00]])
    
# # x = np.array((814.9165742210662, 317.62283151311124))
# # x_hom = np.array((814.9165742210662, 317.62283151311124,1))
# x_hom = np.array([0,0,1])
# scaler = 1/(H[2,0]*x_hom[0] + H[2,1]*x_hom[1] + 1)
# x_t_hom = scaler * H.dot(x_hom)
# print(x_t_hom)
# H_inv = np.linalg.inv(H)
# H_inv = H_inv / H_inv[2,2]
# scaler = 1/(H_inv[2,0]*x_t_hom[0] + H_inv[2,1]*x_t_hom[1] + 1)
# print(H_inv)
# print( scaler * H_inv.dot(x_t_hom))










