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


def get_inliers(src_points, target_points):
    """
    Returns a list of inliers and a list of outliers using RANSAC.
    """
    assert len(src_points) == len(target_points)

    src_points = np.concatenate([np.array(p) for p in src_points if type(p) != np.ndarray])
    target_points = np.concatenate([np.array(p) for p in target_points if type(p) != np.ndarray])
    
    s = 4
    N = int(np.log10(1-0.99)/np.log10((1-(1-0.05)**s)))
    T  = min(len(src_points), 6)
    d = 10

    samples_indices_history = []
    num_inliers_history = []

    for _ in range(N):
        indices = np.random.choice(len(src_points), s, replace=False)
        samples_indices_history.append(indices)
        src_samples = src_points[indices]
        target_samples = target_points[indices]
        H = compute_homography_mat(src_samples, target_samples)

        outside_indices = np.array(list(set(range(len(src_points))) - set(src_samples)) )

        mapped_points = src_points[outside_indices].dot(H.T)
        mapped_points[:,:-1] /= mapped_points[:,-1]
        mapped_points = mapped_points[:,:-1]

        dist = np.linalg.norm(mapped_points - target_points[outside_indices], axis=1)
        inlier_indices = np.where(dist <= d)[0]
        num_inliers_history.append(len(inlier_indices))

        if num_inliers_history[-1] >= T:
            


        













def transform_point(point, H):
    if type(point) == list or type(point) == tuple:
        point = np.array(point)

    point_hom = np.concatenate([point, [1]])
    scaler = 1/(H[2,0]*point[0] + H[2,1]*point[1] + 1)
    mapped_point = scaler * np.dot(H[0:2,:], point_hom)

    return mapped_point






def warp_image(image, H):
    H_inv = np.linalg.inv(H) 
    H_inv = H_inv / H_inv[2,2]
    # u == x
    # v == y
    
    # top_left = transform_point([0,0],H)
    # bottom_right = transform_point([image.shape[0],image.shape[1]], H)
    min_u = np.inf
    min_v = np.inf
    max_u = 0
    max_v = 0
    for v in range(image.shape[0]):
        for u in range(image.shape[1]):
            mapped_point = transform_point([u,v], H)
            min_u = min(mapped_point[0],min_u)
            min_v = min(mapped_point[1],min_v)
            max_u = max(mapped_point[0],max_u)
            max_v = max(mapped_point[1],max_v)


            
    min_u = int(min_u)
    min_v = int(min_v)
    max_u = int(max_u)
    max_v = int(max_v)
   
    u_range = np.arange(image.shape[1])
    v_range = np.arange(image.shape[0])


    # the spline describing a continuous interpolation of the image.
    target_image = np.zeros((max_v-min_v, max_u-min_u,3))

    for channel in range(3):
        I_cont = RectBivariateSpline(v_range, u_range, image[:,:,channel])

        for v in np.arange(min_v,max_v):
            for u in np.arange(min_u,max_u):
                point = np.array([u,v])
                mapped_u, mapped_v = transform_point(point, H_inv)
                target_image[v-min_v,u-min_u,channel] = I_cont(mapped_v, mapped_u)

                # print(f'mapped {u},{v} to {mapped_u}, {mapped_v}')

    return target_image, min_u, min_v
                
def read_image(path):
    img = plt.imread(path)
    print(img.shape)
    return img[:,:,:-1]

def stitch_images(path_1, path_2, correspondance_points=5, save=True, load=True):
    image_1 = read_image(path_1)
    image_2 = read_image(path_2)
    image_1_name = path_1.split('/')[-1].split('.')[0]
    image_2_name = path_2.split('/')[-1].split('.')[0]
    try:
        with open(f'{image_1_name}.pkl', 'rb') as f:
            image_1_points = pickle.load(f)
        
        with open(f'{image_2_name}.pkl', 'rb') as f:
            image_2_points = pickle.load(f)
        
    except:
        image_1_points = get_points(image_1, correspondance_points)
        image_2_points = get_points(image_2, correspondance_points)

 

    H = compute_homography_mat(image_1_points, image_2_points)

    warpped_image_1, min_u, min_v = warp_image(image_1, H)

    res = np.zeros((warpped_image_1.shape[0] + image_2.shape[0],
                    warpped_image_1.shape[1] + image_2.shape[1], 3))



    res[:warpped_image_1.shape[0], :warpped_image_1.shape[1], :] = warpped_image_1

    res[-min_v:-min_v + image_2.shape[0], -min_u:-min_u + image_2.shape[1], :] = image_2


    return res


res = stitch_images('b1.png','b2.png' )

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










