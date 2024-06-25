# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def visualize_rotation_distribution(pre_r, kappa, w, num_points=100):
#     pre_r = torch.tensor(pre_r, dtype=torch.float32)
#     kappa = torch.tensor(kappa, dtype=torch.float32)
#     w = torch.tensor(w, dtype=torch.float32)

#     # Generate a grid of rotation matrices
#     theta = torch.linspace(0, np.pi, num_points)
#     phi = torch.linspace(0, 2 * np.pi, num_points)
#     theta, phi = torch.meshgrid(theta, phi)
#     x = torch.sin(theta) * torch.cos(phi)
#     y = torch.sin(theta) * torch.sin(phi)
#     z = torch.cos(theta)
#     grid_rotations = torch.stack([x, y, z], dim=-1)

#     # Compute probability density on the grid
#     grid_rotations_flat = grid_rotations.view((-1, 3))
#     dot_product = torch.cos(torch.acos(grid_rotations_flat[:, :2].matmul(pre_r[:, :2].t())))
#     loss_pixelwise = -torch.log(torch.square(kappa) + 1) + kappa * torch.acos(dot_product) + torch.log(1 + torch.exp(-kappa * np.pi))
#     loss_pixelwise = torch.where(torch.isinf(loss_pixelwise) | torch.isnan(loss_pixelwise), torch.tensor(0.0), loss_pixelwise)
#     probability_density = (w.t() @ loss_pixelwise[:, 0].view(32, 1) + w.t() @ loss_pixelwise[:, 1].view(32, 1)).view(num_points, num_points)

#     # Plot the distribution
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), facecolors=plt.cm.viridis(probability_density / probability_density.max()), rstride=5, cstride=5, alpha=0.7, shade=False)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Rotation Distribution')

#     plt.show()

# # Example usage
# # Assuming pre_r, kappa, and w are available
# # visualize_rotation_distribution(pre_r, kappa, w)


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def sample_unit_sphere(num_points=10000):
#     """Sample points on the unit sphere."""
#     phi = np.random.uniform(0, 2 * np.pi, num_points)
#     costheta = np.random.uniform(-1, 1, num_points)
#     theta = np.arccos(costheta)
    
#     x = np.cos(theta) * np.cos(phi)
#     y = np.cos(theta) * np.sin(phi)
#     z = np.sin(theta)

#     return np.stack([x, y, z], axis=-1),np.stack([phi,theta])

# def visualize_so3_distribution(pre_r, kappa, w, num_points=10000):
#     pre_r = torch.tensor(pre_r, dtype=torch.float32)
#     kappa = torch.tensor(kappa, dtype=torch.float32)
#     w = torch.tensor(w, dtype=torch.float32)

#     # Sample points on the unit sphere
#     sphere_points = sample_unit_sphere(num_points)
#     sphere_points = torch.tensor(sphere_points, dtype=torch.float32)

#     # Compute probability density at sampled points
#     dot_product = torch.cos(torch.acos(sphere_points[:, :2].matmul(pre_r[:, :2].t())))
#     loss_pixelwise = -torch.log(torch.square(kappa) + 1) + kappa * torch.acos(dot_product) + torch.log(1 + torch.exp(-kappa * np.pi))
#     loss_pixelwise = torch.where(torch.isinf(loss_pixelwise) | torch.isnan(loss_pixelwise), torch.tensor(0.0), loss_pixelwise)
#     probability_density = (w.t() @ loss_pixelwise[:, 0].view(32, 1) + w.t() @ loss_pixelwise[:, 1].view(32, 1)).view(num_points)

#     # Map points to 3D space using spherical coordinates
#     x = sphere_points[:, 0]
#     y = sphere_points[:, 1]
#     z = sphere_points[:, 2]

#     # Plot the distribution
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x.numpy(), y.numpy(), z.numpy(), c=probability_density.numpy(), cmap='viridis', s=5)

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('SO(3) Distribution')

#     plt.show()

# Example usage
# Assuming pre_r, kappa, and w are available
# visualize_so3_distribution(pre_r, kappa, w)

#======使用numpy的形式=========
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

#采样
def sample_unit_sphere(num_points=10000):
    """Sample points on the unit sphere."""
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack([x, y, z], axis=-1),phi,theta
#可视化分布
def visualize_so3_distribution(pre_R, Kappa, W, num_points=10000):
    # Sample points on the unit sphere
    sphere_points,phi,theta = sample_unit_sphere(num_points)
    #torch转为numpy
    pre_r=pre_R.cpu().numpy()
    kappa=Kappa.cpu().numpy()
    w=W.cpu().numpy()
    #转换为旋转矩阵
    # yaw=np.random.uniform(-np.pi,np.pi,num_points)#np.zeros_like(phi)
    yaw=np.zeros_like(phi)
    rotation_matrix = Rotation.from_euler('xyz', np.column_stack((phi, theta,yaw)), degrees=False).as_dcm()
    # Compute probability density at sampled points
    cosine_simi1=np.arccos(np.matmul(rotation_matrix[:,:, 0], pre_r[:,:, 0].T))
    cosine_simi2=np.arccos(np.matmul(rotation_matrix[:,:, 1], pre_r[:,:, 1].T))
    dot_product=np.stack((cosine_simi1,cosine_simi2),axis=2)
    loss_pixelwise = np.log(np.square(kappa) + 1) - kappa * dot_product - np.log(1 + np.exp(-kappa * np.pi))
    loss_pixelwise = np.where(np.isinf(loss_pixelwise) | np.isnan(loss_pixelwise), 0.0, loss_pixelwise)
    # probability_density = np.dot(w.T, loss_pixelwise[:,:, 0].reshape(-1,32, 1) + w.T @ loss_pixelwise[:,:, 1].reshape(-1,32, 1)).reshape(num_points)
    probability_density = np.sum(loss_pixelwise[:,:,:1]*w,axis=1)+np.sum(loss_pixelwise[:,:,1:2]*w,axis=1)

    # Map points to 3D space using spherical coordinates
    x, y, z = sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2]
    # x, y, z = phi,theta,yaw

    # Plot the distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=probability_density.reshape(num_points), cmap='viridis', s=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('SO(3) Distribution')
    plt.savefig('my_plot.png')
    plt.show()
#已弃用
def visualize_so3_distribution_discard(pre_R, Kappa, W, num_points=10000):
    # Sample points on the unit sphere
    sphere_points, phi, theta = sample_unit_sphere(num_points)
    # torch转为numpy
    pre_r = pre_R.cpu().numpy()
    kappa = Kappa.cpu().numpy()
    w = W.cpu().numpy()
    # 转换为旋转矩阵
    yaw = np.zeros_like(phi)
    rotation_matrix = Rotation.from_euler('xyz', np.column_stack((phi, theta, yaw)), degrees=False).as_dcm()
    # Compute probability density at sampled points
    cosine_simi1 = np.arccos(np.matmul(rotation_matrix[:, :, 0], pre_r[:, :, 0].T))
    cosine_simi2 = np.arccos(np.matmul(rotation_matrix[:, :, 1], pre_r[:, :, 1].T))
    dot_product = np.stack((cosine_simi1, cosine_simi2), axis=2)
    loss_pixelwise = np.log(np.square(kappa) + 1) - kappa * dot_product - np.log(1 + np.exp(-kappa * np.pi))
    loss_pixelwise = np.where(np.isinf(loss_pixelwise) | np.isnan(loss_pixelwise), 0.0, loss_pixelwise)
    probability_density = np.sum(loss_pixelwise[:, :, :1] * w, axis=1) + np.sum(loss_pixelwise[:, :, 1:2] * w, axis=1)

    # Map points to 3D space using spherical coordinates
    x, y, z = sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2]

    # Plot the distribution
    fig = plt.figure(figsize=(12, 5))

    # Plot the 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(x, y, z, c=probability_density.reshape(num_points), cmap='viridis', s=5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('SO(3) Distribution')

    # Plot the histogram of rotation axis distribution
    ax2 = fig.add_subplot(122, projection='3d')
    hist, xedges, yedges = np.histogram2d(phi, theta, bins=50, range=[[0, 2 * np.pi], [0, np.pi]])

    x_midpoints = (xedges[1:] + xedges[:-1]) / 2
    y_midpoints = (yedges[1:] + yedges[:-1]) / 2

    x, y = np.meshgrid(x_midpoints, y_midpoints)
    ax2.bar3d(x.ravel(), y.ravel(), 0, 0.02, 0.02, hist.ravel(), shade=True)

    ax2.set_xlabel('Phi')
    ax2.set_ylabel('Theta')
    ax2.set_zlabel('Frequency')
    ax2.set_title('Rotation Axis Distribution Histogram')

    # Display the colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Probability Density')

    plt.savefig('my_plot.png')
    plt.show()

# Example usage
# Assuming pre_r, kappa, and w are available
# visualize_so3_distribution(pre_r, kappa, w)

import numpy as np
#从采样点数据恢复旋转矩阵
def spherical_to_rotation_matrix(azimuth, elevation):
    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    cos_elevation = np.cos(elevation)
    sin_elevation = np.sin(elevation)

    # Rotation matrix
    R_x = np.array([[1, 0, 0],
                    [0, cos_elevation, -sin_elevation],
                    [0, sin_elevation, cos_elevation]])

    R_z = np.array([[cos_azimuth, -sin_azimuth, 0],
                    [sin_azimuth, cos_azimuth, 0],
                    [0, 0, 1]])

    rotation_matrix = np.dot(R_z, R_x)

    return rotation_matrix



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#可视化混合成分的旋转矩阵
def vis_rotation_matrix(rotation_matrices, ratation, W=0):
    if not hasattr(vis_rotation_matrix,'static_variable'):#使用闭包的方式定义静态变量
        vis_rotation_matrix.static_variable=0

    rotation_matrices=rotation_matrices.cpu().numpy()
    ratation=ratation.cpu().numpy()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(121, projection='3d')
    num_matrices,_,_=rotation_matrices.shape
    # 遍历每个旋转矩阵并绘制坐标轴
    for i in range(num_matrices):
        # 提取旋转矩阵的坐标轴方向
        x_axis = rotation_matrices[i, :, 0]
        y_axis = rotation_matrices[i, :, 1]
        z_axis = rotation_matrices[i, :, 2]
        
        # 绘制坐标轴
        ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', alpha=0.5)
        ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', alpha=0.5)
        ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', alpha=0.5)

    # 设置坐标轴范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # 设置坐标轴标签
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    ax= fig.add_subplot(122, projection='3d')
    x_axis = ratation[:, 0]
    y_axis = ratation[:, 1]
    z_axis = ratation[:, 2]
    
    # 绘制坐标轴
    ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', alpha=0.5)
    ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', alpha=0.5)
    ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', alpha=0.5)

    # 设置坐标轴范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # 设置坐标轴标签
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    # 显示图像
    plt.title('Visualization of 32 Rotation Matrices in a Unified Coordinate System')
    plt.show()
    plt.savefig(f'./vis/my_rotation{vis_rotation_matrix.static_variable}.png')
    vis_rotation_matrix.static_variable+=1
#可视化混合成分的旋转矩阵--仅仅可视化预测矩阵
def vis_one_rotation_matrix(rotation_matrices, ratation, W=0):
    if not hasattr(vis_rotation_matrix,'static_variable'):#使用闭包的方式定义静态变量
        vis_rotation_matrix.static_variable=0

    rotation_matrices=rotation_matrices.cpu().numpy()
    # ratation=ratation.cpu().numpy()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    num_matrices,_,_=rotation_matrices.shape
    # 遍历每个旋转矩阵并绘制坐标轴
    for i in range(num_matrices):
        # 提取旋转矩阵的坐标轴方向
        x_axis = rotation_matrices[i, :, 0]
        y_axis = rotation_matrices[i, :, 1]
        z_axis = rotation_matrices[i, :, 2]
        
        # 绘制坐标轴
        ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', alpha=0.5)
        ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', alpha=0.5)
        ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', alpha=0.5)

    # 设置坐标轴范围
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # 设置坐标轴标签
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # ax= fig.add_subplot(122, projection='3d')
    # x_axis = ratation[:, 0]
    # y_axis = ratation[:, 1]
    # z_axis = ratation[:, 2]
    
    # # 绘制坐标轴
    # ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', alpha=0.5)
    # ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', alpha=0.5)
    # ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', alpha=0.5)

    # # 设置坐标轴范围
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])

    # # 设置坐标轴标签
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # # 显示图像
    # plt.title('Visualization of 32 Rotation Matrices in a Unified Coordinate System')
    plt.show()
    # plt.savefig(f'./vis/my_rotation{vis_rotation_matrix.static_variable}.png')
    plt.savefig('vis.png')

    vis_rotation_matrix.static_variable+=1
if __name__=='__main__':
    num_matrices = 32
    rotation_matrices = np.random.rand(num_matrices, 3, 3)
    vis_rotation_matrix(rotation_matrices)





