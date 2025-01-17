import numpy as np
from numpy import pi as PI
from numpy import linalg as LA
import scipy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib import colors

def mat_2_quat(rot_mat):
    # [x, y, z, w] convention
    R = Rotation.from_matrix(rot_mat)
    return R.as_quat()

def quat_2_mat(quaternion):
    # [x, y, z, w] convention
    R = Rotation.from_quat(quaternion)
    return np.asarray(R.as_matrix())

def gram(data):
    # shape of data is (n,d)
    d = data.shape[1]
    G = np.zeros((d, d))
    for x in data:
        v = x.reshape(d,1)
        G = G + np.matmul(v, v.T)
    return G

def det_2_by_2(M):
    return M[0,0]*M[1,1] - M[1,0]*M[0,1]

def det_3_by_3(M):
    return M[0,0]*(M[1,1]*M[2,2]-M[1,2]*M[2,1]) - M[0,1]*(M[1,0]*M[2,2]-M[2,0]*M[1,2]) + M[0,2]*(M[1,0]*M[2,1]-M[2,0]*M[1,1])

def det_4_by_4(M):
    return M[0,0]*M[1,1]*M[2,2]*M[3,3] - M[0,0]*M[1,1]*M[2,3]*M[3,2] - M[0,0]*M[1,2]*M[2,1]*M[3,3] + M[0,0]*M[1,2]*M[2,3]*M[3,1] + \
           M[0,0]*M[1,3]*M[2,1]*M[3,2] - M[0,0]*M[1,3]*M[2,2]*M[3,1] - M[0,1]*M[1,0]*M[2,2]*M[3,3] + M[0,1]*M[1,0]*M[2,3]*M[3,2] + \
           M[0,1]*M[1,2]*M[2,0]*M[3,3] - M[0,1]*M[1,2]*M[2,3]*M[3,0] - M[0,1]*M[1,3]*M[2,0]*M[3,2] + M[0,1]*M[1,3]*M[2,2]*M[3,0] + \
           M[0,2]*M[1,0]*M[2,1]*M[3,3] - M[0,2]*M[1,0]*M[2,3]*M[3,1] - M[0,2]*M[1,1]*M[2,0]*M[3,3] + M[0,2]*M[1,1]*M[2,3]*M[3,0] + \
           M[0,2]*M[1,3]*M[2,0]*M[3,1] - M[0,2]*M[1,3]*M[2,1]*M[3,0] - M[0,3]*M[1,0]*M[2,1]*M[3,2] + M[0,3]*M[1,0]*M[2,2]*M[3,1] + \
           M[0,3]*M[1,1]*M[2,0]*M[3,2] - M[0,3]*M[1,1]*M[2,2]*M[3,0] - M[0,3]*M[1,2]*M[2,0]*M[3,1] + M[0,3]*M[1,2]*M[2,1]*M[3,0]

def quat_from_axis_angle(s, theta):
    # [x, y, z, w] convention
    s = s/LA.norm(s)
    st = np.sin(theta/2)
    return [st*s[0], st*s[1], st*s[2], np.cos(theta/2)]

def quat_mul(p,q):
    # [x, y, z, w] convention
    return np.array([p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0],
                     p[2] * q[0] - p[0] * q[2] + p[1] * q[3] + p[3] * q[1],
                     p[0] * q[1] - p[1] * q[0] + p[2] * q[3] + p[3] * q[2],
                     p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2]])

def plot_frames(data, hold_show=False, elev=30, azim=45, title="frames"):
    if np.shape(data[0]) in [(4,), (4,1), (1,4)]:
        Rs = []
        for q in data:
            Rs.append(quat_2_mat(q))
    elif np.shape(data[0]) == (3,3):
        Rs = data
    else:
        raise Exception("data must be rotation matrices or quaternions")
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d')
    for R in Rs:
        ax.plot3D([0, R[0,0]], [0, R[1,0]], [0, R[2,0]], 'r')
        ax.plot3D([0, R[0,1]], [0, R[1,1]], [0, R[2,1]], 'g')
        ax.plot3D([0, R[0,2]], [0, R[1,2]], [0, R[2,2]], 'b')
    arrx, arry, arrz = arrow('x', offset=1.5)
    ax.plot_surface(arrx, arry, arrz, color='k', zorder=10)
    ax.plot3D([0, 1.5], [0, 0], [0, 0], 'k', zorder=5, linewidth=3)
    arrx, arry, arrz = arrow('y', offset=1.5)
    ax.plot_surface(arrx, arry, arrz, color='k', zorder=10)
    ax.plot3D([0, 0], [0, 1.5], [0, 0], 'k', zorder=5, linewidth=3)
    arrx, arry, arrz = arrow('z', offset=1.5)
    ax.plot_surface(arrx, arry, arrz, color='k', zorder=10)
    ax.plot3D([0, 0], [0, 0], [0, 1.5], 'k', zorder=5, linewidth=3)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.grid(False)
    if not hold_show:
        plt.show()

def plot_data_S2(data, hold_show=False, elev=30, azim=45, title="data on S^2"):
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
    cmap = colors.LinearSegmentedColormap.from_list("", ["yellow", "white"])
    # sphere
    u, v = np.mgrid[0:2 * PI:30j, 0:PI:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', marker='o', s=1, zorder=10)
    ax.plot_surface(x, y, z, cmap=cmap, linewidth=0, antialiased=False, zorder=1)
    ax.set_box_aspect((np.ptp([1, -1]), np.ptp([1, -1]), np.ptp([1, -1])))
    ax.axes.set_xlim3d(left=-1.5, right=1.5)
    ax.axes.set_ylim3d(bottom=-1.5, top=1.5)
    ax.axes.set_zlim3d(bottom=-1.5, top=1.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    if not hold_show:
        plt.show()

def exp_SO3(s, theta):
    # s must be already normalised
    [x,y,z] = s
    ct = np.cos(theta)
    st = np.sin(theta)
    one_minus_ct = 1-ct
    return np.array([[ct + x * x * one_minus_ct, x * y * one_minus_ct - z * st, x * z * one_minus_ct + y * st],
                    [y * x * one_minus_ct + z * st, ct + y * y * one_minus_ct, y * z * one_minus_ct - x * st],
                    [z * x * one_minus_ct - y * st, z * y * one_minus_ct + x * st, ct + z * z * one_minus_ct]])

def log_S(b, x):
    d = np.arccos(np.dot(b,x))
    return d/np.sin(d)*(x - np.cos(d)*b)

def exp_S(b, y):
    tmp = LA.norm(y)
    return np.cos(tmp)*b + np.sin(tmp)*y/tmp

def correct_sign(b, x):
    if np.size(x) == np.size(b):
        if np.dot(b,x) < 0:
            return -x
        return x
    x_ = np.copy(x)
    for i in range(x.shape[0]):
        if np.dot(b, x_[i,:]) < 0:
            x_[i,:] = -x_[i,:]
    return x_

def arrow(axis, offset=1.0):
    # cone, axis is Z:
    h = 0.1
    r = 0.05
    num_points = 30
    zc = np.linspace(0, h, num_points)
    theta = np.linspace(0, 2*PI, num_points)
    zc, theta = np.meshgrid(zc, theta)
    rho = (r / h) * (h - zc)
    xc = rho * np.cos(theta)
    yc = rho * np.sin(theta)
    zc = zc + offset
    # lid:
    rho = np.linspace(0, r, num_points)
    theta = np.linspace(0, 2 * PI, num_points)
    rho, theta = np.meshgrid(rho, theta)
    xl = rho * np.cos(theta)
    yl = rho * np.sin(theta)
    zl = offset*np.ones_like(xl)
    x = np.vstack([xc, xl])
    y = np.vstack([yc, yl])
    z = np.vstack([zc, zl])
    if axis == 'x':
        return z, y, -x
    elif axis == 'y':
        return x, z, -y
    elif axis == 'z':
        return x, y, z
    else:
        raise Exception("axis must be 'x' or 'y' or 'z'")


