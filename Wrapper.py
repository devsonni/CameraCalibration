import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import scipy.optimize

class CameraCalibration:
    def __init__(self, nx, ny, square_size, img_dir):
        self.nx = nx
        self.ny = ny
        self.square_size = square_size
        self.img_dir = img_dir
        self.img_points_set = []
        self.world_points_set = []
        self.H_matrix_set = []
        self.num_images = 0

    def get_img_points(self, image):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
        
        if ret:
            corners = corners.reshape(-1, 2)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return np.array([corners])
        return None

    def get_world_points(self):
        x, y = np.meshgrid(np.linspace(0, self.nx - 1, self.nx), np.linspace(0, self.ny - 1, self.ny))
        x = np.flip((x.reshape(-1, 1) * self.square_size), axis=0)
        y = (y.reshape(-1, 1) * self.square_size)
        return np.float32(np.hstack((y, x)))

    @staticmethod
    def get_homography(points1, points2):
        H, _ = cv2.findHomography(points1, points2)
        return H

    @staticmethod
    def compute_Vij(H, i, j):
        i, j = i - 1, j - 1
        v_ij = np.array([H[0, i] * H[0, j],
                         H[0, i] * H[1, j] + H[1, i] * H[0, j],
                         H[1, i] * H[1, j],
                         H[2, i] * H[0, j] + H[0, i] * H[2, j],
                         H[2, i] * H[1, j] + H[1, i] * H[2, j],
                         H[2, i] * H[2, j]])
        return v_ij

    @staticmethod
    def compute_V(H):
        V = []
        for h in H:
            V.append(CameraCalibration.compute_Vij(h, 1, 2).T)
            V.append((CameraCalibration.compute_Vij(h, 1, 1) - CameraCalibration.compute_Vij(h, 2, 2)).T)
        return np.array(V)

    @staticmethod
    def compute_B(V):
        _, _, v = np.linalg.svd(V)
        return v[-1, :]

    def compute_K(self, H):
        V = self.compute_V(H)
        b = self.compute_B(V)
        b11, b12, b22, b13, b23, b33 = b

        v0 = (b12 * b13 - b11 * b23) / (b11 * b22 - b12 ** 2)
        lamda = b33 - (b13 ** 2 + v0 * (b12 * b13 - b11 * b23)) / b11
        alpha = np.sqrt(lamda / b11)
        beta = np.sqrt(lamda * b11 / (b11 * b22 - b12 ** 2))
        gamma = -b12 * alpha ** 2 * beta / lamda
        u0 = gamma * v0 / beta - b13 * alpha ** 2 / lamda

        K = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [0, 0, 1]])
        return K

    @staticmethod
    def compute_Rt(K, H):
        extrinsic = []
        K_inv = np.linalg.inv(K)
        for h in H:
            h1, h2, h3 = h.T
            lamda = 1 / np.linalg.norm(K_inv.dot(h1), ord=2)
            r1 = lamda * K_inv.dot(h1)
            r2 = lamda * K_inv.dot(h2)
            r3 = np.cross(r1, r2)
            t = lamda * K_inv.dot(h3)
            RT = np.vstack((r1, r2, r3, t)).T
            extrinsic.append(RT)
        return extrinsic

    @staticmethod
    def projection(initial_params, world_points, RT):
        alpha, beta, gamma, u0, v0, k1, k2 = initial_params
        K = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [0, 0, 1]])
        m_i_ = []
        for M in world_points:
            M = np.float64(np.hstack((M, 0, 1)))
            projected_pt = np.dot(RT, M)
            projected_pt = projected_pt / projected_pt[-1]

            x, y = projected_pt[0], projected_pt[1]
            r = x ** 2 + y ** 2

            uv = np.dot(K, projected_pt)
            u = uv[0] / uv[-1]
            v = uv[1] / uv[-1]

            u_hat = u + (u - u0) * (k1 * r + k2 * (r ** 2))
            v_hat = v + (v - v0) * (k1 * r + k2 * (r ** 2))

            m_i_.append([u_hat, v_hat])
        return np.array(m_i_)

    def reprojection_error(self, initial_params, world_points, img_points_set, RT):
        final_error = []
        for i, RT3 in enumerate(RT):
            mi_hat = self.projection(initial_params, world_points[i], RT3)
            mi = img_points_set[i].reshape(-1, 2)

            error = [np.linalg.norm(m - m_, ord=2) for m, m_ in zip(mi, mi_hat.squeeze())]
            final_error.append(np.mean(error))
        return final_error

    def loss(self, initial_params, world_points, img_points_set, RT):
        final_error = []
        for i, RT3 in enumerate(RT):
            mi_hat = self.projection(initial_params, world_points[i], RT3)
            mi = img_points_set[i].reshape(-1, 2)

            error = [np.linalg.norm(m - m_, ord=2) for m, m_ in zip(mi, mi_hat.squeeze())]
            final_error.append(np.sum(error))
        return final_error

    def optimize(self, initial_params, world_points_set, img_points_set, RT):
        opt = scipy.optimize.least_squares(
            fun=self.loss, x0=initial_params, method="lm", args=(world_points_set, img_points_set, RT))
        params = opt.x

        alpha, beta, gamma, u0, v0, k1, k2 = params
        K_new = np.array([[alpha, gamma, u0],
                          [0, beta, v0],
                          [0, 0, 1]])
        kc = (k1, k2)

        return K_new, kc

    def calibrate(self):
        for image_path in sorted(glob.glob(f"{self.img_dir}/*.jpg")):
            img = cv2.imread(image_path)
            img_points = self.get_img_points(img)
            if img_points is not None:
                world_points = self.get_world_points()
                H = self.get_homography(world_points, img_points[0])
                self.img_points_set.append(img_points)
                self.world_points_set.append(world_points)
                self.H_matrix_set.append(H)
                self.num_images += 1

        K_init = self.compute_K(self.H_matrix_set)
        print("The intrinsic matrix K is:\n", K_init)

        RT = self.compute_Rt(K_init, self.H_matrix_set)
        print("The extrinsic matrix [R|t] is:\n", RT[0])

        initial_params = [K_init[0, 0], K_init[1, 1], K_init[0, 1], K_init[0, 2], K_init[1, 2], 0, 0]
        projection_error = self.reprojection_error(initial_params, self.world_points_set, self.img_points_set, RT)
        print("Projection error:\n", np.mean(projection_error))

        K_new, kc = self.optimize(initial_params, self.world_points_set, self.img_points_set, RT)
        print("The new intrinsic matrix K is:\n", K_new)
        print("kc is:\n", kc)

        RT_new = self.compute_Rt(K_new, self.H_matrix_set)
        print("The new extrinsic matrix [R|t] is:\n", RT_new[0])

        new_img_points = []
        for i, rt in enumerate(RT_new):
            world_point = np.column_stack((self.world_points_set[i], np.ones(len(self.world_points_set[i]))))
            r1, r2, r3, t = rt.T
            R = np.stack((r1, r2, r3), axis=1)
            t = t.reshape(-1, 1)
            img_pt, _ = cv2.projectPoints(world_point, R, t, K_new, (kc[0], kc[1], 0, 0))
            new_img_points.append(img_pt.squeeze())

        final_params = [K_new[0, 0], K_new[1, 1], K_new[0, 1], K_new[0, 2], K_new[1, 2], kc[0], kc[1]]
        projection_error_new = self.reprojection_error(final_params, self.world_points_set, new_img_points, RT_new)
        print("Reprojection error:\n", np.mean(projection_error_new))

        distortion = np.array([kc[0], kc[1], 0, 0, 0], dtype=float)
        for i, image_path in enumerate(sorted(glob.glob(f"{self.img_dir}/*.jpg"))):
            img = cv2.imread(image_path)
            img = cv2.undistort(img, K_new, distortion)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)
            if ret:
                img = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                plt.imshow(img)
                plt.savefig(f'result{i}.png')

if __name__ == "__main__":
    nx = 9  # Number of inside corners in x
    ny = 6  # Number of inside corners in y
    square_size = 21.5  # 21.5mm
    img_dir = "Calibration_Imgs"

    calibrator = CameraCalibration(nx, ny, square_size, img_dir)
    calibrator.calibrate()
