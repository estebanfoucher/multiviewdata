"""
precision_analysis.py

Class: PrecisionAnalysis

Purpose:
 - Triangulate a 3D point from two corresponding image points and the cameras' intrinsics/extrinsics.
 - Linearize the triangulation to compute a 3D covariance from pixel measurement noise.
 - Compute sensitivity coefficients (meters per pixel) describing how a 1-pixel error in an image point moves the reconstructed 3D point.

Notes:
 - This implementation uses the linear DLT triangulation (SVD) and a numerical Jacobian (central differences) to avoid brittle analytic derivations.
 - If OpenCV is available, the undistortion helpers will use it. If not, the code can operate on already-undistorted points.

API (high level):

PrecisionAnalysis(K1, K2, R, t, dist1=None, dist2=None)
 - K1, K2: 3x3 numpy arrays (intrinsic matrices)
 - R: 3x3 rotation that maps points from camera1 frame to camera2 frame (X2 = R X1 + t)
 - t: 3x1 translation vector from camera1 to camera2 (in meters)
 - dist1, dist2: optional distortion coefficient arrays (radial/tangential)

Main methods:
 - triangulate(pt1, pt2) -> X (3D point in camera1 frame, in meters)
 - covariance_for_pixel_noise(pt1, pt2, sigma_pixels) -> Cov3x3 (3D covariance)
 - sensitivity_wrt_point(pt1, pt2, which=2, eps=1e-3) -> dict with per-axis and magnitude meters/pixel
 - summary(pt1, pt2, sigma_pixels) -> dict with 3D point, covariance, stds, radius_1sigma, radius_95, sensitivity

Example usage is included at bottom.
"""

import numpy as np
from numpy.linalg import svd, inv, norm

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


class PrecisionAnalysis:
    def __init__(self, K1, K2, R, t, dist1=None, dist2=None, image_size=None):
        """Initialize with calibration data.

        K1, K2: (3,3)
        R: (3,3)
        t: (3,) or (3,1)
        dist*: optional, passed to OpenCV undistort if available (as 1xN or flat array)
        """
        self.K1 = np.asarray(K1, dtype=float)
        self.K2 = np.asarray(K2, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.t = np.asarray(t, dtype=float).reshape(3)
        self.dist1 = None if dist1 is None else np.asarray(dist1).reshape(-1)
        self.dist2 = None if dist2 is None else np.asarray(dist2).reshape(-1)
        self.image_size = image_size

        # Build projection matrices P1 = K1 [I|0], P2 = K2 [R|t]
        self.P1 = np.hstack((self.K1, np.zeros((3,1))))
        Rt = np.hstack((self.R, self.t.reshape(3,1)))
        self.P2 = self.K2.dot(Rt)

    def undistort_point(self, pt, K, dist):
        """Undistort a single pixel point (u,v).

        If OpenCV is available it will be used. Otherwise this function returns the input point
        unchanged (assumes points are already undistorted).
        """
        if dist is None or not _HAS_CV2:
            # Can't undistort; assume user passed undistorted points.
            return np.asarray(pt, dtype=float)
        uv = np.array([[pt]], dtype=float)  # shape (1,1,2)
        und = cv2.undistortPoints(uv, K, dist, P=K)
        return und.ravel()

    def triangulate_linear(self, pt1, pt2, undistort=True):
        """Triangulate using linear DLT (SVD). Returns 3D point in camera1 coordinates.

        pt1, pt2: (u,v) pixel coordinates
        """
        p1 = tuple(pt1)
        p2 = tuple(pt2)
        if undistort:
            if self.dist1 is not None and _HAS_CV2:
                p1 = tuple(self.undistort_point(p1, self.K1, self.dist1))
            if self.dist2 is not None and _HAS_CV2:
                p2 = tuple(self.undistort_point(p2, self.K2, self.dist2))

        u1, v1 = p1
        u2, v2 = p2
        # build A such that A X = 0, X is homogeneous 4-vector
        A = np.zeros((4,4))
        A[0] = u1 * self.P1[2] - self.P1[0]
        A[1] = v1 * self.P1[2] - self.P1[1]
        A[2] = u2 * self.P2[2] - self.P2[0]
        A[3] = v2 * self.P2[2] - self.P2[1]

        # Solve by SVD
        U, S, Vt = svd(A)
        Xh = Vt[-1]
        X = Xh[:3] / Xh[3]
        return X

    def _triangulate_wrapper(self, uv_flat, undistort=True):
        """Helper used by numerical Jacobian: uv_flat = [u1,v1,u2,v2]"""
        pt1 = (uv_flat[0], uv_flat[1])
        pt2 = (uv_flat[2], uv_flat[3])
        return self.triangulate_linear(pt1, pt2, undistort=undistort)

    def numerical_jacobian(self, pt1, pt2, eps=1e-3, undistort=True):
        """Compute numerical Jacobian dX / d[u1,v1,u2,v2].

        eps in pixels (small perturbation). Central differences used.
        Returns J shape (3,4)
        """
        uv0 = np.array([pt1[0], pt1[1], pt2[0], pt2[1]], dtype=float)
        J = np.zeros((3,4), dtype=float)
        f0 = self._triangulate_wrapper(uv0, undistort=undistort)
        for i in range(4):
            du = np.zeros_like(uv0)
            du[i] = eps
            xp = self._triangulate_wrapper(uv0 + du, undistort=undistort)
            xm = self._triangulate_wrapper(uv0 - du, undistort=undistort)
            J[:,i] = (xp - xm) / (2*eps)
        return J

    def covariance_for_pixel_noise(self, pt1, pt2, sigma_pixels=0.389, eps=1e-3, undistort=True, pedantic=False):
        """Compute 3x3 covariance of reconstructed 3D point given per-pixel isotropic Gaussian noise.

        sigma_pixels: standard deviation in pixels (applies to both u and v in both cameras).
        Returns (Cov3x3, J)
        """
        J = self.numerical_jacobian(pt1, pt2, eps=eps, undistort=undistort)
        # Pixel covariance: assume independent and identical for u1,v1,u2,v2
        Sigma_pix = (sigma_pixels**2) * np.eye(4)
        Cov3 = J.dot(Sigma_pix).dot(J.T)

        if pedantic:
            return Cov3, J
        return Cov3

    def sensitivity_wrt_point(self, pt1, pt2, which=2, eps=1e-3, undistort=True):
        """Return sensitivity (meters per pixel) of 3D point to a pixel change in one of the image points.

        which: 1 or 2 (first or second camera point). Returns dict:
          {'du': value_m_per_pixel, 'dv': value_m_per_pixel, 'magnitude_du':, 'magnitude_dv':}
        """
        J = self.numerical_jacobian(pt1, pt2, eps=eps, undistort=undistort)
        if which == 1:
            jac = J[:, 0:2]
        elif which == 2:
            jac = J[:, 2:4]
        else:
            raise ValueError('which must be 1 or 2')
        # Columns are partial derivatives wrt u and v
        du_vec = jac[:,0]
        dv_vec = jac[:,1]
        return {
            'du_vec': du_vec,
            'dv_vec': dv_vec,
            'du_m_per_pixel': norm(du_vec),
            'dv_m_per_pixel': norm(dv_vec),
            'combined_magnitude': norm(jac, ord='fro')  # rms-like
        }

    def summary(self, pt1, pt2, sigma_pixels=0.389, eps=1e-3, undistort=True):
        """Convenience: triangulate, compute covariance, stds, radii, sensitivities.

        Returns dict with keys: X (3,), Cov (3x3), stds (3,), radii (1sigma, 95%), sens1, sens2, depth_sensitivity
        """
        X = self.triangulate_linear(pt1, pt2, undistort=undistort)
        Cov, J = None, None
        Cov = self.covariance_for_pixel_noise(pt1, pt2, sigma_pixels=sigma_pixels, eps=eps, undistort=undistort)
        # Eigen-analysis
        w, V = np.linalg.eigh(Cov)
        # standard deviations along principal axes
        stds = np.sqrt(np.maximum(w, 0.0))
        # 1-sigma spherical-equivalent radius (RMS)
        rms = np.sqrt(np.trace(Cov)/3.0)
        # 95% confidence approx assuming Gaussian for each axis: r95 ~= 1.96 * rms
        r95 = 1.96 * rms
        sens1 = self.sensitivity_wrt_point(pt1, pt2, which=1, eps=eps, undistort=undistort)
        sens2 = self.sensitivity_wrt_point(pt1, pt2, which=2, eps=eps, undistort=undistort)
        
        # Compute depth sensitivity (sensitivity of depth = norm(X) to pixel changes)
        depth = norm(X)
        if depth > 0:
            # Unit vector in direction of X
            X_unit = X / depth
            # Sensitivity of depth to changes in X: d(depth)/dX = X_unit
            # Then sensitivity to pixel changes: d(depth)/dpixel = X_unit^T * J
            J = self.numerical_jacobian(pt1, pt2, eps=eps, undistort=undistort)
            depth_sensitivity = X_unit.dot(J)  # Shape (4,) - sensitivity to [u1, v1, u2, v2]
            depth_sensitivity_magnitude = norm(depth_sensitivity)
        else:
            depth_sensitivity = np.zeros(4)
            depth_sensitivity_magnitude = 0.0
            
        return {
            'X': X,
            'Cov': Cov,
            'stds_principal_axes': stds,
            'rms_radius_1sigma': rms,
            'radius_95percent': r95,
            'sensitivity_cam1': sens1,
            'sensitivity_cam2': sens2,
            'depth': depth,
            'depth_sensitivity': depth_sensitivity,
            'depth_sensitivity_magnitude': depth_sensitivity_magnitude
        }


# Example usage (replace with your matched pixel coordinates):
if __name__ == '__main__':
    # This example only shows structure; fill in your own points.
    import json
    extrinsics_path = "stereo_michel_24_08_2025/scene_3/extrinsics_calibration.json"
    with open(extrinsics_path, "r") as f:
        extrinsics = json.load(f)
    K1 = np.array(extrinsics["camera_matrix1"])
    K2 = np.array(extrinsics["camera_matrix2"])
    R = np.array(extrinsics["rotation_matrix"])
    t = np.array(extrinsics["translation_vector"])
    dist1 = np.array(extrinsics["dist_coeffs1"])
    dist2 = np.array(extrinsics["dist_coeffs2"])
    
    pa = PrecisionAnalysis(K1, K2, R, t, dist1=dist1, dist2=dist2)

    # Example matched pixel coordinates (replace with your real points):
    pt1 = (1553.0, 397.0)
    pt2 = (782.0, 677.0)
    #pt1 = (1218.0, 741.0)
    #pt2 = (537.0, 500.0)

    s = pa.summary(pt1, pt2, sigma_pixels=0.8)
    print('Triangulated X (m):', s['X'])
    print('Covariance (m^2):\n', s['Cov'])
    print('1-sigma RMS radius (m):', s['rms_radius_1sigma'])
    print('95% radius (m):', s['radius_95percent'])
    print('Sensitivity cam2 (m/pixel):', s['sensitivity_cam2']['du_m_per_pixel'], s['sensitivity_cam2']['dv_m_per_pixel'])
    print('Depth (m):', s['depth'])
    print('Depth sensitivity (m/pixel):', s['depth_sensitivity_magnitude'])
