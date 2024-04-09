"""
noisebase.projective
--------------------

Compute various data transformations related to camera intrinsics, normals, and positions
"""
import numpy as np

def normalize(v):
    """Individually normalize an array of vectors

    Args:
        v (ndarray, CHW): will be normalized along first dimension
    """
    return v / np.linalg.norm(v, axis=0, keepdims=True)

class FlipRotate:
    """Class representing a flip/rotation data augmentation

    After initialization the transformation can be applied to many cameras and arrays
    """
    def __init__(
        self,
        orientation,
        height,
        width,
        window
    ):
        """
        Args:
            orientation (int): [0,7] representing the 8 possible axis-aligned flips and rotations. 0 is identity.
            height (int): height of the camera resolution (in pixels)
            width (int): width of the camera resolution (in pixels)
            window (int): height and width of the cropped image
        """
        self.orientation = orientation
        self.height = height
        self.width = width
        self.window = window

    def apply_camera(self, target, up, pos, p, offset):
        """Applies orientation change to camera intrinsics

        Args:
            target (ndarray, size (3)): a world-space point the camera is pointing at (center of the frame)
            up (ndarray, size (3)): vector in world-space that points upward in screen-space
            pos (ndarray, size (3)): the camera's position in world-space
            p (ndarray, size (4,4)): projective matrix of the camera e.g.
                [0.984375, 0.,   0.,      0.     ],
                [0.,       1.75, 0.,      0.     ],
                [0.,       0.,   1.0001, -0.10001],
                [0.,       0.,   1.,      0.     ]
            offset (ndarray, size (2)): offset of random crop (window) from top left corner of camera frame (in pixels)

        Returns:
            W (ndarray, size (3)): vector in world-space that points forward in screen-space
            V (ndarray, size (3)): vector in world-space that points up in screen-space
            U (ndarray, size (3)): vector in world-space that points right in screen-space
            pos (ndarray, size (3)): unchanged camera position
            offset (ndarray, size (2)): transformed offset, MAY BE NEGATIVE!
            pv (ndarray, size (4,4)): computed view-projection matrix, ROW VECTOR
        """

        # make orthonormal camera basis
        W = normalize(target - pos) # forward
        U = normalize(np.cross(W, up)) # right
        V = normalize(np.cross(U, W)) # up

        # flip rotate offset and basis
        if self.orientation % 2 < 1:
            U = -U
            offset[1] = self.width - offset[1] - self.window

        if self.orientation % 4 < 2:
            V = -V
            offset[0] = self.height - offset[0] - self.window

        if self.orientation % 8 < 4:
            U, V = V, U
            offset = (self.height + self.width)//2 - self.window - np.flip(offset)

        # view matrix for transformed camera basis
        view_basis = np.pad(np.stack([U, V, W], 0), [[0,1], [0,1]]) + np.diag([0.,0.,0.,1.])

        # view matrix for camera position
        view_translate = np.pad(-pos[:, np.newaxis], [[0, 1], [3, 0]]) + np.diag([1.,1.,1.,1.])

        # combined view matrix
        v = np.matmul(view_basis, view_translate)
        
        # view-projection matrix
        # DirectX ROW VECTOR ALERT!
        pv = np.matmul(v.T,p.T).astype(np.float32)
        
        return W, V, U, pos, offset, pv

    def apply_array(self, x):
        """Applies orientation change to per-pixel data

        Args:
            x (ndarray, CHW...): will be transformed, may have additional final dimensions
        """
        if self.orientation % 2 < 1:
            x = np.flip(x, 2)

        if self.orientation % 4 < 2:
            x = np.flip(x, 1)

        if self.orientation % 8 < 4:
            x = np.flip(np.transpose(x, [0, 2, 1] + list(range(3, x.ndim))), (1, 2))

        return np.ascontiguousarray(x)

def screen_space_normal(w_normal, W, V, U):
    """Transforms per-sample world-space normals to screen-space / relative to camera direction

    Args:
        w_normal (ndarray, 3HWS): per-sample world-space normals
        W (ndarray, size (3)): vector in world-space that points forward in screen-space
        V (ndarray, size (3)): vector in world-space that points up in screen-space
        U (ndarray, size (3)): vector in world-space that points right in screen-space
    
    Returns:
        normal (ndarray, 3HWS): per-sample screen-space normals
    """
    # TODO: support any number of extra dimensions like apply_array
    return np.einsum('ij, ihws -> jhws', np.stack([W, U, V], axis=1), w_normal) # column vectors

def screen_space_position(w_position, pv, height, width):
    """Projects per-sample world-space positions to screen-space (pixel coordinates)

    Args:
        w_normal (ndarray, 3HWS): per-sample world-space positions
        pv (ndarray, size (4,4)): camera view-projection matrix
        height (int): height of the camera resolution (in pixels)
        width (int): width of the camera resolution (in pixels)
    
    Returns:
        projected (ndarray, 2HWS): Per-sample screen-space position (pixel coordinates).
            IJ INDEXING! for gather ops and consistency, 
            see backproject_pixel_centers in noisebase.torch.projective for use with grid_sample.
            Degenerate positions give inf.
    """
    # TODO: support any number of extra dimensions like apply_array
    homogeneous = np.concatenate(( # Pad to homogeneous coordinates
        w_position,
        np.ones_like(w_position)[0:1]
    ))

    # ROW VECTOR ALERT!
    # DirectX uses row vectors...
    projected = np.einsum('ij, ihws -> jhws', pv, homogeneous)
    projected = np.divide(
        projected[0:2], projected[3], 
        out = np.zeros_like(projected[0:2]),
        where = projected[3] != 0
    )

    # directx pixel coordinate fluff
    projected = projected * np.reshape([0.5 * width, -0.5 * height], (2, 1, 1, 1)).astype(np.float32) \
        + np.reshape([width / 2, height / 2], (2, 1, 1, 1)).astype(np.float32)

    projected = np.flip(projected, 0) #height, width; ij indexing

    return projected

def motion_vectors(w_position, w_motion, pv, prev_pv, height, width):
    """Computes per-sample screen-space motion vectors (in pixels)

    Args:
        w_position (ndarray, 3HWS): per-sample world-space positions
        w_motion (ndarray, 3HWS): per-sample world-space positions
        pv (ndarray, size (4,4)): camera view-projection matrix
        prev_pv (ndarray, size (4,4)): camera view-projection matrix from previous frame
        height (int): height of the camera resolution (in pixels)
        width (int): width of the camera resolution (in pixels)
    
    Returns:
        motion (ndarray, 2HWS): Per-sample screen-space motion vectors (in pixels).
            IJ INDEXING! for gather ops and consistency, 
            see backproject_pixel_centers in noisebase.torch.projective for use with grid_sample.
            Degenerate positions give inf.
    """
    # TODO: support any number of extra dimensions like apply_array (only the docstring here)
    current = screen_space_position(w_position, pv, height, width)
    prev = screen_space_position(w_position+w_motion, prev_pv, height, width)

    motion = prev-current

    return motion

def log_depth(w_position, pos):
    """Computes per-sample compressed depth (disparity-ish)

    Args:
        w_position (ndarray, 3HWS): per-sample world-space positions
        pos (ndarray, size (3)): the camera's position in world-space
    
    Returns:
        motion (ndarray, 1HWS): per-sample compressed depth
    """
    # TODO: support any number of extra dimensions like apply_array
    d = np.linalg.norm(w_position - np.reshape(pos, (3, 1, 1, 1)), axis=0, keepdims=True)
    return np.log(1 + 1/d)