import numpy as np

WIDTH = 1280
HEIGHT = 720
STEP = 1
MAX_DEPTH = 2.0
GROUND_THRESH = 0.02

SX = WIDTH / 1280
SY = HEIGHT / 720

SCALE = 1000 # m-mm

INTRINSICS_MATRIX = np.array([
    [623.53829072,   0.    ,     640.        ],
    [0.     ,    623.53829072, 360.        ],
    [ 0.     ,      0.      ,     1.        ],
])

INTRINSICS_MATRIX[0, 0] *= SX
INTRINSICS_MATRIX[1, 1] *= SY
INTRINSICS_MATRIX[0, 2] *= SX
INTRINSICS_MATRIX[1, 2] *= SY 

EXTRINSIC_MATRIX = np.array([
    [ 0.       ,   1.        ,  0.       ,   0.        ],
    [ 0.4940094,  -0.        , -0.8694565,  -0.30628583],
    [-0.8694565,  -0.        , -0.4940094,   1.5511894 ],
    [ 0.       ,   0.        ,  0.       ,   1.        ],
    ]
)

WORK_SPACE = [
    [0.65, 1.1],
    [0.45, 0.66],
    [-0.7, 0]
]

check_point_cloud_path='../3D-Diffusion-Policy/infer/viusalize.json'