"""MediaPipe index constants used across the pipeline."""

# 52 face blendshape names in MediaPipe FaceLandmarker output order.
BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
BS = {n: i for i, n in enumerate(BLENDSHAPE_NAMES)}


def _bs_flip_perm() -> list[int]:
    """Blendshape permutation for a horizontal image flip (Left <-> Right)."""
    perm = []
    for n in BLENDSHAPE_NAMES:
        if n.endswith("Left"):
            perm.append(BS[n[:-4] + "Right"])
        elif n.endswith("Right"):
            perm.append(BS[n[:-5] + "Left"])
        else:
            perm.append(BS[n])
    return perm


BS_FLIP = _bs_flip_perm()

# Pose landmark indices (33-point BlazePose topology).
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
POSE_UPPER = list(range(0, 23))  # head, shoulders, arms, hand proxies

_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
POSE_FLIP = list(range(33))
for a, b in _POSE_PAIRS:
    POSE_FLIP[a], POSE_FLIP[b] = b, a

# Hand landmark indices (21 points).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
FINGER_MCP = [5, 9, 13, 17]      # index, middle, ring, pinky
FINGER_PIP = [6, 10, 14, 18]
FINGER_TIP = [8, 12, 16, 20]

# Face mesh indices.
FACE_EYE_OUTER_L, FACE_EYE_OUTER_R = 263, 33   # subject's left / right outer eye corner
FACE_NOSE_TIP = 1
FACE_FOREHEAD = 10
FACE_CHIN = 152
