import bbox_visualizer as bbv
import cv2
import numpy as np
from einops import rearrange, reduce


LABELMAP_GEN1 = {'car': 0, 'pedestrian': 1}
LABELMAP_GEN4 = {'pedestrian': 0, 'two wheeler': 1, 'car': 2, 'truck': 3, 'bus': 4, 'traffic sign': 5, 'traffic light': 6}
LABELMAP_GEN4_SHORT = {'pedestrian': 0, 'two wheeler': 1, 'car': 2}
LABELMAP_VGA = {'car': 0, 'pedestrian': 1, 'two wheeler': 2, 'truck': 3, 'bus': 4, 'traffic sign': 5, 'traffic light': 6, 'other': 7}

CLASS_MAP_SEVD = {
    'car': 0, 'van': 0, 'truck': 0,
    'pedestrian': 1,
    'cyclist': 2, 'bicycle': 2, 'motorcycle': 2,
    'misc': 3,
}

classid2colors = {
    0: (0, 0, 255),  # ped -> blue (rgb)
    1: (0, 255, 255),  # 2-wheeler cyan (rgb)
    2: (255, 255, 0),  # car -> yellow (rgb)
    3: (255, 0, 0),  # truck -> red (rgb)
    4: (255, 0, 255),  # bus -> magenta (rgb)
    5: (0, 255, 0),  # traffic sign -> green (rgb)
    6: (0, 0, 0),  # traffic light -> black (rgb)
    7: (255, 255, 255),  # other -> white (rgb)
}

dataset2labelmap = {
    "gen1": LABELMAP_GEN1,
    "gen4": LABELMAP_GEN4_SHORT,
    "VGA": LABELMAP_VGA,
    "SEVD": CLASS_MAP_SEVD,
}

dataset2scale = {
    "gen1": 1,
    "gen4": 1,
    "VGA": 1,
    "SEVD": 1.0,
}

dataset2size = {
    "gen1": (304*1, 240*1),
    "gen4": (640*1, 360*1),
    "VGA": (640*1, 480*1),
    "SEVD": (640*1, 480*1),
}


def ev_repr_to_img(x: np.ndarray, repr_type: str = 'histogram'):
    ch, ht, wd = x.shape[-3:]

    if repr_type == 'voxel_grid':
        img_diff = np.asarray(reduce(x, 'C H W -> H W', 'sum'), dtype='int32')
    
    else:
        assert ch > 1 and ch % 2 == 0
        ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
        img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
        img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
        img_diff = img_pos - img_neg

    img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = 255  # 合計がプラスなら白
    img[img_diff < 0] = 0    # 合計がマイナスなら黒
    
    return img

def draw_bboxes_with_id(img, boxes, dataset_name: str) -> None:
    """
    画像 img にバウンディングボックスを描画する関数
    """
    # カラーマップの生成（描画に使う色のリスト）
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    labelmap = dataset2labelmap[dataset_name]
    scale_multiplier = dataset2scale[dataset_name]

    add_score = True
    ht, wd, ch = img.shape
    dim_new_wh = (int(wd * scale_multiplier), int(ht * scale_multiplier))
    if scale_multiplier != 1:
        img = cv2.resize(img, dim_new_wh, interpolation=cv2.INTER_AREA)
    
    # boxes の各要素は (cls_id, cx, cy, w, h) としてループ
    if len(boxes[0]) == 5:
        for cls_id, cx, cy, w, h in boxes:
            score = 1.0

            # (cx, cy) を中心座標とした左上座標を計算
            pt1 = (int(cx - w / 2), int(cy - h / 2))
            pt2 = (int(cx + w / 2), int(cy + h / 2))
            bbox = (pt1[0], pt1[1], pt2[0], pt2[1])

            # スケール補正
            bbox = tuple(int(x * scale_multiplier) for x in bbox)

            class_id = int(cls_id)
            class_name = labelmap[class_id % len(labelmap)]
            bbox_txt = class_name
            if add_score:
                bbox_txt += f' {score:.2f}'
            color_tuple_rgb = classid2colors[class_id]
            img = bbv.draw_rectangle(img, bbox, bbox_color=color_tuple_rgb)
            img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color_tuple_rgb, top=True)

    elif len(boxes[0]) == 7:
        for x1, y1, x2, y2, obj_conf, class_conf, class_id in boxes:
            score = obj_conf * class_conf

            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            bbox = (pt1[0], pt1[1], pt2[0], pt2[1])

            bbox = tuple(int(x * scale_multiplier) for x in bbox)

            class_id = int(class_id)
            class_name = labelmap[class_id % len(labelmap)]
            bbox_txt = class_name
            if add_score:
                bbox_txt += f' {score:.2f}'
            color_tuple_rgb = classid2colors[class_id]
            img = bbv.draw_rectangle(img, bbox, bbox_color=color_tuple_rgb)
            img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color_tuple_rgb, top=True)
    else:
        raise ValueError("Invalid boxes format")
    
    return img


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, max_flow=None):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if max_flow is not None:
        mag = np.sqrt(np.square(flow_uv[:,:,0]) + np.square(flow_uv[:,:,1]))
        flow_uv[mag>max_flow, 0] = 0
        flow_uv[mag>max_flow, 1] = 0

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)