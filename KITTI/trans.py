

import numpy as np
import open3d as o3d


'''
解析 KITTI 数据集的点云文件和标签文件
尝试基本的可视化

'''



def load_velodyne_bin(bin_path):
    """读取 KITTI .bin 点云文件
    
    Args:
        bin_path: .bin 文件路径
        
    Returns:
        points: (N, 4) numpy 数组, 每行 [x, y, z, intensity]
    """
    points = np.fromfile(bin_path, dtype=np.float32)
    points = points.reshape(-1, 4)
    return points


# 例如 .../label_2/000008.txt

def load_kitti_labels(label_path):
    """读取 KITTI 标签文件
    
    Args:
        label_path: .txt 标签文件路径
        
    Returns:
        objects: list of dict, 每个dict是一个标注目标
    """
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 15:
                continue
            obj = {
                'type': parts[0],                               # 类别: Car/Pedestrian/Cyclist/...
                'truncated': float(parts[1]),                   # 截断程度 0~1
                'occluded': int(parts[2]),                      # 遮挡等级 0~3
                'alpha': float(parts[3]),                       # 观测角(弧度)
                'bbox2d': np.array(parts[4:8], dtype=np.float32),  # 2D框 [l,t,r,b] 像素
                'dims': np.array(parts[8:11], dtype=np.float32),   # 3D尺寸 [h,w,l] 米
                'loc': np.array(parts[11:14], dtype=np.float32),   # 3D中心 [x,y,z] 米
                'ry': float(parts[14]),                        # 偏航角(弧度)
            }
            objects.append(obj)
    return objects

# loc 是底部中心点。KITTI 约定 3D 框的 y 坐标在物体底面（地面位置）
# 同时 KITTI y 是向下的，后续我们得到 8 角点坐标时，需要将 y 转换为向上


def load_calib(calib_path):
    """读取 KITTI 标定文件
    
    Args:
        calib_path: .txt 标定文件路径
        
    Returns:
        P2: (3,4) 左彩色相机投影矩阵 — 内参+外参合并
        R0: (3,3) 畸变校正旋转矩阵 — 相机内部校正
        V2C: (3,4) LiDAR→相机的外参变换矩阵
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) < 10:
                continue
            key, val = line.split(':', 1)
            calib[key.strip()] = np.array(
                val.strip().split(), dtype=np.float32)
    
    P2 = calib['P2'].reshape(3, 4)
    R0 = calib['R0_rect'].reshape(3, 3)
    V2C = calib['Tr_velo_to_cam'].reshape(3, 4)
    return P2, R0, V2C

# 理解标定中的 P2：这是链路中的最后一步 校正后相机 3D → 图像 2D 像素


# 正向转换
def lidar_to_rect(pts_lidar, V2C, R0):
    """LiDAR 3D点 → 校正后相机坐标

    Args:
        pts_lidar: (N, 3) 或 (N, 4) LiDAR 坐标系下的点
        V2C: (3, 4) Tr_velo_to_cam 外参矩阵
        R0: (3, 3) R0_rect 校正矩阵
        
    Returns:
        pts_rect: (N, 3) 校正后相机坐标系下的点
    """
    # 取前3列 (x,y,z)，补1变齐次坐标 (N,4)
    pts_xyz = pts_lidar[:, :3] if pts_lidar.shape[1] == 4 else pts_lidar
    N = pts_xyz.shape[0]
    pts_4d = np.hstack([pts_xyz, np.ones((N, 1), dtype=np.float32)])  # (N,4)
    
    # 步骤1: V2C 外参变换 — LiDAR → 原始相机坐标
    pts_cam = (V2C @ pts_4d.T).T    # (N,3)
    
    # 步骤2: R0 校正 — 原始相机 → 校正后相机
    pts_rect = (R0 @ pts_cam.T).T  # (N,3)
    
    return pts_rect


# 逆向转换
def rect_to_lidar(pts_rect, V2C, R0):
    """校正后相机坐标 → LiDAR 坐标
    
    Args:
        pts_rect: (N, 3) 校正后相机坐标系下的点
        V2C: (3, 4) 外参矩阵
        R0: (3, 3) 校正矩阵
        
    Returns:
        pts_lidar: (N, 3) LiDAR 坐标系下的点
    """
    N = pts_rect.shape[0]
    pts_4d = np.hstack([pts_rect, np.ones((N, 1), dtype=np.float32)])
    
    # 将 V2C (3x4) 补齐为 4x4 齐次矩阵
    V2C_4x4 = np.eye(4, dtype=np.float32)
    V2C_4x4[:3, :] = V2C
    
    # 将 R0 (3x3) 补齐为 4x4 齐次矩阵
    R0_4x4 = np.eye(4, dtype=np.float32)
    R0_4x4[:3, :3] = R0
    
    # 合并: RV = R0 @ V2C (4x4)，然后求逆
    RV = R0_4x4 @ V2C_4x4
    RV_inv = np.linalg.inv(RV)
    
    # 逆变换: pts_lidar = RV_inv @ pts_rect
    pts_lidar = (RV_inv @ pts_4d.T).T
    return pts_lidar[:, :3]


def compute_box_3d_cam(obj):
    """在相机坐标系下计算3D框的8个角点

    直接使用 KITTI 标注的原生参数:
      - dims: [h, w, l] (无需重排)
      - loc:  底部中心 (无需转几何中心)
      - ry:   绕相机Y轴 (无需转heading)

    相机坐标系: x右, y下, z前
      - l(长) 沿 x 轴
      - h(高) 沿 y 轴 (y向下, 所以顶部 y = -h)
      - w(宽) 沿 z 轴

    Args:
        obj: dict, 单个标注目标 (来自 load_kitti_labels)

    Returns:
        corners_3d: (8, 3) 校正后相机坐标系下的8个角点
    """
    h, w, l = obj['dims']        # KITTI顺序: [h, w, l], 直接使用
    x, y, z = obj['loc']         # 底部中心, 相机坐标系
    ry = obj['ry']               # 绕Y轴偏航角, 直接使用

    # 8个角点的局部坐标 (相对于底部中心, 未旋转)
    # 底面 (y=0):   角点 0,1,2,3  — loc的y值就是底面 因此在本地坐标系下 y 都为 0
    # 顶面 (y=-h):  角点 4,5,6,7  — y向下, 顶部y更小(更负)
    x_corners = np.array([l/2, l/2, -l/2, -l/2,
                          l/2, l/2, -l/2, -l/2])
    y_corners = np.array([0, 0, 0, 0,
                          -h, -h, -h, -h])       # 底面0, 顶面-h
    z_corners = np.array([w/2, -w/2, -w/2, w/2,
                          w/2, -w/2, -w/2, w/2])

    # 绕 Y 轴旋转矩阵 (ry 的原生定义, 无需转换)
    # 旋转发生在 x-z 平面, y 轴不变
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # 旋转: R @ (3, 8) → (3, 8)
    corners = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
    corners_3d = R @ corners                                  # 旋转后 (3, 8)

    # 平移: 加上底部中心坐标
    corners_3d[0, :] += x       # x方向平移
    corners_3d[1, :] += y       # y方向平移
    corners_3d[2, :] += z       # z方向平移

    return corners_3d.T  # (8, 3)


def box_cam_to_lidar_v2(obj, V2C, R0):
    """方案二: 先在相机坐标系算8角点, 再整体转到LiDAR

    流程:
      1. compute_box_3d_cam: 在相机坐标系下用 ry 算8角点
      2. rect_to_lidar:      将8角点批量从相机转到LiDAR

    Args:
        obj: dict, 单个标注目标
        V2C: (3,4) 外参矩阵
        R0: (3,3) 校正矩阵

    Returns:
        corners_lidar: (8, 3) LiDAR坐标系下的8个角点
    """
    # Step 1: 在相机坐标系下计算8个角点
    corners_cam = compute_box_3d_cam(obj)      # (8, 3) 相机坐标

    # Step 2: 将8个角点整体转到LiDAR坐标系
    corners_lidar = rect_to_lidar(corners_cam, V2C, R0)  # (8, 3) LiDAR坐标

    return corners_lidar



def rect_to_img(pts_rect, P2):
    """校正后相机3D点 → 图像2D像素坐标
    
    使用内参 P2 进行投影。P2 = K @ [R|t]，已包含内参K。
    
    Args:
        pts_rect: (N, 3) 校正后相机坐标系下的3D点
        P2: (3, 4) 投影矩阵 (内参+外参合并)
        
    Returns:
        pts_img: (N, 2) 图像像素坐标 [u, v]
        depth: (N,) 每个点的深度 (相机到点的Z距离)
    """
    N = pts_rect.shape[0]
    # 补1 → 齐次坐标 (N, 4)
    pts_4d = np.hstack([pts_rect, np.ones((N, 1), dtype=np.float32)])
    
    # P2 投影: (3,4) @ (4,N) → (3,N)
    # 结果每列: [u*depth, v*depth, depth]
    pts_2d_hom = (P2 @ pts_4d.T).T  # (N, 3)
    
    # 透视除法: depth 是
    depth = pts_2d_hom[:, 2].copy()
    pts_img = pts_2d_hom[:, :2].copy()
    pts_img[:, 0] /= depth   # u = u*depth / depth
    pts_img[:, 1] /= depth   # v = v*depth / depth
    
    return pts_img, depth

# depth 归一化是必要的，可以从数学上进行推导


# lidar_to_img 把外参（V2C+R0）和内参（P2）串联成完整链路：像素 = P2 · R0 · V2C · LiDAR点
def lidar_to_img(pts_lidar, P2, R0, V2C):
    """LiDAR 3D点 → 图像2D像素 (完整投影链路)
    
    串联外参(V2C+R0)和内参(P2)三步:
      1. V2C: LiDAR → 原始相机坐标  [外参]
      2. R0:  原始 → 校正后相机坐标  [校正]
      3. P2:  校正后相机 → 图像像素  [内参]
    
    Args:
        pts_lidar: (N, 3) 或 (N, 4) LiDAR坐标点
        P2: (3, 4) 投影矩阵
        R0: (3, 3) 校正矩阵
        V2C: (3, 4) 外参矩阵
        
    Returns:
        pts_img: (N, 2) 图像像素 [u, v]
        depth: (N,) 深度
    """
    # 步骤1+2: 外参 — LiDAR → 校正后相机
    pts_rect = lidar_to_rect(pts_lidar, V2C, R0)
    
    # 步骤3: 内参 — 校正后相机 → 图像像素
    pts_img, depth = rect_to_img(pts_rect, P2)
    
    return pts_img, depth



def visualize_bev(points, boxes_lidar, sample_id='000008'):
    """BEV鸟瞰图可视化: 点云 + 3D框
    
    Args:
        points: (N, 4) LiDAR点云 [x,y,z,intensity]
        boxes_lidar: list of dict, 每个含 'corners'(8,3), 'type'
        sample_id: str, 样本ID (用于标题)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 1. 绘制点云 (只画前方0~80米, 左右±40米范围的点)
    x_range = (points[:, 0] > 0) & (points[:, 0] < 80)
    y_range = (np.abs(points[:, 1]) < 40)
    mask = x_range & y_range
    pts = points[mask]
    
    # 用反射强度着色
    scatter = ax.scatter(pts[:, 0], pts[:, 1],
                         c=pts[:, 3], cmap='viridis',
                         s=0.3, alpha=0.6)
    plt.colorbar(scatter, ax=ax, label='Intensity', shrink=0.6)
    
    # 2. 绘制3D框 (只画底面4个角点构成的多边形)
    colors = {'Car': 'red', 'Pedestrian': 'lime', 'Cyclist': 'cyan'}
    for box in boxes_lidar:
        if box['type'] not in colors:
            continue
        corners = box['corners']
        # 底面4个角点: 索引0,1,2,3
        bottom = corners[:4, :2]  # 只取x,y
        # 闭合多边形
        poly = plt.Polygon(bottom, fill=False,
                           edgecolor=colors[box['type']],
                           linewidth=2)
        ax.add_patch(poly)
        
        # 画朝向箭头 (角点0→角点1)
        ax.annotate('', xy=(corners[1, 0], corners[1, 1]),
                    xytext=(corners[0, 0], corners[0, 1]),
                    arrowprops=dict(arrowstyle='->',
                    color=colors[box['type']], lw=1.5))
    
    # 3. 设置坐标轴
    ax.set_xlim(0, 80)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.set_xlabel('X (前) →', fontsize=12)
    ax.set_ylabel('Y (左) →', fontsize=12)
    ax.set_title(f'BEV 鸟瞰图 — Sample {sample_id}', fontsize=14)
    
    # 标注图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Car'),
        Line2D([0], [0], color='lime', lw=2, label='Pedestrian'),
        Line2D([0], [0], color='cyan', lw=2, label='Cyclist'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('bev.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # 构建路径
    img_path = os.path.join(KITTI_ROOT, SPLIT, 'image_2', SAMPLE_ID + '.png')
    bin_path = os.path.join(KITTI_ROOT, SPLIT, 'velodyne', SAMPLE_ID + '.bin')
    label_path = os.path.join(KITTI_ROOT, SPLIT, 'label_2', SAMPLE_ID + '.txt')
    calib_path = os.path.join(KITTI_ROOT, SPLIT, 'calib', SAMPLE_ID + '.txt')
    
    # Step 1: 加载点云
    points = load_velodyne_bin(bin_path)
    print(f"[Step1] 点云: {points.shape}")
    
    # Step 2: 解析标注
    objects = load_kitti_labels(label_path)
    print(f"[Step2] 标注目标: {len(objects)}")
    
    # Step 3: 读取标定
    P2, R0, V2C = load_calib(calib_path)
    print(f"[Step3] P2{P2.shape} R0{R0.shape} V2C{V2C.shape}")
    
    # Step 4+5: 标注框转到LiDAR坐标系 + 计算8角点
    boxes_lidar = []
    for obj in objects:
        if obj['type'] not in ['Car', 'Pedestrian', 'Cyclist']: continue
        center, dims, heading = box_cam_to_lidar(obj, V2C, R0)
        corners = get_8_corners(center, dims, heading)
        boxes_lidar.append({'type': obj['type'], 'corners': corners})
    print(f"[Step4+5] 转换 {len(boxes_lidar)} 个框到LiDAR坐标系")
    
    # Step 7: BEV可视化
    visualize_bev(points, boxes_lidar, SAMPLE_ID)
    print("[Step7] BEV图已保存到 bev.png")
    
    # Step 8: 3D框投影到图像
    # image = cv2.imread(img_path)
    # project_3d_box_to_image(image, boxes_lidar, points, P2, R0, V2C, SAMPLE_ID)
    # print("[Step8] 投影图已保存到 projection.png")
    
    # print("\n✓ 全流程完成！检查 bev.png 和 projection.png")