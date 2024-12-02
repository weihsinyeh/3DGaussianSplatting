import open3d as o3d
import open3d as o3d
import numpy as np

# 讀取原始點雲
pcd = o3d.io.read_point_cloud("/project/g/r13922043/hw4_dataset/dataset/train/sparse/0/points3D.ply")

# 獲取原始點的坐標和邊界範圍
points = np.asarray(pcd.points)
min_bound = points.min(axis=0)  # 每個維度的最小值
max_bound = points.max(axis=0)  # 每個維度的最大值

# 渲染原始點雲並保存影像
w, h = 400, 400
render = o3d.visualization.rendering.OffscreenRenderer(w, h)
render.scene.add_geometry("original_point_cloud", pcd, o3d.visualization.rendering.MaterialRecord())
render.scene.set_background([1, 1, 1, 1]) 

# 設置相機
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = bounds.get_extent()
render.setup_camera(60.0, center, center + np.array([0, 0, extent[2]]), [0, -1, 0])

# 渲染並保存原始點雲影像
original_img = render.render_to_image()
o3d.io.write_image("original_screenshot.png", original_img)

# 在邊界範圍內隨機生成點的座標
random_points = np.random.uniform(low=min_bound, high=max_bound, size=points.shape)

# 獲取原始顏色（如果存在），否則生成隨機顏色
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    random_colors = np.random.uniform(low=0, high=1, size=colors.shape)  # RGB 隨機值
else:
    # 如果原始點雲沒有顏色，直接生成新的隨機顏色
    random_colors = np.random.uniform(low=0, high=1, size=(len(random_points), 3))

# 更新點雲的隨機坐標與顏色
pcd.points = o3d.utility.Vector3dVector(random_points)
pcd.colors = o3d.utility.Vector3dVector(random_colors)

# 儲存隨機化後的點雲
output_path = "randompoint3D.ply"
o3d.io.write_point_cloud(output_path, pcd)
print(f"Randomized point cloud saved to {output_path}")

# 可視化隨機化的點雲範圍
print(f"Randomized Points Bounds: {random_points.min(axis=0)} to {random_points.max(axis=0)}")

# 設置渲染參數
w, h = 400, 400
render = o3d.visualization.rendering.OffscreenRenderer(w, h)
render.scene.add_geometry("randomized_point_cloud", pcd, o3d.visualization.rendering.MaterialRecord())
render.scene.set_background([1, 1, 1, 1]) 

# 設置相機
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = bounds.get_extent()
render.setup_camera(60.0, center, center + np.array([0, 0, extent[2]]), [0, -1, 0])

# 渲染隨機化後的點雲並保存影像
img = render.render_to_image()
o3d.io.write_image("randomized_screenshot_with_colors.png", img)
