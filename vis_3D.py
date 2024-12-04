import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("/tmp2/r13922043/dlcv-fall-2024-hw4-weihsinyeh/gaussian-splatting/output/f27ffa1f-1/point_cloud/iteration_30000/point_cloud.ply") # 0.025
pcd = o3d.io.read_point_cloud("/tmp2/r13922043/dlcv-fall-2024-hw4-weihsinyeh/gaussian-splatting/output/92057955-b/point_cloud/iteration_30000/point_cloud.ply") # random
pcd = o3d.io.read_point_cloud("./bestmodel/point_cloud/iteration_60000/point_cloud.ply") # best model

num_points = len(pcd.points)
print(f"There are {num_points} points")
w = 400
h = 400
render = o3d.visualization.rendering.OffscreenRenderer(w, h)

render.scene.add_geometry("point_cloud", pcd, o3d.visualization.rendering.MaterialRecord())

render.scene.set_background([1, 1, 1, 1]) 

bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = bounds.get_extent()
render.setup_camera(60.0, center, center + np.array([0, 0, extent[2]]), [0, -1, 0])

img = render.render_to_image()
o3d.io.write_image("screenshot.png", img)