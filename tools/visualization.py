import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import copy
import os
from pathlib import Path
from datetime import datetime

def get_color(deg):
    '''
    This function is used to determine the color of the arrow that shows angle error.
    It gets color based on the degree of rotation.
    '''
    deg = abs(deg)
    if deg < 5:
        return 'green'
    elif deg < 10:
        return 'orange'
    else:
        return 'red'
        
def plot_point_cloud(source, target, result = None, show_grid=False, x_diff=None, y_diff=None, z_diff=None):
    '''
    Visualizes two point clouds and optionally a result point cloud.

    args:
        source: o3d.geometry.PointCloud, the source point cloud to visualize 
        target: o3d.geometry.PointCloud, the target point cloud to visualize
        result: o3d.geometry.PointCloud, optional, the result point cloud to visualize
        show_grid: bool, optional, whether to show the grid in the 3D plot
        x_diff: float, optional, the difference in X angle to visualize
        y_diff: float, optional, the difference in Y angle to visualize
        z_diff: float, optional, the difference in Z angle to visualize

    result:
        A 3D plotly figure visualizing the source and target point clouds, and optionally
        the result point cloud, with arrows indicating angle differences.
    '''
    # Point cloud data
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    if result is not None:
        result_points = np.asarray(result.points)

    source_scatter = go.Scatter3d(
        x=source_points[:, 0], y=source_points[:, 1], z=source_points[:, 2],
        mode='markers', marker=dict(size=2, color='rgb(255, 180, 0)'), name='Source (Yellow)')

    target_scatter = go.Scatter3d(
        x=target_points[:, 0], y=target_points[:, 1], z=target_points[:, 2],
        mode='markers', marker=dict(size=2, color='rgb(0, 166, 237)'), name='Target (Blue)')

    data = [source_scatter, target_scatter]

    if result is not None:
        result_scatter = go.Scatter3d(
            x=result_points[:, 0], y=result_points[:, 1], z=result_points[:, 2],
        # mode='markers', marker=dict(size=2, color='rgb(153, 0, 76)'), name='Result (Purple)')
        mode='markers', marker=dict(size=2, color='rgb(255, 180, 0)'), name='Result (Yellow)')
        data.append(result_scatter)

    # Create the figure
    fig = go.Figure(data=data)

    # Plots arrows to show angle difference 
    if x_diff and y_diff and z_diff:
        all_points = [source_points, target_points]
        if result is not None:
            all_points.append(result_points)

        center = np.mean(np.vstack(all_points), axis=0)
        scale_factor = 0.01  # Adjust this for visual scaling

        angles = [x_diff, y_diff, z_diff]
        axes = [
            {"axis": "X", "vec": np.array([1, 0, 0]), "angle": angles[0], "label": f'ΔX (Roll): {angles[0]:.2f}°', "color": get_color(angles[0])},
            {"axis": "Y", "vec": np.array([0, 1, 0]), "angle": angles[1], "label": f'ΔY (Pitch): {angles[1]:.2f}°', "color": get_color(angles[1])},
            {"axis": "Z", "vec": np.array([0, 0, 1]), "angle": angles[2], "label": f'ΔZ (Yaw): {angles[2]:.2f}°', "color": get_color(angles[2])},
        ]

        for axis in axes:
            magnitude = abs(axis["angle"]) * scale_factor
            vec = axis["vec"] * axis["angle"] * scale_factor
            tip = center + vec
            color = axis["color"]

            # Line (arrow)
            fig.add_trace(go.Scatter3d(
                x=[center[0], tip[0]], y=[center[1], tip[1]], z=[center[2], tip[2]],
                mode='lines',
                line=dict(color=color, width=5),
                showlegend=False
            ))

            # Arrow Head (cone)
            fig.add_trace(go.Cone(
                x=[tip[0]], y=[tip[1]], z=[tip[2]],
                u=[vec[0]], v=[vec[1]], w=[vec[2]],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                sizemode="absolute",
                sizeref=0.04 * magnitude, 
                anchor="tip"
            ))

            # Annotation
            text_pos = tip + 0.02 * np.sign(vec)
            fig.add_trace(go.Scatter3d(
                x=[text_pos[0]], y=[text_pos[1]], z=[text_pos[2]],
                mode='text',
                text=[axis["label"]],
                textposition="top center",
                showlegend=False,
                textfont=dict(size=14, color=color)
            ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=show_grid),
            yaxis=dict(visible=show_grid),
            zaxis=dict(visible=show_grid),
            aspectmode='data'
        ),
        width=900,  
        height=700  
    )

    fig.show()

# Open3d helper function to draw 2 point clouds
def draw_point_clouds(pcd1, pcd2):
    '''
    args:
        pcd1: o3d.geometry.PointCloud
        pcd2: o3d.geometry.PointCloud
    
    result: 
        Visualizes pcd2 with yellow and pcd1 with cyan ransformed with an alignment transformation. 
    '''
    pcd1_temp = copy.deepcopy(pcd1)
    pcd2_temp = copy.deepcopy(pcd2)
    pcd1_temp.paint_uniform_color([1, 0.706, 0])
    pcd2_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd1_temp, pcd2_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    
# open3d helper function to draw registration result
def draw_registration_result(source, target, transformation, method_name = None):
    '''
    args:
        source: o3d.geometry.PointCloud
        target: o3d.geometry.PointCloud
        transformation: np.array, 4x4 matrix (an intial guess of the transformation to roughly align PCs on top of each other) --> Global registration
    
    result: 
        Saves source, target, and transformed source point clouds as .ply files (for later viewing).
        If a display is available, also visualizes the registration result.
    '''
    # Make copies so we never mutate the inputs
    source_orig = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp = copy.deepcopy(source)

    # Color for easier later inspection
    # source_orig.paint_uniform_color([1, 0.706, 0])
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])

    # Apply transformation to the "result" copy
    source_temp.transform(transformation)

    # Save point clouds for offline visualization
    out_dir = Path(__file__).resolve().parents[1] / "results" / "registration_plys"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%d_%H%M_%f")
    source_path = out_dir / f"{stamp}_source.ply"
    target_path = out_dir / f"{stamp}_target.ply"
    transformed_path = out_dir / f"{stamp}_source_transformed.ply"

    try:
        if method_name is not None:
            source_path = out_dir / f"{stamp}_{method_name}_source.ply"
            target_path = out_dir / f"{stamp}_{method_name}_target.ply"
            transformed_path = out_dir / f"{stamp}_{method_name}_source_transformed.ply"

        o3d.io.write_point_cloud(str(source_path), source_orig)
        o3d.io.write_point_cloud(str(target_path), target_temp)
        o3d.io.write_point_cloud(str(transformed_path), source_temp)
        print(f"[visualization] Saved .ply files to: {out_dir}")
    except Exception as e:
        print(f"[visualization] WARNING: Failed to write .ply files to {out_dir}: {e}")

    # Only try to open a window if a display exists (avoid headless GLFW warnings)
    if os.environ.get("DISPLAY"):
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                          zoom=0.4459,
                                          front=[0.9288, -0.2951, -0.2242],
                                          lookat=[1.6784, 2.0612, 1.4451],
                                          up=[-0.3402, -0.9189, -0.1996])