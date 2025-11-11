from utils import read_nvm_file


def main():
    xyz_arr, image2points, image2name, rgb_arr = read_nvm_file(
        "/home/n11373598/work/glace/datasets/aachen/reconstruction.nvm",
        return_rgb=True,
    )
    name2id = {v: k for k, v in image2name.items()}

    name = "db/1139.jpg"
    image_id = name2id[name]

    points = image2points[image_id]
    xyz = xyz_arr[points]
    rgb = rgb_arr[points]

    import open3d as o3d

    pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cl, inlier_ind = pc1.remove_radius_outlier(
        nb_points=16, radius=5, print_progress=True
    )
    cl.colors = o3d.utility.Vector3dVector(rgb[inlier_ind] / 255.0)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(cl)
    render_option = vis.get_render_option()
    render_option.point_size = 10.0
    vis.run()
    vis.destroy_window()

    print()


if __name__ == "__main__":
    main()
