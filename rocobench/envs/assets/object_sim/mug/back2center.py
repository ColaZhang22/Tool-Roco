import trimesh
import os

# 你的 STL 文件夹路径
stl_folder = "../object_sim/mug"
# 输出文件夹（可以覆盖原文件或新建文件夹）
output_folder = "/Users/zhangke/Desktop/robot-collab-toolusing/rocobench/envs/assets/object_sim/mug_centered"
os.makedirs(output_folder, exist_ok=True)

# 循环处理 contact0~contact13
for i in range(14):
    filename = f"contact{i}.stl"
    filepath = os.path.join("/Users/zhangke/Desktop/robot-collab-toolusing/rocobench/envs/assets/object_sim/mug", filename)
    
    # 加载 mesh
    mesh = trimesh.load(filepath)
    
    # 计算几何中心
    center = mesh.centroid
    # 平移到局部原点
    mesh.apply_translation(-center)
    
    # 保存到新路径
    outpath = os.path.join(output_folder, filename)
    mesh.export(outpath)
    
    print(f"{filename} 已居中，中心移动了 {center}")

filename = f"mug.stl"
filepath = "/Users/zhangke/Desktop/robot-collab-toolusing/rocobench/envs/assets/object_sim/mug/mug.stl"
# 加载 mesh
mesh = trimesh.load(filepath)

# 计算几何中心
center = mesh.centroid
# 平移到局部原点
mesh.apply_translation(-center)

# 保存到新路径
outpath = os.path.join(output_folder, filename)
mesh.export(outpath)

print(f"{filename} 已居中，中心移动了 {center}")

print("所有 collision mesh 已经居中完成！")
