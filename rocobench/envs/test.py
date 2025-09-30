import mujoco
import mujoco_viewer
import time

# 加载模型
model = mujoco.MjModel.from_xml_path("/Users/zhangke/Desktop/robot-collab-toolusing/rocobench/envs/task_cabinet.xml")
data = mujoco.MjData(model)

# 创建 Viewer
viewer = mujoco_viewer.MujocoViewer(model, data)

# 选择要显示的 geom（可以修改为你关心的 geom 名称）
geom_name = "mug_contact0"  # 替换成你的 geom 名称
geom_id = model.geom_name2id(geom_name)

while True:
    # 模拟一步
    mujoco.mj_step(model, data)

    # 获取 geom 坐标
    geom_pos = data.geom_xpos[geom_id]

    # 清屏打印
    print(f"\r{geom_name} world position: {geom_pos}", end="")

    # 渲染
    viewer.render()

    # 可以加点延时，看起来更稳定
    time.sleep(0.01)
