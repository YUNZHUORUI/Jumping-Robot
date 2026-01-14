from stable_baselines3 import PPO
from drone_env import DroneHoverEnv
import time


def test():
    # 1. 确保 render_mode="human"
    env = DroneHoverEnv(xml_file='appearance model/scene.xml', render_mode="human")

    # 加载模型
    model_path = "trained model/final_model.zip"
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print("找不到模型文件，请先运行 train.py")
        return

    obs, _ = env.reset()
    print("开始测试...")

    while True:
        # 预测动作
        action, _states = model.predict(obs, deterministic=True)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        # --- 必须加上这一行才能看到画面 ---
        env.render()
        # --------------------------------

        # 稍微延时
        time.sleep(env.dt)

        # 如果想循环测试，可以加上自动重置
        if terminated:
            obs, _ = env.reset()


if __name__ == "__main__":
    test()