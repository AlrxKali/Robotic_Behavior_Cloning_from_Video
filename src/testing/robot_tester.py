import torch
import numpy as np
import mujoco
import glfw
import time
from threading import Thread
from pathlib import Path
from daml.model import DAMLModel
from daml.config import DAMLConfig

class RobotTester:
    """Tests the trained DAML model by running generated trajectories in a MuJoCo simulation."""
    def __init__(self, model_path: str = "models/franka_model.pth"):
        # Load trained model
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.config = checkpoint["config"]
        self.model = DAMLModel(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Initialize MuJoCo simulation
        self.mj_model = mujoco.MjModel.from_xml_path("panda_mujoco/world.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.setup_visualization()

        # Control parameters
        self.running = True
        self.fps = 30

    def setup_visualization(self):
        """Initializes MuJoCo visualization settings."""
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.mj_model, maxgeom=10000)
        self.width, self.height = 640, 480

    def execute_trajectory(self, initial_state):
        """Executes the trajectory generated by the DAML model."""
        with torch.no_grad():
            state = torch.FloatTensor(initial_state).unsqueeze(0)
            trajectory = self.model(state)[0].numpy()

            for point in trajectory:
                if not self.running:
                    break

                position = point[:3]
                orientation = point[3:7]
                gripper_state = point[7] if len(point) > 7 else 0

                self.control(position, orientation)
                self.gripper(gripper_state > 0.5)
                mujoco.mj_step(self.mj_model, self.mj_data)
                time.sleep(1.0 / self.fps)

    def control(self, xpos_d, xquat_d):
        """Controls the robot arm by computing and applying joint torques."""
        xpos = self.mj_data.body("panda_hand").xpos
        xquat = self.mj_data.body("panda_hand").xquat

        jacp = np.zeros((3, self.mj_model.nv))
        jacr = np.zeros((3, self.mj_model.nv))
        bodyid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, jacr, bodyid)

        error = np.concatenate([xpos_d - xpos, -np.subtract(xquat, xquat_d)])
        J = np.concatenate((jacp, jacr))
        v = J @ self.mj_data.qvel

        for i in range(1, 8):
            dofadr = self.mj_model.joint(f"panda_joint{i}").dofadr
            self.mj_data.actuator(f"panda_joint{i}").ctrl = (
                self.mj_data.joint(f"panda_joint{i}").qfrc_bias
                + J[:, dofadr].T @ np.diag([600.0] * 3 + [30.0] * 3) @ error
                - J[:, dofadr].T @ np.diag([2 * np.sqrt(600.0)] * 3 + [2 * np.sqrt(30.0)] * 3) @ v
            )

    def gripper(self, open=True):
        """Controls the gripper to open or close."""
        ctrl = 0.04 if open else 0
        self.mj_data.actuator("pos_panda_finger_joint1").ctrl = ctrl
        self.mj_data.actuator("pos_panda_finger_joint2").ctrl = ctrl

    def render(self):
        """Runs the MuJoCo visualization loop."""
        glfw.init()
        window = glfw.create_window(self.width, self.height, "Robot Test", None, None)
        glfw.make_context_current(window)
        context = mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_100)
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        while not glfw.window_should_close(window) and self.running:
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)

            mujoco.mjv_updateScene(
                self.mj_model, self.mj_data,
                mujoco.MjvOption(), mujoco.MjvPerturb(),
                self.cam, mujoco.mjtCatBit.mjCAT_ALL,
                self.scene
            )

            mujoco.mjr_render(viewport, self.scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(1.0 / self.fps)

        self.running = False
        glfw.terminate()

    def run_test(self):
        """Runs the complete robot test, including visualization and trajectory execution."""
        render_thread = Thread(target=self.render)
        render_thread.start()

        initial_state = np.zeros(8)  # Modify as needed
        self.execute_trajectory(initial_state)

        self.running = False
        render_thread.join()

if __name__ == "__main__":
    tester = RobotTester()
    tester.run_test()
