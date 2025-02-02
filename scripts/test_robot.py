import argparse
from src.testing.robot_tester import RobotTester

def test_robot(model_path: str):
    """Runs the trained model in a MuJoCo simulation."""
    tester = RobotTester(model_path)
    tester.run_test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained DAML model in MuJoCo.")
    parser.add_argument("--model", type=str, default="models/franka_model.pth", help="Path to the trained model.")

    args = parser.parse_args()
    test_robot(args.model)
