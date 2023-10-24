import argparse
import asyncio
import json
import logging
import os

from environments.robot_car.client.client_base import CarClient
from environments.robot_car.client.client_utils import get_timestamp_str, init_cameras
from environments.robot_car.client.state import CarState
from environments.robot_car.client.inference import LatentForwardInference

# from environments.robot_car.client.alex_inference import AlexModelInference


async def closed_loop_navigation(host, port, cameras, model, total_steps, output_dir):
    # Connect to the car server
    car = CarClient(host, port)
    await car.connect()

    for i in range(total_steps):
        print(f"********* Step {i} *********")
        # Get current state
        car_state = CarState()
        await car_state.capture_from_cameras(car, cameras)

        # Get next action from model
        action = model.get_next_action(car_state, k=1)
        print(f"Got next action from model: {action}")

        # Send action to car
        action_id = get_timestamp_str()
        await car.start_action(action_id)
        await car.steer_angle(action["angle"])
        if action["direction"] == "forward":
            await car.forward(action["speed"], action["time"])
        else:
            await car.reverse(action["speed"], action["time"])
        await car.end_action()

        # Save current state to file
        saved_img_files = car_state.save_to_files(output_dir, action_id)
        logging.info(json.dumps({"action_id": action_id, **saved_img_files, **action}))

    end_state = CarState()
    await end_state.capture_from_cameras(car, cameras)
    end_state_files = end_state.save_to_files(output_dir, "end")
    logging.info(json.dumps(end_state_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21219)
    parser.add_argument("--cameras", type=str, nargs="+", default=[])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--goal_state", type=str, default=os.path.join(os.getcwd(), "goal_state"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "output"))
    parser.add_argument("--model_dir", type=str, default=os.path.join(os.getcwd(), "models"))
    args = parser.parse_args()

    assert len(args.cameras) > 0, "Must specify at least one camera"
    assert args.steps > 0, "Number of steps must be at least 1"

    # Load pictures of goal state from files
    goal_state = CarState()
    goal_state.load_from_files(args.goal_state, len(args.cameras))

    # Load model
    # model = AlexModelInference(goal_state, args.model_dir)
    model = LatentForwardInference(goal_state, args.model_dir)

    # Create output directory and log file
    os.makedirs(args.output_dir, exist_ok=True)
    time_str = get_timestamp_str()
    log_filename = os.path.join(args.output_dir, "log_%s.txt" % time_str)
    print(f"Writing client log to {log_filename}")
    logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_filename, filemode="w")

    cameras = init_cameras(args.cameras)
    try:
        asyncio.run(closed_loop_navigation(args.host, args.port, cameras, model, args.steps, args.output_dir))
    finally:
        for c in cameras:
            c.release()
