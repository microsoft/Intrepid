import argparse
import asyncio
import cv2 as cv
import json
import logging
import os
import random

from environments.robot_car.client.client_base import CarClient
from environments.robot_car.client.client_utils import get_timestamp_str, init_cameras

CAMERA_RESOLUTION = (640, 480)

# 20 degrees = straight forward
ANGLES = [-10, 0, 10, 20, 30, 40, 50]
DIRECTIONS = ["forward", "reverse"]
SPEEDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TIMES = [0.1, 0.2, 0.3, 0.4, 0.5]


async def do_random_actions(host, port, output_dir, cameras):
    # Create output directory and log file
    os.makedirs(output_dir, exist_ok=True)
    time_str = get_timestamp_str()
    log_filename = os.path.join(output_dir, "client_log_%s.txt" % time_str)
    print(f"Writing client log to {log_filename}")
    logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_filename, filemode="w")

    # Connect to the car server
    car = CarClient(host, port)
    await car.connect()

    while True:
        action_id = get_timestamp_str()
        rand_angle = random.choice(ANGLES)
        rand_dir = random.choice(DIRECTIONS)
        rand_speed = random.choice(SPEEDS)
        rand_time = random.choice(TIMES)
        delay_time = max(TIMES) - rand_time

        # Move the car
        await car.start_action(action_id)
        await car.steer_angle(rand_angle)
        if rand_dir == "forward":
            await car.forward(rand_speed, rand_time)
        else:
            await car.reverse(rand_speed, rand_time)
        if delay_time > 0:
            await car.sleep(delay_time)
        await car.take_pic()
        await car.end_action()

        # Take pictures using external cameras
        pics = []
        pic_filenames = []
        for i, cam in enumerate(cameras):
            ret, pic = cam.read()
            if not ret:
                print(f"Error: Could not read from camera {i}")
                continue
            print(f"Got image of size {pic.shape} from camera {i}")
            pics.append(pic)
        for i, pic in enumerate(pics):
            pic_scaled = cv.resize(pic, CAMERA_RESOLUTION)
            filename = f"cam{i}_{action_id}.jpg"
            cv.imwrite(os.path.join(output_dir, filename), pic_scaled)
            pic_filenames.append(filename)

        # Log the action
        log_output = {
            "action_id": action_id,
            "angle": rand_angle,
            "direction": rand_dir,
            "speed": rand_speed,
            "time": rand_time,
        }
        for i, filename in enumerate(pic_filenames):
            log_output[f"cam{i}"] = filename
        logging.info(json.dumps(log_output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21219)
    parser.add_argument("--cameras", type=str, nargs="+", default=[])
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "pics"))
    args = parser.parse_args()

    cameras = init_cameras(args.cameras)
    try:
        asyncio.run(do_random_actions(args.host, args.port, args.output_dir, cameras))
    finally:
        for c in cameras:
            c.release()
