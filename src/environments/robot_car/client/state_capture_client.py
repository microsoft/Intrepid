import argparse
import asyncio
import os

from environments.robot_car.client.client_base import CarClient
from environments.robot_car.client.client_utils import get_timestamp_str, init_cameras
from environments.robot_car.client.state import CarState

async def do_capture(host, port, output_dir, cameras):
    car = CarClient(host, port)
    await car.connect()

    car_state = CarState()
    await car_state.capture_from_cameras(car, cameras)
    car_state.save_to_files(output_dir)
    print(f"Saved images to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=21219)
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.getcwd(), get_timestamp_str()))
    parser.add_argument('--cameras', type=str, nargs='+', default=[])
    args = parser.parse_args()

    assert len(args.cameras) > 0, "Must specify at least one camera"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    cameras = init_cameras(args.cameras)
    try:
        asyncio.run(do_capture(args.host, args.port, args.output_dir, cameras))
    finally:
        for c in cameras:
            c.release()
