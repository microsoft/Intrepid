import argparse
import asyncio
import base64
import sys
import time

from environments.robot_car.client.client_utils import get_timestamp_str

# The CarClient class sends commands and receives data from the car server
class CarClient:
    def __init__(self, host, port):
        self._host = host
        self._port: int = port
        self._action_id: str = ""

    async def connect(self):
        # Buffer limit must be big enough to receive images
        reader, writer = await asyncio.open_connection(self._host, self._port, limit=2**24)
        self._reader = reader
        self._writer = writer

    async def disconnect(self):
        self._writer.close()
        await self._writer.wait_closed()

    async def _send_to_server(self, cmd_str: str):
        print(f"Sending command to server: {cmd_str.strip()}")
        self._writer.write(cmd_str.encode())
        await self._writer.drain()
    
    async def _recv_from_server(self):
        resp = await self._reader.readline()
        line = resp.decode().strip()
        print(f"Received response from server: {line}")
        return line

    async def _recv_and_decode_base64_from_server(self):
        resp = await self._reader.readline()
        data = base64.b64decode(resp.strip())
        print(f"Received binary response from server of length {len(data)}")
        return data

    async def steer_angle(self, angle: float):
        await self._send_to_server(f'angle,{angle}\n')

    async def forward(self, speed: float, time=None):
        if time is None:
            await self._send_to_server(f'forward,{speed}\n')
        else:
            await self._send_to_server(f'forward,{speed},{time}\n')

    async def reverse(self, speed: float, time=None):
        if time is None:
            await self._send_to_server(f'reverse,{speed}\n')
        else:
            await self._send_to_server(f'reverse,{speed},{time}\n')

    async def sleep(self, time: float):
        await self._send_to_server(f'sleep,{time}\n')

    async def stop(self):
        await self._send_to_server('stop\n')

    async def take_pic(self):
        await self._send_to_server('take_pic\n')
    
    async def download_pic(self):
        await self._send_to_server('send_pic\n')
        data = await self._recv_and_decode_base64_from_server()
        return data

    async def start_action(self, action_id: str):
        if not action_id or ',' in action_id:
            print(f"Error: action id cannot be empty or contain ',' character: {action_id}")
            return
        self._action_id = action_id
        print(f"Starting action with id {self._action_id}")
        await self._send_to_server(f'setid,{self._action_id}\n')

    async def end_action(self):
        await self._send_to_server('wait\n')
        msg = await self._recv_from_server()
        if not msg:
            print(f"Error: No response from server for action id {self._action_id}")
            sys.exit(1)
        elif str(self._action_id) not in msg:
            print(f"Error: Current action id {self._action_id} not found in server response")
            sys.exit(1)
        else:
            print()

# Do some random actions for debugging
async def car_client_test(host, port):
    print(f"Connecting to car server at {host}:{port}")
    car = CarClient(host, port)
    await car.connect()
    speed = 1.0
    t = 2.0
    angle = 35

    print("start_action")
    await car.start_action(get_timestamp_str())
    time.sleep(t)
    print("end_action")
    await car.end_action()
    time.sleep(t)

    print("start_action")
    await car.start_action(get_timestamp_str())
    print("take_pic")
    await car.take_pic()
    print("forward")
    await car.forward(speed)
    time.sleep(t)
    print("reverse")
    await car.reverse(speed)
    time.sleep(t)
    print("forward")
    await car.forward(speed)
    time.sleep(t)
    print(f"left {angle}")
    await car.steer_angle(-1*angle)
    time.sleep(t)
    print(f"right {angle}")
    await car.steer_angle(angle)
    time.sleep(t)
    print("straight")
    await car.steer_angle(0)
    time.sleep(t)
    print("take_pic")
    await car.take_pic()
    time.sleep(t)
    print("stop")
    await car.stop()
    time.sleep(t)
    print("end_action")
    await car.end_action()

    print("start_action")
    await car.start_action(get_timestamp_str())
    print("forward for time (should stop automatically)")
    await car.forward(speed, 5.0)
    print('take_pic')
    await car.take_pic()
    print("end_action")
    await car.end_action()
    await car.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=21219)
    args = parser.parse_args()

    print("Running car client test...")
    asyncio.run(car_client_test(args.host, args.port))
    