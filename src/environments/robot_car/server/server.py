import argparse
import asyncio
import base64
import logging
import os
import random
import time
from io import BytesIO
from time import strftime, localtime

# Use real libraries for Raspberry Pi
# from picamera import PiCamera
# from picarx import Picarx

# Use mock libraries when testing on a regular computer
from mock_pi_libraries import Picarx, PiCamera

CAMERA_RESOLUTION = (256, 256)

# The CarServer runs on the car itself and receives commands from the client
class CarServer:
    def __init__(self, picture_out_dir="."):
        # Init car and keep track of state
        self._car = Picarx()
        self._angle = 0
        self._factor = 100.0
        self._delta_t = 0.005
        self._speed = 0
        self._dir_is_fwd = True
        self._action_id = "0"

        # Init camera and photo output directory
        self._output_dir = picture_out_dir
        self._camera = PiCamera()
        self._camera.start_preview()

    def _parse_float_cmd(self, cmd_str):
        # Parse a command string of the form "cmd_name, float1, float2, ..."
        cmd_parts = cmd_str.split(",")
        cmd_args = [float(x.strip()) for x in cmd_parts[1:]]
        return (cmd_parts[0], *cmd_args)

    def _gradual_change(self, fn, start_val, stop_val):
        incr = 1
        if start_val > stop_val:
            incr = -1
        print(f"gradual {start_val}, {stop_val}, {incr}")
        for angle in range(int(start_val), int(stop_val), incr):
            fn(angle)
            time.sleep(self._delta_t)

    def forward(self, speed):
        if not self._dir_is_fwd:
            self._gradual_change(self._car.backward, self._speed * self._factor, 0)
            self._gradual_change(self._car.forward, 0, speed * self._factor)
        else:
            self._gradual_change(self._car.forward, self._speed * self._factor, speed * self._factor)
        self._speed = speed
        self._dir_is_fwd = True

    def backward(self, speed):
        if self._dir_is_fwd:
            self._gradual_change(self._car.forward, self._speed * self._factor, 0)
            self._gradual_change(self._car.backward, 0, speed * self._factor)
        else:
            self._gradual_change(self._car.backward, self._speed * self._factor, speed * self._factor)
        self._speed = speed
        self._dir_is_fwd = False

    def stop(self):
        if self._dir_is_fwd:
            print("stop forward")
            self._gradual_change(self._car.forward, self._speed * self._factor, 0)
        else:
            print("stop reverse")
            self._gradual_change(self._car.backward, self._speed * self._factor, 0)
        self._car.stop()
        self._speed = 0

    def take_pic_to_file(self):
        time_str = strftime("%Y-%m-%d-%H-%M-%S", localtime(time.time()))
        filename = "car_%s.jpg" % time_str
        output_file = os.path.join(self._output_dir, filename)
        self._camera.capture(output_file, resize=CAMERA_RESOLUTION)
        print("saved %s" % output_file)
        return filename

    def take_pic_to_buffer(self):
        buffer = BytesIO()
        self._camera.capture(buffer, format="jpeg", resize=CAMERA_RESOLUTION)
        data = buffer.getvalue()
        print("captured jpeg image of length %d to buffer" % len(data))
        return data

    async def event_loop(self):
        while True:
            cmd_str = await self._cmd_queue.get()
            print("Got command: %s" % cmd_str)

            if not cmd_str:
                print("Command is empty, skipping")

            elif cmd_str.startswith("setid"):
                cmd_parts = cmd_str.split(",")
                if len(cmd_parts) < 2:
                    print("Missing argument for command, skipping")
                else:
                    self._action_id = cmd_parts[1].strip()
                    logging.info(cmd_str)

            elif cmd_str.startswith("forward"):
                cmd = self._parse_float_cmd(cmd_str)
                if not cmd:
                    print("Missing argument for command, skipping")
                else:
                    speed = cmd[1]
                    self.forward(speed)
                    logging.info(cmd_str)
                    if len(cmd) > 2:
                        time = cmd[2]
                        await asyncio.sleep(time)
                        self.stop()

            elif cmd_str.startswith("reverse"):
                cmd = self._parse_float_cmd(cmd_str)
                if not cmd:
                    print("Missing argument for command, skipping")
                else:
                    speed = cmd[1]
                    self.backward(speed)
                    logging.info(cmd_str)
                    if len(cmd) > 2:
                        time = cmd[2]
                        await asyncio.sleep(time)
                        self.stop()

            elif cmd_str.startswith("angle"):
                cmd = self._parse_float_cmd(cmd_str)
                if not cmd:
                    print("Missing argument for command, skipping")
                else:
                    new_angle = cmd[1]
                    self._gradual_change(self._car.set_dir_servo_angle, self._angle, new_angle)
                    self._angle = new_angle
                    logging.info(cmd_str)

            elif cmd_str.startswith("sleep"):
                cmd = self._parse_float_cmd(cmd_str)
                if not cmd:
                    print("Missing argument for command, skipping")
                else:
                    sleep_time = cmd[1]
                    await asyncio.sleep(sleep_time)
                    logging.info(cmd_str)

            elif cmd_str == "stop":
                self.stop()
                logging.info(cmd_str)

            elif cmd_str == "take_pic":
                await asyncio.sleep(0.1)
                filename = self.take_pic_to_file()
                logging.info("%s,%s" % (cmd_str, filename))

            else:
                print("Error: Unknown command '%s'" % cmd_str)

            self._cmd_queue.task_done()

    # https://stackoverflow.com/questions/48506460/python-simple-socket-client-server-using-asyncio
    async def start_server(self, port):
        print("Starting server on port %d" % port)
        print("Waiting for client")

        # Set buffer limit to 1MB because we might be sending jpeg images
        server = await asyncio.start_server(self.handle_client, host=None, port=port, limit=2**20)
        async with server:
            await server.serve_forever()

    async def handle_client(self, reader, writer):
        print("Client connected")
        request = None
        while True:
            # Process incoming request
            request = (await reader.readline()).decode("utf8")
            request = request.strip().lower()

            # Commands to close the connection
            if request == "quit":
                print('Received "quit" command. Closing connection.')
                break
            if not request:
                print("Received empty command. Closing connection.")
                break

            # Otherwise, execute the command
            if request == "wait":
                # Wait until all queued commands finish
                print('Received "wait" command. Waiting until all queued commands finish.')
                await self._cmd_queue.join()
                response = "Finished all commands for action %s\n" % self._action_id
                writer.write(response.encode())
                await writer.drain()

            elif request == "send_pic":
                # Take a picture and immediately send it back to client
                print('Received "send_pic" command. Waiting until all queued commands finish.')
                await self._cmd_queue.join()
                print("Taking picture and sending it back to client.")
                pic_bytes = self.take_pic_to_buffer()
                pic_base64 = base64.b64encode(pic_bytes) + b"\n"
                writer.write(pic_base64)
                await writer.drain()

            else:
                # Otherwise, put the action to take into the shared command queue
                await self._cmd_queue.put(request)

        print("Client diconnected")
        writer.close()

    async def do_random_actions(self):
        while True:
            if self._cmd_queue.empty():
                rand_angle = random.choice([-30, -20, -10, 0, 10, 20, 30])
                rand_dir = random.choice(["forward", "reverse"])
                rand_speed = random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                rand_time = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
                await self._cmd_queue.put("angle,%f" % rand_angle)
                await self._cmd_queue.put("%s,%f,%f" % (rand_dir, rand_speed, rand_time))
                await self._cmd_queue.put("sleep,%f" % (3.0 - rand_time))
                await self._cmd_queue.put("take_pic")
            else:
                await asyncio.sleep(0.1)

    async def start(self, port, run_local=False):
        self._cmd_queue = asyncio.Queue()
        if run_local:
            await asyncio.gather(self.do_random_actions(), self.event_loop())
        else:
            await asyncio.gather(self.start_server(port), self.event_loop())


if __name__ == "__main__":
    # Create output directory and log file
    output_dir = os.path.join(os.getcwd(), "pics")
    os.makedirs(output_dir, exist_ok=True)
    time_str = strftime("%Y-%m-%d-%H-%M-%S", localtime(time.time()))
    log_filename = os.path.join(output_dir, "server_log_%s.txt" % time_str)
    logging.basicConfig(level=logging.INFO, format="%(message)s", filename=log_filename, filemode="w")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_local", action="store_true", help="Run random commands locally instead of running server")
    parser.add_argument("--port", type=int, default=21219, help="Port to listen on, ignored if --run_local is used")
    args = parser.parse_args()

    c_srv = CarServer(output_dir)
    asyncio.run(c_srv.start(args.port, args.run_local))
