# Mock versions of raspberry pi specific libraries


class Picarx:
    def forward(self, speed):
        print(f"[mock picarx] forward(speed={speed})")

    def backward(self, speed):
        print(f"[mock picarx] backward(speed={speed})")

    def set_dir_servo_angle(self, angle):
        print(f"[mock picarx] set_dir_servo_angle(angle={angle})")

    def stop(self):
        print("[mock picarx] stop()")


class PiCamera:
    def start_preview(self):
        print("[mock picamera] start_preview()")

    def capture(self, output_file, **kwargs):
        print(f"[mock picamera] capture(output_file={output_file})")
