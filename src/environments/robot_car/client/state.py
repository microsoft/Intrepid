import cv2 as cv
import glob
import numpy as np
import os
from PIL import Image

class CarState:
    # All pictures stored as numpy arrays in PyTorch pixel order
    # Picture from the car's camera
    car_pic = None
    # List of pictures from external cameras
    external_pics = None


    def __init__(self, image_size=256):
        self.image_size = (image_size, image_size)
        pass

    async def capture_from_cameras(self, car_client, external_cameras):
        # Take pictures using external cameras
        pics = []
        for i, cam in enumerate(external_cameras):
            ret, pic = cam.read()
            if not ret:
                print(f"Error: Could not read from camera {i}")
            else:
                pics.append(cv.resize(pic, self.image_size, cv.INTER_CUBIC))

        # Get car picture and convert to OpenCV array
        car_jpg = await car_client.download_pic()
        car_array = np.frombuffer(car_jpg, dtype=np.uint8)
        car_pic = cv.imdecode(car_array, cv.IMREAD_COLOR)
        car_pic_resized = cv.resize(car_pic, self.image_size, cv.INTER_CUBIC)

        # Store the pictures
        self.car_pic = self._cv_to_torch(car_pic_resized)
        self.external_pics = [self._cv_to_torch(pic) for pic in pics]

    def save_to_files(self, output_dir, action_id = None):
        assert self.car_pic is not None and self.external_pics is not None, "CarState is not initialized"

        filenames = {
            "action_id": action_id
        }

        # Save the car photo
        os.makedirs(output_dir, exist_ok=True)
        filename = f"car_{action_id}.jpg" if action_id else "car.jpg"
        car_pic_cv = self._torch_to_cv(self.car_pic)
        cv.imwrite(os.path.join(output_dir, filename), car_pic_cv)
        filenames["cam_car"] = filename

        # Save the external camera photos
        for i, pic in enumerate(self.external_pics):
            filename = f"cam{i}_{action_id}.jpg" if action_id else f"cam{i}.jpg"
            pic_cv = self._torch_to_cv(pic)
            cv.imwrite(os.path.join(output_dir, filename), pic_cv)
            filenames[f"cam{i}"] = filename

        return filenames

    def load_from_files(self, input_dir, camera_count, car_filename = None, external_filenames = None):
        # Load the car photo
        if car_filename is None:
            filenames = glob.glob(os.path.join(input_dir, "*car*.jpg"))
            assert len(filenames) == 1, f"Could not find unique file matching '*car*.jpg' in directory {input_dir}"
            car_pic = cv.imread(filenames[0])
        else:
            car_pic = cv.imread(os.path.join(input_dir, car_filename))
        self.car_pic = self._cv_to_torch(cv.resize(car_pic, self.image_size, cv.INTER_CUBIC))

        # Load the external camera photos
        external_pics = []
        assert external_filenames is None or len(external_filenames) == camera_count, "Number of external camera filenames does not match camera count"
        for i in range(camera_count):
            if external_filenames is None:
                filenames = glob.glob(os.path.join(input_dir, f"cam{i}*.jpg"))
                assert len(filenames) == 1, f"Could not find unique file matching 'cam{i}*.jpg' in directory {input_dir}"
                pic = cv.imread(filenames[0])
            else:
                pic = cv.imread(os.path.join(input_dir, external_filenames[i]))
            external_pics.append(pic)
        self.external_pics = [self._cv_to_torch(cv.resize(pic, self.image_size, cv.INTER_CUBIC)) for pic in external_pics]

    def load_from_files_pil(self, input_dir, camera_count):
        # Load the car photo
        filenames = glob.glob(os.path.join(input_dir, "*car*.jpg"))
        assert len(filenames) == 1, f"Could not find unique file matching '*car*.jpg' in directory {input_dir}"
        car_pic = Image.open(filenames[0])
        self.car_pic = np.asarray(car_pic.resize(self.image_size), dtype=np.uint8).transpose(2, 0, 1)

        # Load the external camera photos
        self.external_pics = []
        for i in range(camera_count):
            filenames = glob.glob(os.path.join(input_dir, f"cam{i}*.jpg"))
            assert len(filenames) == 1, f"Could not find unique file matching 'cam{i}*.jpg' in directory {input_dir}"
            pic = Image.open(filenames[0])
            self.external_pics.append(np.asarray(pic.resize(self.image_size), dtype=np.uint8).transpose(2, 0, 1))

    def concat(self):
        # Generate a single numpy array suitable for passing into PyTorch
        # The car picture comes last in the output array
        assert self.car_pic is not None and self.external_pics is not None, "CarState is not initialized"
        return np.concatenate((*self.external_pics, self.car_pic), axis=2).astype(np.float32)

    def stack(self):
        # Generate a single numpy array suitable for passing into PyTorch
        # The car picture comes last in the output array
        assert self.car_pic is not None and self.external_pics is not None, "CarState is not initialized"
        return np.stack((*self.external_pics, self.car_pic), axis=0).astype(np.float32)

    def _cv_to_torch(self, pic):
        # Convert OpenCV BGR to RGB
        pic = cv.cvtColor(pic, cv.COLOR_BGR2RGB)
        # OpenCV stores images in shape (height, width, channels)
        # Convert to shape (channels, height, width)
        return np.transpose(pic, (2, 0, 1))

    def _torch_to_cv(self, pic):
        # Convert from shape (channels, height, width) to (height, width, channels)
        pic = np.transpose(pic, (1, 2, 0))
        # Convert RGB to BGR
        return cv.cvtColor(pic, cv.COLOR_RGB2BGR)
