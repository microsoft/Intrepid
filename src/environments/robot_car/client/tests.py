import glob
import numpy as np
import os
from PIL import Image

from environments.robot_car.client.alex_inference import AlexModelInference
from environments.robot_car.client.state import CarState

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# Tests to compare PIL and OpenCV image loading
def load_images_with_pil(input_dir, camera_count):
    external_pics = []
    for i in range(camera_count):
        cam_file = glob.glob(os.path.join(input_dir, f"cam{i}*.jpg"))
        assert len(cam_file) == 1, f"Could not find unique file matching 'cam{i}*.jpg' in directory {input_dir}"
        with Image.open(cam_file[0]).resize((250, 250)) as img:
            cam_pic = np.asarray(img).transpose(2, 0, 1).astype("uint8")
            external_pics.append(cam_pic)

    car_file = glob.glob(os.path.join(input_dir, "car*.jpg"))
    assert len(car_file) == 1, f"Could not find unique file matching 'car*.jpg' in directory {input_dir}"
    with Image.open(car_file[0]).resize((250, 250)) as img:
        car_pic = np.asarray(img).transpose(2, 0, 1).astype("uint8")

    return (car_pic, external_pics)


def test_image_load(input_dir, camera_count=2):
    # load with CarState object
    cs = CarState()
    cs.load_from_files(input_dir, camera_count)
    cs_array = cs.concat()

    # load with PIL
    pil_state = CarState()
    pil_state.car_pic, pil_state.external_pics = load_images_with_pil(input_dir, camera_count)
    pil_array = np.concatenate(pil_state.external_pics + [pil_state.car_pic], axis=2)

    # save loaded results
    pil_state.save_to_files(os.path.join("test_data", "pil"))
    cs.save_to_files(os.path.join("test_data", "cv"))

    # compare results
    assert (
        cs_array.shape == pil_array.shape
    ), f"Shape of CarState result {cs_array.shape} does not match PIL result {pil_array.shape}"
    assert (
        cs_array.max() == pil_array.max()
    ), f"Max of CarState result {cs_array.max()} does not match PIL result {pil_array.max()}"
    assert (
        cs_array.min() == pil_array.min()
    ), f"Min of CarState result {cs_array.min()} does not match PIL result {pil_array.min()}"
    # difference = np.abs(cs_array - pil_array)
    # assert np.allclose(cs_array, pil_array, atol=16), "CarState and PIL results do not match"

    error = CarState()
    error.car_pic = np.abs(cs.car_pic - pil_state.car_pic)
    error.external_pics = [
        np.abs(cs.external_pics[0] - pil_state.external_pics[0]),
        np.abs(cs.external_pics[1] - pil_state.external_pics[1]),
    ]
    error.save_to_files(os.path.join("test_data", "error"))


def test_pil_vs_cv(start_state_dir, goal_state_dir, k, desired_action, camera_count=2):
    # load from files
    start_state_cv = CarState()
    start_state_cv.load_from_files(start_state_dir, camera_count)
    goal_state_cv = CarState()
    goal_state_cv.load_from_files(goal_state_dir, camera_count)
    start_state_pil = CarState()
    start_state_pil.car_pic, start_state_pil.external_pics = load_images_with_pil(start_state_dir, camera_count)
    goal_state_pil = CarState()
    goal_state_pil.car_pic, goal_state_pil.external_pics = load_images_with_pil(goal_state_dir, camera_count)

    # run inference
    model_cv = AlexModelInference(goal_state_cv, MODEL_PATH)
    predicted_action_cv = model_cv.get_next_action(start_state_cv, k)
    model_pil = AlexModelInference(goal_state_pil, MODEL_PATH)
    predicted_action_pil = model_pil.get_next_action(start_state_pil, k)

    print(f"Desired action: {desired_action}")
    print(f"Predicted action with OpenCV: {predicted_action_cv}")
    print(f"Predicted action with PIL: {predicted_action_pil}")
    print(
        f"Distance: {np.linalg.norm(model_cv._normalize_action(predicted_action_cv) - model_pil._normalize_action(predicted_action_pil))}"
    )
    print()


# Tests for model prediction
def normalized_error(model, predicted, desired):
    return np.linalg.norm(model._normalize_action(predicted) - model._normalize_action(desired))


def print_action_comparison(model, predicted_action, desired_action, normalized=True):
    predicted_action_norm = model._normalize_action(predicted_action)
    if normalized:
        print(f"Predicted action: {predicted_action_norm}")
        print(f"Desired action: {desired_action}")
        print(f"Distance: {np.linalg.norm(predicted_action_norm - desired_action)}")
    else:
        print(f"Predicted action: {predicted_action}")
        print(f"Desired action: {desired_action}")
        print(f"Distance: {normalized_error(model, predicted_action, desired_action)}")
    print()


def test_alex_model(
    start_state_dir, goal_state_dir, k, desired_action, camera_count=2, normalized=False, reconstruct_output=None
):
    start_state = CarState()
    start_state.load_from_files_pil(start_state_dir, camera_count)
    goal_state = CarState()
    goal_state.load_from_files_pil(goal_state_dir, camera_count)

    model = AlexModelInference(goal_state, MODEL_PATH)
    predicted_action = model.get_next_action(start_state, k, reconstruct_output=reconstruct_output)
    print_action_comparison(model, predicted_action, desired_action, normalized)


if __name__ == "__main__":
    print("Testing image loading...")
    test_image_load(os.path.join("test_data", "state_1"))

    action_state_1_to_2 = {"angle": 0, "direction": "forward", "speed": 0.3, "time": 0.1}
    action_state_3_to_4 = {"angle": 0, "direction": "reverse", "speed": 0.5, "time": 0.3}
    action_state_5_to_6 = {"angle": 50, "direction": "forward", "speed": 0.3, "time": 0.2}

    # sampled from training data
    action_state_train_1_to_2 = {"angle": 20, "direction": "forward", "speed": 0.0, "time": 0.5}
    action_state_train_3_to_4 = {"angle": 20, "direction": "forward", "speed": 0.3, "time": 0.2}
    action_state_train_5_to_6 = {"angle": 30, "direction": "reverse", "speed": 0.4, "time": 0.4}

    print("Testing PIL vs OpenCV...")
    test_pil_vs_cv(os.path.join("test_data", "state_1"), os.path.join("test_data", "state_2"), 1, action_state_1_to_2)
    test_pil_vs_cv(os.path.join("test_data", "state_3"), os.path.join("test_data", "state_4"), 1, action_state_3_to_4)
    test_pil_vs_cv(os.path.join("test_data", "state_5"), os.path.join("test_data", "state_6"), 1, action_state_5_to_6)

    print("Testing Alex's inverse model...")
    test_alex_model(
        os.path.join("test_data", "state_1"),
        os.path.join("test_data", "state_2"),
        1,
        action_state_1_to_2,
        reconstruct_output="test_data/reconstruction_1_to_2.jpg",
    )
    test_alex_model(
        os.path.join("test_data", "state_3"),
        os.path.join("test_data", "state_4"),
        1,
        action_state_3_to_4,
        reconstruct_output="test_data/reconstruction_3_to_4.jpg",
    )
    test_alex_model(
        os.path.join("test_data", "state_5"),
        os.path.join("test_data", "state_6"),
        1,
        action_state_5_to_6,
        reconstruct_output="test_data/reconstruction_5_to_6.jpg",
    )
    test_alex_model(
        os.path.join("test_data", "state_train_1"),
        os.path.join("test_data", "state_train_2"),
        1,
        action_state_train_1_to_2,
        reconstruct_output="test_data/reconstruction_train_1_to_2.jpg",
    )
    test_alex_model(
        os.path.join("test_data", "state_train_3"),
        os.path.join("test_data", "state_train_4"),
        1,
        action_state_train_3_to_4,
        reconstruct_output="test_data/reconstruction_train_3_to_4.jpg",
    )
    test_alex_model(
        os.path.join("test_data", "state_train_5"),
        os.path.join("test_data", "state_train_6"),
        1,
        action_state_train_5_to_6,
        reconstruct_output="test_data/reconstruction_train_5_to_6.jpg",
    )
