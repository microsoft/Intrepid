import gc
import json
import numpy as np
import os
import pickle
import sys
from PIL import Image
from tqdm import tqdm


def process_data(input_dir):
    total_actions = []
    total_observations = []
    total_done = []

    # Each trajectory is in a subdirectory
    _, sub_dirs, _ = next(os.walk(input_dir))
    for sub_dir in tqdm(sub_dirs):
        print(f"Processing directory {sub_dir}")
        traj_actions, traj_done = [], []
        traj_observations = []
        try:
            with open(os.path.join(input_dir, sub_dir, "actions.txt")) as trajectories_file:
                for line in trajectories_file:
                    step_info = json.loads(line)
                    assert all(
                        [key in step_info for key in ["cam0", "cam1", "cam_car", "angle", "direction", "speed", "time"]]
                    )

                    with Image.open(os.path.join(input_dir, sub_dir, step_info["cam0"])).resize((250, 250)) as image:
                        cam0_array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)

                    with Image.open(os.path.join(input_dir, sub_dir, step_info["cam1"])).resize((250, 250)) as image:
                        cam1_array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)

                    with Image.open(os.path.join(input_dir, sub_dir, step_info["cam_car"])).resize((250, 250)) as image:
                        car_array = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)

                    traj_observations.append(np.concatenate((cam0_array, cam1_array, car_array), axis=2))

                    assert step_info["direction"] in ["forward", "reverse"]
                    traj_actions.append(
                        np.array(
                            [
                                step_info["angle"],
                                0.0 if step_info["direction"] == "forward" else 1.0,
                                step_info["speed"],
                                step_info["time"],
                            ],
                            dtype=np.float32,
                        )
                    )
                    traj_done.append(False)

                # End of trajectory
                traj_done[-1] = True

        except Exception:
            print(f"{sub_dir} couldn't be processed")

        # Shift actions by 1
        # Remove first action and last observation
        traj_actions = traj_actions[1:]
        traj_observations = traj_observations[:-1]
        traj_done = traj_done[:-1]

        # Accumulate results
        assert len(traj_actions) == len(traj_observations) == len(traj_done)
        total_actions += traj_actions
        total_observations += traj_observations
        total_done += traj_done

    X = np.array(total_observations, dtype=np.uint8)
    A = np.array(total_actions, dtype=np.float32)
    A_min = A.min(axis=0)
    A_max = A.max(axis=0)
    A_norm = (total_actions - A_min) / (A_max - A_min)

    result = {
        "X": X,
        "A": A_norm,
        "done": np.array(total_done, dtype=np.uint8),
        "action-unnorm-min": A_min,
        "action-unnorm-max": A_max,
    }

    print(f"X.shape = {X.shape}")
    print(f"A.shape = {A.shape}")
    print(f"min(A) = {A_min}")
    print(f"max(A) = {A_max}")

    return result


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Usage: python rc_car_data_processing.py <input_dir> <output_dir>"
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    output_file = os.path.join(output_dir, "dataset.p")

    assert os.path.isdir(input_dir), f"{input_dir} does not exist"
    assert not os.path.isfile(output_file), f"{output_file} already exists"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing data from {input_dir}")
    result = process_data(input_dir)
    print("Done processing")
    gc.collect()
    print(f"Saving pickle file to {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(result, f)
    print("Done saving pickle file")
