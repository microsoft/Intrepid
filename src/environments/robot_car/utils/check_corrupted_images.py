import os
import json
import cv2 as cv
from tqdm import tqdm

def load_pic(filename, resize=(256,256)):
    pic = cv.imread(filename)
    return pic

def run_check(root_dir):
    # get all subdirectories
    # each should contain a file called actions.txt
    subdirs = [d for d in os.listdir(root_dir) if
                os.path.isdir(os.path.join(root_dir, d)) and
                os.path.isfile(os.path.join(root_dir, d, 'actions.txt'))]
    assert len(subdirs) > 0, f"No subdirectories found in {root_dir}"

    for dir in tqdm(subdirs):
        # read log file for this subdirectory
        with open(os.path.join(root_dir, dir, 'actions.txt')) as f:
            log = f.readlines()
        log = [json.loads(a) for a in log if a.strip() != '']

        # load a single trajectory
        traj_pics = []
        for line in log:
            traj_pics.append([
                os.path.join(root_dir, dir, line['cam0']),
                os.path.join(root_dir, dir, line['cam1']),
                os.path.join(root_dir, dir, line['cam_car']),
            ])

        for traj in tqdm(traj_pics):
            for pic in traj:
                if not os.path.isfile(pic):
                    print(f"File {pic} does not exist")
                loaded = load_pic(pic)
                if loaded is None:
                    print(f"File {pic} could not be loaded")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Root directory containing trajectories in subdirectories")
    args = parser.parse_args()

    run_check(args.root_dir)
