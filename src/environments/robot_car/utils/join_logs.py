import argparse
import glob
import json
import shutil
import os

# Usage: python join_logs.py <input_dir> <output_dir>
parser = argparse.ArgumentParser()
parser.add_argument("input_dir", type=str)
parser.add_argument("output_dir", type=str)
args = parser.parse_args()

# Find logs in input directory
client_log = glob.glob(os.path.join(args.input_dir, "client*.txt"))
assert len(client_log) == 1, f"Error: Search for client log in directory {args.input_dir} returned {client_log}. Expected exactly one file."
client_log = client_log[0]

server_log = glob.glob(os.path.join(args.input_dir, "server*.txt"))
assert len(server_log) == 1, f"Error: Search for server log in directory {args.input_dir} returned {server_log}. Expected exactly one file."
server_log = server_log[0]

# Read client log and server log
print(f"Reading client log {client_log}")
with open(client_log, 'r') as f:
    client_lines = f.readlines()
assert len(client_lines) > 0, "Error: Client log is empty"

print(f"Reading server log {server_log}")
with open(server_log, 'r') as f:
    server_lines = f.readlines()
assert len(server_lines) > 0, "Error: Server log is empty"

# Convert client log to json objects
client_actions = []
for line_num, line in enumerate(client_lines):
    try:
        obj = json.loads(line)
        client_actions.append(obj)
    except json.decoder.JSONDecodeError:
        print(f"Error: Could not parse client log line {line_num} as JSON: {line}")

# Parse server log
server_actions = {}
current_action_id = None
current_action = {}
for line_num, line in enumerate(server_lines):
    try:
        line_split = line.split(',')

        if line_split[0] == "setid":
            # Line indicates start of a new action
            # Save previous action to dictionary
            if current_action_id is not None:
                server_actions[current_action_id] = current_action
            # Reset state
            current_action_id = line_split[1].strip()
            current_action = {}

        elif line_split[0] == "angle":
            current_action["angle"] = float(line_split[1])

        elif line_split[0] == "forward":
            current_action["direction"] = "forward"
            current_action["speed"] = float(line_split[1])
            current_action["time"] = float(line_split[2])

        elif line_split[0] == "reverse":
            current_action["direction"] = "reverse"
            current_action["speed"] = float(line_split[1])
            current_action["time"] = float(line_split[2])
        
        elif line_split[0] == "take_pic":
            filename = line_split[1].strip()
            if not filename:
                raise Exception(f"Missing filename")
            current_action["cam_car"] = filename
        
        elif line_split[0] == "sleep":
            # Ignore sleep lines
            pass

        else:
            raise Exception(f"Unknown line type: {line_split[0].strip()}")
    
    except Exception as e:
        print(f"Error: Could not parse server log line {line_num}: {line.strip()}")
        print(e)
if current_action_id is not None:
    server_actions[current_action_id] = current_action

print(f"Number of client actions parsed: {len(client_actions)}")
print(f"Number of server actions parsed: {len(server_actions)}")

# Join client and server logs
joined_actions = []
error_actions = set()
for client_action in client_actions:
    # Validate client action has all required fields
    required_fields = {"action_id", "angle", "direction", "speed", "time", "cam0", "cam1"}
    action_containes_field = [field in client_action for field in required_fields]
    if not all(action_containes_field):
        missing_fields = [field for field, contains in zip(required_fields, action_containes_field) if not contains]
        action_id = client_action.get("action_id", None)
        print(f"Error: Client action {action_id} is missing fields {missing_fields}")
        if action_id:
            error_actions.add(action_id)
        continue

    # Find corresponding server action
    action_id = client_action["action_id"]
    if action_id not in server_actions:
        print(f"Error: Could not find server action for client action {action_id}")
        error_actions.add(action_id)
        continue
    server_action = server_actions[action_id]

    # Validate that client and server actions match
    if server_action.get("angle", None) != client_action.get("angle", None):
        print(f"Error: Action {action_id}: Client angle {client_action.get('angle', None)} does not match server angle {server_action.get('angle', None)}")
        error_actions.add(action_id)
    if server_action.get("direction", None) != client_action.get("direction", None):
        print(f"Error: Action {action_id}: Client direction {client_action.get('direction', None)} does not match server direction {server_action.get('direction', None)}")
        error_actions.add(action_id)
    if server_action.get("speed", None) != client_action.get("speed", None):
        print(f"Error: Action {action_id}: Client speed {client_action.get('speed', None)} does not match server speed {server_action.get('speed', None)}")
        error_actions.add(action_id)
    if server_action.get("time", None) != client_action.get("time", None):
        print(f"Error: Action {action_id}: Client time {client_action.get('time', None)} does not match server time {server_action.get('time', None)}")
        error_actions.add(action_id)
    
    # Validate that server action has a camera image
    if "cam_car" not in server_action:
        print(f"Error: Action {action_id}: Server action does not have a filename for car camera image")
        error_actions.add(action_id)

    # Generate joined action
    joined_action = client_action.copy()
    joined_action["cam_car"] = server_action.get("cam_car", None)
    joined_actions.append(joined_action)

# Verify that files in joined actions exist
for action in joined_actions:
    action_id = action["action_id"]

    if action_id in error_actions:
        continue

    for cam_name in ["cam_car", "cam0", "cam1"]:
        assert cam_name in action, f"Error: Action {action_id}: Missing filename for camera {cam_name}"
        photo_file = os.path.join(args.input_dir, action[cam_name])
        if not os.path.isfile(photo_file):
            print(f"Error: Action {action_id}: Could not find file {photo_file} for camera {cam_name}")
            error_actions.add(action_id)

assert len(joined_actions) >= 0, "Error: total number of valid actions is 0"

# Split joined actions into continuous error-free sequences
joined_sequences = []
current_sequence = []
for action in joined_actions:
    if action["action_id"] in error_actions:
        # We got an error, so this is the end of the current sequence
        if len(current_sequence) >= 2:
            joined_sequences.append(current_sequence)
            current_sequence = []
    else:
        current_sequence.append(action)
if len(current_sequence) >= 2:
    joined_sequences.append(current_sequence)

# Print statistics
for param in ["angle", "speed", "time"]:
    values = [action[param] for action in joined_actions]
    print()
    print(f"Min {param}: {min(values)}")
    print(f"Max {param}: {max(values)}")
    print(f"Average {param}: {sum(values) / len(values)}")
print()
print(f"Number of valid action sequences: {len(joined_sequences)}")
print(f"Lengths of each sequence: {[len(sequence) for sequence in joined_sequences]}")

# Write output
for sequence in joined_sequences:
    sequence_id = sequence[0]["action_id"]
    sequence_out_dir = os.path.join(args.output_dir, sequence_id)
    print()
    print(f"Sequence starting with {sequence_id} has length {len(sequence)}")

    if os.path.isdir(sequence_out_dir):
        # Output already exists for sequence
        print(f"Skipping output for directory {sequence_out_dir} because it already exists")
    else:
        # Write log of actions to a file
        os.makedirs(sequence_out_dir)
        output_log = os.path.join(sequence_out_dir, "actions.txt")
        print(f"Writing output log to {output_log}")
        with open(output_log, 'w') as f:
            for action in sequence:
                f.write(json.dumps(action))
                f.write("\n")

        # Copy images to output directory
        print(f"Copying {3*len(sequence)} images to {sequence_out_dir}")
        for action in sequence:
            for cam_name in ["cam_car", "cam0", "cam1"]:
                photo_file = os.path.join(args.input_dir, action[cam_name])
                shutil.copy(photo_file, sequence_out_dir)
