# import cv2
import glob
import random
import imageio
import numpy as np

from PIL import Image, ImageDraw, ImageFont


# nfnames = len(glob.glob("./figures/*png"))
#
# images = []
# for i in range(0, nfnames):
#     images.append(imageio.imread("./figures/fig_%d.png" % (i + 1)))
# imageio.mimsave('./demo.gif', images)
# exit(0)


width = 8
height = 5
world = np.zeros((width, height))
world[1:7, 1] = 1
world[1:7, 2] = 1
world[1:7, 3] = 1

start_state = (0, 1)
tile_size = 100

self_loop = False


background_img = Image.open("./background2.png").resize((width * tile_size, height * tile_size))
duck_img = Image.open("./duck.png").resize((tile_size, tile_size))
duck_img_flip = duck_img.transpose(Image.FLIP_LEFT_RIGHT)
agent_img = Image.open("./agent.png").resize((tile_size, tile_size))

# agent_img_px
agent_img_px = np.zeros((tile_size, tile_size))
agent_img_np = np.array(agent_img)[:, :, :3]
for i in range(tile_size):
    for j in range(tile_size):
        if min(agent_img_np[i, j, :3]) > 240:
            agent_img_px[i, j] = 0
        else:
            agent_img_px[i, j] = 255

agent_img_px = Image.fromarray(np.uint8(agent_img_px))


def transition(state_, action_):

    state_x, state_y = state_

    if action_ == 0:  # left

        new_state_x = state_x - 1
        new_state_y = state_y

    elif action_ == 1:  # right

        new_state_x = state_x + 1
        new_state_y = state_y

    elif action_ == 2:  # top

        new_state_x = state_x
        new_state_y = state_y - 1

    elif action_ == 3:  # bottom

        new_state_x = state_x
        new_state_y = state_y + 1

    else:
        raise AssertionError("Action must be in [0, 1, 2, 3]")

    if 0 <= new_state_x <= width - 1 and 0 <= new_state_y <= height - 1 and world[new_state_x, new_state_y] == 0:

        return (new_state_x, new_state_y)
    else:
        return state_


ctr = 0
def savefig(state, ducks, arrows, known_states):

    global ctr

    image = Image.new("RGB", (2 * width * tile_size, height * tile_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    image.paste(background_img, (0, 0), background_img)

    fnt = ImageFont.load_default()
    # fnt = ImageFont.truetype("arial.ttf", 32)
    # fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
    # draw.text((width * tile_size + tile_size + 10, height * tile_size // 2 + 10),
    #           "Extracted Latent Endogenous State Dynamics", font=fnt,
    #           fill=(0, 0, 0, 255))

    # # Draw agent
    # draw.ellipse(
    #     (
    #         (state[0]) * tile_size,
    #         (state[1]) * tile_size,
    #         (state[0] + 1) * tile_size,
    #         (state[1] + 1) * tile_size
    #     ), fill="red", outline="red")

    image.paste(agent_img, (state[0] * tile_size, state[1] * tile_size), mask=agent_img_px)

    # Place duck
    for duck in ducks:
        duck_x, duck_y, duck_dir = duck
        center = int(duck_x * tile_size), int(duck_y * tile_size)

        if duck_dir == 1:
            image.paste(duck_img, center, duck_img)
        else:
            image.paste(duck_img_flip, center, duck_img)

    z = 0.25
    # Create the state diagram on the right
    for known_state in known_states:
        # Draw agent
        draw.ellipse(
            (
                (width + known_state[0]) * tile_size + z * tile_size,
                (known_state[1]) * tile_size + z * tile_size,
                (width + known_state[0] + 1) * tile_size - z * tile_size,
                (known_state[1] + 1) * tile_size - z * tile_size
            ), fill=(107, 173, 53), outline=(107, 173, 53))

    na = np.array(image)

    cv2.putText(na, 'PPE extracts the latent endogenous state dynamics',
                (width * tile_size + tile_size - 18, height * tile_size // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.77, (0, 0, 0), 2, cv2.LINE_AA)

    # Place arrows
    for arrow in arrows:
        start, end = arrow
        start_x, start_y = int((start[0] + 0.5) * tile_size), int((start[1] + 0.5) * tile_size)
        end_x, end_y = int((end[0] + 0.5) * tile_size), int((end[1] + 0.5) * tile_size)
        # na = cv2.arrowedLine(na, (start_x, start_y), (end_x, end_y), (242, 175, 42), int(0.06 * tile_size))
        na = cv2.arrowedLine(na, (start_x, start_y), (end_x, end_y), (255, 255, 255), int(0.06 * tile_size))

    r = 0.5 - z # 0.4
    # import pdb
    # pdb.set_trace()
    for known_state_1 in known_states:

        x1, y1 = known_state_1

        #################
        if self_loop:
            if x1 == 0 or (x1 == width - 1 and y1 not in [0, height - 1]):     # leftmost

                start_x, start_y = int((x1 + 0.5 - r + width) * tile_size), int((y1 + 0.5) * tile_size)
                end_x, end_y = int((x1 + 0.5 - r - 0.2 + width) * tile_size), int((y1 + 0.5) * tile_size)

                na = cv2.arrowedLine(na,
                                     (start_x, start_y),
                                     (end_x, end_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))

                na = cv2.arrowedLine(na,
                                     (end_x, end_y),
                                     (start_x, start_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))

            if x1 == width - 1 or (x1 == 0 and y1 not in [0, height - 1]):     # rightmost

                start_x, start_y = int((x1 + 0.5 + r + width) * tile_size), int((y1 + 0.5) * tile_size)
                end_x, end_y = int((x1 + 0.5 + r + 0.2 + width) * tile_size), int((y1 + 0.5) * tile_size)

                na = cv2.arrowedLine(na,
                                     (start_x, start_y),
                                     (end_x, end_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))

                na = cv2.arrowedLine(na,
                                     (end_x, end_y),
                                     (start_x, start_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))

            if y1 == 0 or (y1 == height - 1 and x1 not in [0, width - 1]):          # topmost

                start_x, start_y = int((x1 + 0.5 + width) * tile_size), int((y1 + 0.5 - r) * tile_size)
                end_x, end_y = int((x1 + 0.5 + width) * tile_size), int((y1 + 0.5 - r - 0.2) * tile_size)

                na = cv2.arrowedLine(na,
                                     (start_x, start_y),
                                     (end_x, end_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))

                na = cv2.arrowedLine(na,
                                     (end_x, end_y),
                                     (start_x, start_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))

            if y1 == height - 1 or (y1 == 0 and x1 not in [0, width - 1]):         # bottom most

                start_x, start_y = int((x1 + 0.5 + width) * tile_size), int((y1 + 0.5 + r) * tile_size)
                end_x, end_y = int((x1 + 0.5 + width) * tile_size), int((y1 + 0.5 + r + 0.2) * tile_size)

                na = cv2.arrowedLine(na,
                                     (start_x, start_y),
                                     (end_x, end_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))

                na = cv2.arrowedLine(na,
                                     (end_x, end_y),
                                     (start_x, start_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))
        ################

        for known_state_2 in known_states:

            x2, y2 = known_state_2

            if abs(x1 - x2) + abs(y1 - y2) == 1:

                if x1 == x2 and y2 == y1 - 1:

                    # top arrow
                    start_x, start_y = int((x1 + 0.5 + width) * tile_size), int((y1 + 0.5 - r) * tile_size)
                    end_x, end_y = int((x2 + 0.5 + width) * tile_size), int((y2 + 0.5 + r) * tile_size)

                elif x1 == x2 and y2 == y1 + 1:

                    # bottom arrow
                    start_x, start_y = int((x1 + 0.5 + width) * tile_size), int((y1 + 0.5 + r) * tile_size)
                    end_x, end_y = int((x2 + 0.5 + width) * tile_size), int((y2 + 0.5 - r) * tile_size)

                elif x2 == x1 - 1 and y1 == y2:

                    # left arrow
                    start_x, start_y = int((x1 + 0.5 - r + width) * tile_size), int((y1 + 0.5) * tile_size)
                    end_x, end_y = int((x2 + 0.5 + r + width) * tile_size), int((y2 + 0.5) * tile_size)

                elif x2 == x1 + 1 and y1 == y2:

                    # right arrow
                    start_x, start_y = int((x1 + 0.5 + r + width) * tile_size), int((y1 + 0.5) * tile_size)
                    end_x, end_y = int((x2 + 0.5 - r + width) * tile_size), int((y2 + 0.5) * tile_size)

                else:
                    raise AssertionError("Should not reach here")

                if not (800 <= start_x <= 1600 and 800 <= end_x <= 1600 and 0 <= start_y <= 500 and 0 <= end_y <= 500):
                    import pdb
                    pdb.set_trace()

                na = cv2.arrowedLine(na,
                                     (start_x, start_y),
                                     (end_x, end_y),
                                     (106, 51, 170),
                                     int(0.04 * tile_size))

    # Revert back to PIL Image and save
    image = Image.fromarray(na)

    image.save("./figures/fig_%d.png" % (ctr + 1))
    ctr += 1


def get_duck_position(prev_duck_pos):

    duck_position = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)] + \
                    [(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)] + \
                    [(1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3)]

    duck_position = set(duck_position)

    for pos in prev_duck_pos:
        x, y, _ = pos
        duck_position.remove((x, y))

    (x, y) = random.choice(list(duck_position))
    dir = random.randint(0, 1)

    return x, y, dir


def move_ducks(duck_position):

    assert len(duck_position) < 5, "Cannot guarantee no conflict if more than 4 ducks"
    new_duck_position = []

    for pos in duck_position:
        x, y, dir = pos

        allowed_pos = []
        allowed_weight = []

        # if dir=0, then
        #       move to left with 0.6 prob,
        #       go up with 0.1 prob,
        #       go down with 0.1 prob,
        #       flip direction with 0.1 prob,
        #       stay same with remaining 0.1 prob
        #
        # if dir=1, then
        #       move to right with 0.6 prob,
        #       go up with 0.1 prob,
        #       go down with 0.1 prob,
        #       flip direction with 0.1 prob,
        #       stay same with remaining 0.1 prob
        #
        # if any action is not allowed then we remove it

        if dir == 0:

            if 1 <= x - 1 <= 6 and (x - 1, y, dir) not in new_duck_position:
                allowed_pos.append((x - 1, y, dir))
                allowed_weight.append(0.6)
        else:

            if 1 <= x + 1 <= 6 and (x + 1, y, dir) not in new_duck_position:
                allowed_pos.append((x + 1, y, dir))
                allowed_weight.append(0.6)

        if 1 <= y - 1 <= 3 and (x, y - 1, dir) not in new_duck_position:
            allowed_pos.append((x, y - 1, dir))
            allowed_weight.append(0.1)

        if 1 <= y + 1 <= 3 and (x, y + 1, dir) not in new_duck_position:
            allowed_pos.append((x, y + 1, dir))
            allowed_weight.append(0.1)

        if (x, y, 1 - dir) not in new_duck_position:
            allowed_pos.append((x, y, 1 - dir))
            allowed_weight.append(0.1)

        if (x, y, dir) not in new_duck_position:
            allowed_pos.append((x, y, dir))
            allowed_weight.append(0.1)

        allowed_weight = np.array(allowed_weight)
        allowed_weight = allowed_weight / float(allowed_weight.sum())

        pos_ix = np.random.choice(np.arange(len(allowed_pos)), size=1, replace=False, p=allowed_weight)[0]
        pos = allowed_pos[pos_ix]

        new_duck_position.append(pos)

    return new_duck_position


def play(path, known_states):

    duck_position = []
    for _ in range(2):
        duck_coord = get_duck_position(duck_position)
        duck_position.append(duck_coord)

    arrows = []
    state = start_state

    savefig(state, duck_position, arrows, known_states)

    for action in path:

        new_state = transition(state, action)

        # move duck
        duck_position = move_ducks(duck_position)

        # add arrow
        if state != new_state:
            arrows.append((state, new_state))

        state = new_state

        savefig(state, duck_position, arrows, known_states)


state_paths_queue = {start_state: []}
visited_states = set()

for h in range(1, 11):

    new_state_paths_queue = dict()

    for state, path in state_paths_queue.items():

        for action in [0, 1, 2, 3]:

            new_state = transition(state, action)

            if new_state not in new_state_paths_queue:
                new_path = list(path)
                new_path.append(action)
                new_state_paths_queue[new_state] = new_path

    state_paths_queue = new_state_paths_queue
    known_states = state_paths_queue.keys()

    print("Step=%d: Number of unique states found = %d" % (h, len(known_states)))

    # Play each path
    for state, path in state_paths_queue.items():
        if state not in visited_states:
            visited_states.add(state)
            play(path, known_states)
        # else:
        #     if random.random() < 0.05:
        #         play(path, known_states)

# Create gifs
nfnames = len(glob.glob("./figures/*png"))

images = []
for i in range(0, nfnames):
    images.append(imageio.imread("./figures/fig_%d.png" % (i + 1)))

if self_loop:
    imageio.mimsave('./demo_self_loop.gif', images)
else:
    imageio.mimsave('./demo_no_self_loop.gif', images)
