import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_hwr(trajectory, title="HWR Signature"):
    """
    Plot strokes from parsed .hwr trajectory.
    trajectory: dict from read_hwr()
    """
    x = trajectory["x"]
    y = trajectory["y"]
    pen = trajectory["pen_status"]

    plt.figure(figsize=(6, 6))
    plt.title(title)

    # Each stroke = continuous pen_down sequence
    stroke_x, stroke_y = [], []
    for i in range(len(x)):
        if pen[i] == 1:  # pen down
            stroke_x.append(x[i])
            stroke_y.append(y[i])
        else:
            # end of stroke → plot it
            if stroke_x:
                plt.plot(stroke_x, stroke_y, 'k-')
                stroke_x, stroke_y = [], []

    # Last stroke (if not closed)
    # if stroke_x:
    print(stroke_x)
    plt.plot(stroke_x, stroke_y, 'k-')

    plt.gca().invert_yaxis()  # optional: match writing direction
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


import matplotlib.pyplot as plt

def plot_xy_list(trajectory, title="Raw X-Y plot"):
    """
    Plot X vs Y from trajectory dict (ignores pen status).
    """
    x = trajectory["x"].tolist()  # convert numpy array to list
    y = trajectory["y"].tolist()

    print("X list (first 10):", x[:10])
    print("Y list (first 10):", y[:10])

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'bo-', markersize=2, linewidth=1)  # blue line + points
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.show()


# ==== Example usage ====
# traj = read_hwr("example.hwr")
# plot_xy_list(traj, title="Signature XY sequence")


# ==== Example usage ====
# traj = read_hwr("example.hwr")
# plot_hwr(traj, title="Signature from .hwr")


import numpy as np

def read_hwr(file_path):
    """
    Reads a .hwr file (handwriting trajectory).
    Returns: dict with keys: x, y, pen_status, (optional) time/pressure
    """

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Try to infer column structure
    data = []
    for line in lines:
        parts = line.split()
        values = list(map(float, parts))
        data.append(values)

    data = np.array(data)

    # Common case: [x, y, pen_status] or [x, y, time, pen_status]
    result = {}
    if data.shape[1] == 3:
        result["x"] = data[:, 0]
        result["y"] = data[:, 1]
        result["pen_status"] = data[:, 2].astype(int)  # 1 = pen down, 0 = pen up

    elif data.shape[1] == 4:
        result["x"] = data[:, 0]
        result["y"] = data[:, 1]
        result["time"] = data[:, 2]
        result["pen_status"] = data[:, 3].astype(int)

    elif data.shape[1] >= 5:
        result["x"] = data[:, 0]
        result["y"] = data[:, 1]
        result["time"] = data[:, 2]
        result["pressure"] = data[:, 3]
        result["pen_status"] = data[:, 4].astype(int)

    else:
        raise ValueError("Unsupported .hwr format with columns: {}".format(data.shape[1]))

    return result


# ==== Example usage ====
file_path = "/Users/balaji.raok/Documents/online_signature_work/sigComp2011 Full Dataset/sigComp2011-trainingSet/OnlineSignatures/Chinese/TrainingSet/Online Genuine/002_10.HWR"
trajectory = read_hwr(file_path)

print("X:", trajectory["x"])
print("Y:", trajectory["y"])
# print("Pen Status:", trajectory["pen_status"][:10])



# trajectory = read_hwr("/Users/balaji.raok/Downloads/sigComp2011 Full Dataset/sigComp2011-trainingSet/OnlineSignatures/Chinese/TrainingSet/Online Genuine/001_18.HWR")
# plot_hwr(trajectory, title="Example Signature")
plot_xy_list(trajectory, title="Signature XY sequence")

