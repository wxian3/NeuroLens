import xmltodict
import os, math
from renderer import process_mesh
from matplotlib import cm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def parse_lenses_xml(xml_path):
    with open(xml_path, 'r', encoding='utf-8') as file:
        my_xml = file.read()

    # Use xmltodict to parse and convert the XML document
    lens_dict = xmltodict.parse(my_xml)

    lenses = lens_dict.get("lensdatabase", {}).get("lens", [])
    lenses_data = []

    for i, lens_model in enumerate(lenses):
        if "calibration" not in lens_model or not lens_model["calibration"]:
            continue
        calibration = lens_model["calibration"]
        sensor_width = 36.0 / float(lens_model.get("cropfactor", 1.0))
        focal_length = None
        fov = None

        if "distortion" in calibration:
            distortions = calibration["distortion"]
            if not isinstance(distortions, list):
                distortions = [distortions]
            for distortion in distortions:
                if "@model" not in distortion:
                    continue
                if distortion["@model"] == "poly3":
                    if "@k1" not in distortion:
                        continue
                    coeffs = [0., float(distortion["@k1"]), 0.]
                    lens_type = "brown"
                elif distortion["@model"] == "ptlens":
                    if "@a" not in distortion or "@b" not in distortion or "@c" not in distortion:
                        continue
                    coeffs = [float(distortion["@a"]), float(distortion["@b"]), float(distortion["@c"])]
                    lens_type = "abc"
                elif distortion["@model"] == "poly5":
                    if "@k1" not in distortion or "@k2" not in distortion:
                        continue
                    coeffs = [float(distortion["@k1"]), float(distortion["@k2"]), 0.]
                    lens_type = "brown"
                else:
                    continue

                focal_length = float(distortion["@focal"])
                fov = 2 * math.atan(sensor_width / (2 * focal_length)) * (180 / math.pi)

                if fov is not None and 60 <= fov <= 80:
                    lens_model_dict = {
                        "sensor_width": sensor_width,
                        "model_name": str(i),
                        "focal": fov,
                        "coeffs": coeffs,
                        "lens_type": lens_type
                    }
                    lenses_data.append(lens_model_dict)
                    break

        elif "vignetting" in calibration:
            vignetting = calibration["vignetting"]

    return lenses_data

def visualize(k1, k2, k3, lens_name):
    cmap = cm.get_cmap('hsv')
    H, W = 15, 15

    radial = np.array([[(j + 0.5, i + 0.5) for j in range(W * 20)] for i in range(H * 20)])
    radial_dist = (radial - np.array([[W * 10, H * 10]])) / np.array([[W * 10, H * 10]])

    r2 = radial_dist ** 2
    r2 = r2[:, :, 0] + r2[:, :, 1]

    vis = r2*(1 + k1*r2**2 + k2*r2**4 + k3*r2**6)
    norm = vis / np.abs(vis).max()

    ori = np.array([[-(np.arctan(radial_dist[i,j,1] / radial_dist[i,j,0]) ) for j in range(W*20)] for i in range(H*20)])
    ori[:, 0:W*10] = np.pi + ori[:, 0:W*10]
    ori[H*10:H*20, W*10:W*20] = 2 * np.pi + ori[H*10:H*20, W*10:W*20]
    ori = ori / ori.max()

    color = np.array([[cmap(ori[i][j]) for j in range(W*20)]for i in range(H * 20)])

    norm = norm ** 0.7
    color_dist = color * norm[:, :, None]

    img = Image.fromarray((color_dist[:, :, :3] * 255).astype(np.uint8))
    img.save("db_frames_vis/" + lens_name + "_colordist.png")

if __name__ == "__main__":
    db_dir = 'db/'
    lens_type = 'abc'
    num_views = 50
    image_size = 512
    render_rgb = True
    out_fd = "db_output"

    for xml_path in os.listdir(db_dir):
        if xml_path.endswith(".xml"):
            lens_name = xml_path.split("/")[-1][:-4]
            lenses_data = parse_lenses_xml(os.path.join(db_dir,xml_path))

            for i in range(len(lenses_data)):
                if lenses_data[i]["lens_type"] == lens_type:
                    process_mesh(lenses_data[i], lens_name, num_views, image_size, render_rgb, out_fd)
