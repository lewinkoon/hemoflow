import csv
import numpy as np
import os
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_modality_lut
import vtk
from hemoflow.logger import logger


def parse(axis):
    res = []
    folder_path = f"files/{axis}"
    for idx, file in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        slice = {}
        with open(file_path, "rb") as file:
            ds = dcmread(file)

            # assign image array
            if ds[0x0028, 0x0004].value == "MONOCHROME2":
                img = ds.pixel_array
            else:
                img = np.mean(ds.pixel_array, axis=2)
            slice["pxl"] = img
            slice["val"] = apply_modality_lut(img, ds)

            # assign tags
            slice["axis"] = ds[0x0008, 0x103E].value  # axis name
            slice["num"] = ds[0x0020, 0x0013].value  # instance number
            slice["spacing"] = ds[0x0028, 0x0030].value  # pixel spacing
            slice["height"] = ds[0x0018, 0x0088].value  # spacing between slices
            slice["time"] = int(ds[0x0020, 0x9153].value)  # trigger time
            # slice["time"] = int(ds[0x0018, 0x1060].value)  # trigger time

            res.append(slice)
    return res


def filter(series, frame):
    # filter images by axis and time
    res = [slice["val"] for slice in series if slice["time"] == frame]
    return res


def tabulate(fh, rl, ap, voxel, time):
    # convert data into tabular dictionary
    dimensions = fh[0].shape
    res = []
    for z, (imgx, imgy, imgz) in enumerate(zip(ap, fh, rl)):
        for index, (pxlx, pxly, pxlz) in enumerate(
            zip(imgx[::-1].flatten(), imgy[::-1].flatten(), imgz[::-1].flatten())
        ):
            row = {}
            row["x"] = np.unravel_index(index, dimensions)[1] * voxel[0]
            row["y"] = np.unravel_index(index, dimensions)[0] * voxel[1]
            row["z"] = z * voxel[2]
            row["t"] = time
            row["vx"] = pxlx
            row["vy"] = pxly
            row["vz"] = pxlz
            res.append(row)
    return res


def mask(fh, rl, ap, mk):
    fh, rl, ap = [], [], []
    for imgx, imgy, imgz, imgm in zip(ap, fh, rl, mk):
        imgx[imgm == 0] = 0
        imgy[imgm == 0] = 0
        imgz[imgm == 0] = 0

        fh.append(imgy)
        rl.append(imgz)
        ap.append(imgx)

    return fh, rl, ap


def tocsv(data, time):
    if not os.path.exists("output"):
        os.makedirs("output")

    fields = data[0].keys()
    path = f"output/data.csv.{time}"
    with open(path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def tovtk(data, time):
    if not os.path.exists("output"):
        os.makedirs("output")

    points = vtk.vtkPoints()

    vectors = vtk.vtkFloatArray()
    vectors.SetNumberOfComponents(3)
    vectors.SetName("Velocity")

    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetName("Time")

    for point in data:
        points.InsertNextPoint(point["x"], point["y"], point["z"])

        vector = (point["vx"], point["vy"], point["vz"])
        vectors.InsertNextTuple(vector)

        scalar = point["t"]
        scalars.InsertNextTuple([scalar])

    sgrid = vtk.vtkStructuredGrid()
    sgrid.SetDimensions(128, 128, 40)
    sgrid.SetPoints(points)
    sgrid.GetPointData().SetVectors(vectors)
    sgrid.GetPointData().SetScalars(scalars)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(f"output/data.vts.{time}")
    writer.SetInputData(sgrid)
    writer.Write()


def showtag(dataset, group, element):
    try:
        tag_name = dataset[group, element].name
        tag_value = dataset[group, element].value
        logger.info(f"{tag_name}: {tag_value}")
    except:
        logger.error(f"[{group:04x},{element:04x}]: Not found.")


def wrapper(raw, fh, rl, ap, voxel, time):
    fh_filtered = filter(fh, time)
    rl_filtered = filter(rl, time)
    ap_filtered = filter(ap, time)

    data = tabulate(fh_filtered, rl_filtered, ap_filtered, voxel, time)
    if raw:
        tocsv(data, time)
    else:
        tovtk(data, time)
    logger.info(f"Trigger time {time} exported with {len(data)} rows.")
