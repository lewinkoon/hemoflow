import click
from functools import partial
import multiprocessing
import hemoflow.helpers as hf
from hemoflow.logger import logger
import os
import pydicom as pd
import numpy as np
import random


@click.group(
    epilog="Check out readme at https://github.com/lewinkoon/hemoflow for more details."
)
def cli():
    """
    Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field.
    """
    pass


@cli.command(help="Create volumetric velocity field from dicom files.")
@click.option("--frames", type=int, help="Number of frames in the sequence.")
def build(frames):

    # def wrapper(fh, rl, ap, voxel, time):
    #     fh = hf.filter(fh, time)
    #     rl = hf.filter(rl, time)
    #     ap = hf.filter(ap, time)

    #     data = hf.tabulate(fh, rl, ap, voxel, time)
    #     hf.export(data, time)
    #     logger.info(f"Trigger time {time} exported with {len(data)} rows.")

    # create a list of dictionaries with the read data
    fh = hf.parse("FH", frames)
    logger.info(f"FH series: {len(fh)} images.")
    rl = hf.parse("RL", frames)
    logger.info(f"RL series: {len(rl)} images.")
    ap = hf.parse("AP", frames)
    logger.info(f"AP series: {len(ap)} images.")

    # list unique trigger times
    timeframes = sorted(set(item["time"] for item in rl))
    logger.info(f"Timeframes: {timeframes}")

    # get volume dimensions
    volume = (fh[0]["pxl"].shape[0], fh[0]["pxl"].shape[1])
    logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px)")

    # get voxel spacing
    voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
    logger.info(
        f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
    )

    for time in timeframes:
        fh_filtered = hf.filter(fh, time)
        rl_filtered = hf.filter(rl, time)
        ap_filtered = hf.filter(ap, time)

        data = hf.tabulate(fh_filtered, rl_filtered, ap_filtered, voxel, time)
        hf.export(data, time)
        logger.info(f"Trigger time {time} exported with {len(data)} rows.")

    # export csv files with multiprocessing
    # worker = partial(wrapper, fh, rl, ap, voxel)
    # with multiprocessing.Pool() as pool:
    #     pool.map(worker, timeframes)
    # logger.info("Script finished successfully.")


@cli.command(help="Check dicom file metadata.")
@click.argument("dicom", required=False, type=click.Path())
def check(dicom):
    if dicom is None:
        # list all files in the directory
        dir = "files"
        file_list = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_list.append(os.path.join(root, file))

        # log warning if no files are found
        if not file_list:
            logger.error("No DICOM files found in the directory.")
            return None

        # select a random file
        dicom = random.choice(file_list)

    with open(dicom, "rb") as f:
        ds = pd.dcmread(f)

        logger.info(f"File: {dicom}")
        logger.info(f"Image shape: {ds.pixel_array.shape}")

        hf.show_tag(ds, 0x0008, 0x103E)  # axis name
        hf.show_tag(ds, 0x0020, 0x0013)  # instance number
        hf.show_tag(ds, 0x0028, 0x0030)  # pixel spacing
        hf.show_tag(ds, 0x0018, 0x0088)  # spacing between slices
        hf.show_tag(ds, 0x0020, 0x1041)  # slice location
        hf.show_tag(ds, 0x0020, 0x9153)  # trigger time


@cli.command(help="Remove exported data.")
def clean():
    path = "output"

    # check first if path exists
    if not os.path.exists(path):
        logger.error("Output files not exported yet.")
    else:
        try:
            os.rmdir(path)
            logger.info(f"Removed {path}")
        except OSError:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed {file_path}")
                except Exception as e:
                    logger.error(f"Error while deleting {file_path}: {e}")
            os.rmdir(path)
            logger.info(f"Removed {path}")
        except:
            logger.error("Output directory cannot be removed.")


@cli.command(help="Fix multiframe dicom file.")
@click.argument("file", required=True, type=click.Path())
def fix(file):
    # check first if path exists
    path = os.path.basename(file)
    axis = os.path.splitext(path)[0]
    if not os.path.isdir(f"files/{axis}"):
        os.makedirs(f"files/{axis}")

    with open(file, "rb") as f:
        ds = pd.dcmread(f)
        n = int(ds.NumberOfFrames)
        logger.info(f"Detected {n} frames.")

        img = ds.pixel_array.squeeze()

        essentials = [0x0008, 0x0010, 0x0018, 0x0020, 0x0028]

        for idx in range(n):
            target = os.path.join(f"files/{axis}", f"img{idx:03d}.dcm")

            file_meta = pd.dataset.FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = pd.uid.MRImageStorage
            file_meta.MediaStorageSOPInstanceUID = pd.uid.generate_uid()
            file_meta.TransferSyntaxUID = pd.uid.ImplicitVRLittleEndian

            tmp_ds = pd.dataset.Dataset()
            tmp_ds.file_meta = file_meta
            tmp_ds.is_little_endian = (
                tmp_ds.file_meta.TransferSyntaxUID.is_little_endian
            )
            tmp_ds.is_implicit_VR = tmp_ds.file_meta.TransferSyntaxUID.is_implicit_VR

            for group in essentials:
                for key, value in ds.group_dataset(group).items():
                    tmp_ds[key] = value

            sfgs = ds.SharedFunctionalGroupsSequence[0]
            pffgs = ds.PerFrameFunctionalGroupsSequence[idx]

            # copy velocity tags
            for tag_name in ["RescaleIntercept", "RescaleSlope", "RescaleType"]:
                if tag_name in pffgs[(0x0028, 0x9145)][0]:
                    value = pffgs[(0x0028, 0x9145)][0][tag_name].value
                    setattr(tmp_ds, tag_name, value)

            # copy velocity tags
            for tag_name in ["SpacingBetweenSlices", "PixelSpacing", "SliceThickness"]:
                if tag_name in pffgs[(0x0028, 0x9110)][0]:
                    value = pffgs[(0x0028, 0x9110)][0][tag_name].value
                    setattr(tmp_ds, tag_name, value)

            # copy trigger time
            for tag_name in ["NominalCardiacTriggerDelayTime"]:
                if tag_name in pffgs[(0x0018, 0x9118)][0]:
                    value = pffgs[(0x0018, 0x9118)][0][tag_name].value
                    setattr(tmp_ds, tag_name, value)

            del tmp_ds.NumberOfFrames
            tmp_ds.InstanceNumber = idx + 1
            tmp_ds.PixelData = img[idx, :].squeeze().tobytes()
            tmp_ds.save_as(target, write_like_original=False)

            logger.info(f"Image exported as {target}")


@cli.command(help="Patch dicom series metadata.")
@click.argument("path", required=True, type=click.Path())
@click.option(
    "--instance",
    is_flag=True,
    help="Patch instance number of each frame.",
)
@click.option(
    "--channels",
    is_flag=True,
    help="Patch image channels.",
)
def patch(path, instance, channels):
    for idx, file in enumerate(os.listdir(path)):
        file_path = os.path.join(path, file)
        with open(file_path, "rb") as f:
            ds = pd.dcmread(f)

            # fix instance number
            if instance:
                pre = ds[0x0020, 0x0013].value
                post = idx
                ds[0x0020, 0x0013].value = post
                logger.info(f"{file}: Changed instance number from {pre} to {post}.")

            if channels:
                ds.pixel_array = np.mean(ds.pixel_array, axis=2)
                logger.info(f"{file}: Fixed image channels.")

            # check output directory
            if not os.path.exists("output"):
                os.makedirs("output")

            ds.save_as(f"output/{idx:02d}.dcm")
