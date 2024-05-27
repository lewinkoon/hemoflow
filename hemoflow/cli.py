import click
from functools import partial
import multiprocessing
import hemoflow.helpers as hf
from hemoflow.logger import logger
import os
import pydicom as pd
import numpy as np


@click.group(
    epilog="Check out readme at https://github.com/lewinkoon/hemoflow for more details."
)
def cli():
    """
    Visualize velocity image series from a phase contrast magnetic resonance imaging study as a three-dimensional vector field.
    """
    pass


@cli.command(help="Create volumetric velocity field from dicom files.")
@click.argument("path", default="files", type=click.Path())
@click.option("--frames", type=int, help="Number of frames in the sequence.")
def build(frames):

    def wrapper(fh, rl, ap, voxel, time):
        fh = hf.filter(fh, time)
        rl = hf.filter(rl, time)
        ap = hf.filter(ap, time)

        data = hf.tabulate(fh, rl, ap, voxel, time)
        hf.export(data, time)
        logger.info(f"Trigger time {time} exported with {len(data)} rows.")

    # create a list of dictionaries with the read data
    fh = hf.parse("FH", frames)
    logger.info(f"FH series: {len(fh)} images.")
    rl = hf.parse("RL", frames)
    logger.info(f"RL series: {len(rl)} images.")
    ap = hf.parse("AP", frames)
    logger.info(f"AP series: {len(ap)} images.")

    # list unique trigger times
    timeframes = sorted(set(item["time"] for item in fh))
    logger.info(f"Timeframes: {len(timeframes)}")

    # get volume dimensions
    volume = (
        fh[0]["pxl"].shape[0],
        fh[0]["pxl"].shape[1],
        len(set(item["loc"] for item in fh)),
    )
    logger.info(f"Volume dimensions: ({volume[0]} px, {volume[1]} px, {volume[2]} px)")

    # get voxel spacing
    voxel = (fh[0]["spacing"][0], fh[0]["spacing"][1], fh[0]["height"])
    logger.info(
        f"Voxel dimensions: ({voxel[0]:.2f} mm, {voxel[1]:.2f} mm, {voxel[2]:.2f} mm)"
    )

    # export csv files with multiprocessing
    worker = partial(wrapper, fh, rl, ap, voxel)
    with multiprocessing.Pool() as pool:
        pool.map(worker, timeframes)
    logger.info("Script finished successfully.")


@cli.command(help="Check dicom file metadata.")
@click.argument("path", required=True, type=click.Path())
def check(path):
    with open(path, "rb") as file:
        ds = pd.dcmread(file)
        logger.info(f"File: {path}")

        try:
            logger.info(f"Axis: {ds[0x0008, 0x103E].value}")
        except:
            logger.error(f"Axis tag not found.")

        try:
            logger.info(f"Instance number: {ds[0x0020, 0x0013].value}")
        except:
            logger.error(f"Instance number not found.")

        try:
            logger.info(f"Image shape: {ds.pixel_array.shape}")
        except:
            logger.error(f"Pixel array not found.")

        try:
            logger.info(f"Pixel spacing: {ds[0x0028, 0x0030].value} mm")
        except:
            logger.error(f"Pixel spacing not found.")

        try:
            logger.info(f"Spacing between slices: {ds[0x0018, 0x0088].value} mm")
        except:
            logger.error(f"Spacing between slices not found.")

        try:
            logger.info(f"Slice location: {ds[0x0020, 0x1041].value}")
        except:
            logger.error(f"Slice location not found.")

        try:
            logger.info(f"Trigger time: {ds[0x0020, 0x9153].value} ms")
        except:
            logger.error(f"Trigger time not found.")


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


@cli.command(help="Fix multiframe dicom file.")
@click.argument("file", required=True, type=click.Path())
def fix(file):
    # check first if path exists
    path = os.path.basename(file)
    axis = os.path.splitext(path)[0]
    if not os.path.isdir(f"files/{axis}"):
        os.mkdir(f"files/{axis}")

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
                if tag_name in sfgs[(0x0040, 0x9096)][0]:
                    value = sfgs[(0x0040, 0x9096)][0][tag_name].value
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


@cli.command(help="Initialize input directory structure.")
@click.argument("path", default="files", type=click.Path())
def init(path):
    # check first if path exists
    if os.path.exists(path):
        logger.error(f"Directory already exists in {os.path.abspath(path)}")
    else:
        os.mkdir(path)
        # create fh directory
        fh_path = os.path.join(path, "FH")
        os.mkdir(fh_path)
        logger.info(f"Created FH directory in {fh_path}")

        # create ap directory
        ap_path = os.path.join(path, "AP")
        os.mkdir(ap_path)
        logger.info(f"Created AP directory in {ap_path}")

        # create rl directory
        rl_path = os.path.join(path, "RL")
        os.mkdir(rl_path)
        logger.info(f"Created RL directory in {rl_path}")


if __name__ == "__main__":
    cli()
