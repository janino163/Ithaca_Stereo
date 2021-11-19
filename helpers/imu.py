import datetime
import numpy as np
import glob
import csv
import codecs
import os.path as osp
import pymap3d as pm
from collections import namedtuple
from tqdm.auto import tqdm
import os
import datetime
from scipy.spatial.transform import Rotation as R
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from pcutils.plotly_utils import *
__all__ = ["utctoweekseconds", "read_inspva_os2", "read_bestpos_os2", "read_corrimu_os2", "dofs2imu", "oxts2pose", "get_infos_os2", "motion_compensation", "transform_points","load_velo_scan", "cart2hom"]

IMU = namedtuple('IMU', ['lat', 'lon', 'alt', 'roll', 'pitch', 'yaw'])

def utctoweekseconds(unix: int, leapseconds: int):
    """ Returns the GPS week, the GPS day, and the seconds
        and microseconds since the beginning of the GPS week """
    datetimeformat = "%Y-%m-%d %H:%M:%S"
    epoch = datetime.datetime.strptime("1980-01-06 00:00:00", datetimeformat)
    utc = datetime.datetime.utcfromtimestamp(unix)
    tdiff = utc - epoch + datetime.timedelta(seconds=leapseconds)
    gpsweek = tdiff.days // 7
    gpsdays = tdiff.days - 7*gpsweek
    gpsseconds = tdiff.seconds + 86400.0 * \
        (tdiff.days - 7*gpsweek)+tdiff.microseconds/(1e6)
    return gpsweek, gpsdays, gpsseconds,tdiff.microseconds

def read_inspva_os2(filename: str):
    if '\0' in open(filename).read():
        nb = True
    else:
        nb = False

    if(nb):
        # Clean null byte, otherwise Error: line contains NULL byte
        # Save the cleaned file in temporary file .tmp
        with codecs.open(filename, 'rb', 'utf-8') as myfile:
            data = myfile.read()
            # clean file first if dirty
            if data.count('\x00'):
                print('Cleaning null byte')
                with codecs.open('my.csv.tmp', 'w', 'utf-8') as of:
                    for line in data:
                        of.write(line.replace('\x00', ''))
        filename = "my.csv.tmp"

    ###########################################################################
    with codecs.open(filename, 'rb', 'utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        time = []
        week = []
        seconds = []
        infos = []
        for row in csv_reader:
            time.append(int(row[0]))
            week.append(int(row[9]))
            seconds.append(float(row[10]) / 1000.)
            # 'field.latitude', 'field.longitude', 'field.height', 
            # 'field.roll', 'field.pitch', 'field.azimuth'
            # 'field.east_velocity', 'field.north_velocity', 'field.up_velocity'
            infos.append((float(row[11]), float(row[12]), float(row[13]),
                          float(row[17]), float(row[18]), float(row[19]),
                          float(row[15]), float(row[14]), float(row[16]),
                         ))
    time = np.array(time)
    infos = np.array(infos)
    ###########################################################################

    if(nb):
        # Remove the temporary cleaned file
        os.remove("my.csv.tmp")

    return time, infos, week, seconds

def read_bestpos_os2(filename: str):
    if '\0' in open(filename).read():
        nb = True
    else:
        nb = False

    if(nb):
        # Clean null byte, otherwise Error: line contains NULL byte
        # Save the cleaned file in temporary file .tmp
        with codecs.open(filename, 'rb', 'utf-8') as myfile:
            data = myfile.read()
            # clean file first if dirty
            if data.count('\x00'):
                print('Cleaning null byte')
                with codecs.open('my.csv.tmp', 'w', 'utf-8') as of:
                    for line in data:
                        of.write(line.replace('\x00', ''))
        filename = "my.csv.tmp"

    ###########################################################################
    with codecs.open(filename, 'rb', 'utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        time = []
        week = []
        seconds = []
        infos = []
        for row in csv_reader:
            time.append(int(row[0]))
            week.append(int(row[9]))
            seconds.append(float(row[10]) / 1000.)
            # 'field.lat', 'field.lon', 'field.hgt', 'field.hgt_stdev', 'field.pos_type.type', 
            # 'field.diff_age', 'field.sol_age', field.num_svs, field.num_sol_svs, field.num_sol_l1_svs, field.num_sol_multi_svs
            # 'field.lat_stdev', 'field.lon_stdev', 'field.sol_status'
            infos.append((float(row[13]), float(row[14]), float(row[15]), float(row[20]), float(row[12]), 
                          float(row[22]), float(row[23]), float(row[24]), float(row[25]), float(row[26]), float(row[27]),
                          float(row[18]), float(row[19]), float(row[11]),))
    time = np.array(time)
    infos = np.array(infos)
    ###########################################################################

    if(nb):
        # Remove the temporary cleaned file
        os.remove("my.csv.tmp")

    return time, infos, week, seconds

def read_corrimu_os2(filename: str, freq=125):
    if '\0' in open(filename).read():
        nb = True
    else:
        nb = False

    if(nb):
        # Clean null byte, otherwise Error: line contains NULL byte
        # Save the cleaned file in temporary file .tmp
        with codecs.open(filename, 'rb', 'utf-8') as myfile:
            data = myfile.read()
            # clean file first if dirty
            if data.count('\x00'):
                print('Cleaning null byte')
                with codecs.open('my.csv.tmp', 'w', 'utf-8') as of:
                    for line in data:
                        of.write(line.replace('\x00', ''))
        filename = "my.csv.tmp"

    ###########################################################################
    with codecs.open(filename, 'rb', 'utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        time = []
        week = []
        seconds = []
        infos = []
        for row in csv_reader:
            time.append(int(row[0]))
            week.append(int(row[9]))
            seconds.append(float(row[10]) / 1000.)
            imu_data_count = int(row[11])
            # field.roll_rate, field.pitch_rate,  field.yaw_rate
            infos.append((float(row[13]) * freq / imu_data_count, float(row[12]) * freq / imu_data_count, float(row[14]) * freq / imu_data_count, imu_data_count))
    time = np.array(time)
    infos = np.array(infos)
    ###########################################################################

    if(nb):
        # Remove the temporary cleaned file
        os.remove("my.csv.tmp")

    return time, infos, week, seconds


def dofs2imu(dofs):
    # https://docs.novatel.com/oem7/Content/SPAN_Logs/INSATT.htm
    return IMU(dofs[0], dofs[1], dofs[2], dofs[3] / 180. * np.pi, dofs[4] / 180. * np.pi, -dofs[5] / 180. * np.pi)

def oxts2pose(imu, imu_0):
    trans = np.zeros((4, 4))
    trans[:3, 3] = pm.geodetic2enu(imu.lat, imu.lon, imu.alt,
                                   imu_0.lat, imu_0.lon, imu_0.alt)
    trans[:3, :3] = R.from_euler(
        'yxz', [imu.roll, imu.pitch, imu.yaw]).as_matrix()
    trans[3, 3] = 1
    Tr_0 = np.eye(4)
    Tr_0[:3, :3] = R.from_euler(
        'yxz', [imu_0.roll, imu_0.pitch, imu_0.yaw]).as_matrix()
    return np.linalg.solve(Tr_0, trans)

def get_infos_os2(path):
    lidar_path = osp.join(path, "decoded_lidar")
    cam1_path = osp.join(path, "cam1")
    cam1_files =sorted(
        glob.glob(osp.join(cam1_path, "*")))
    lidar_files = sorted(
        glob.glob(osp.join(lidar_path, "*")))
    lidar_timestamp = []
    for f in lidar_files:
        lidar_timestamp.append(int(f.split("/")[-1].split(".")[0]))
    lidar_timestamp = np.array(lidar_timestamp)
    lidar_secs = []
    for time in lidar_timestamp:
        gpsweek, gpsdays, gpsseconds, tdiff_microsec = utctoweekseconds(
            time/1000000000.0, 18)
        lidar_secs.append(gpsseconds)
    lidar_secs = np.array(lidar_secs)
    lidar_secs -= 37
    timestamp, infos, imu_week, imu_secs = read_inspva_os2(osp.join(path, "inspva.csv"))


    _, corrimu_infos, corrimu_week, corrimu_secs = read_corrimu_os2(osp.join(path, "corrimu.csv"), freq=125)
    _, bestpos_infos, bestpos_week, bestpos_secs = read_bestpos_os2(osp.join(path, "bestpos.csv"))
    return lidar_files, lidar_secs, cam1_files, infos, imu_secs, corrimu_infos, corrimu_secs, bestpos_infos, bestpos_secs

def motion_compensation(lidar_path, imu_info, ang_info, TL2I, return_original=False):
    scan = load_velo_scan(lidar_path, 5, np.float64)
    base_time = int(lidar_path.split("/")[-1].split(".")[0])
    _times = scan[:, 4].astype(np.int64)
    delta_time = (_times - base_time) / 1e9
    ptc0 = scan[:, :3].astype(np.float32)
    ptc0_in_IMU = transform_points(ptc0, TL2I)
    velo_in_ENU = imu_info[-3:]
    R_ENU2IMU = R.from_euler(
        'yxz', [dofs2imu(imu_info).roll, dofs2imu(imu_info).pitch, dofs2imu(imu_info).yaw]).as_matrix()
    velo_in_IMU = R_ENU2IMU.T @ velo_in_ENU
    angular_rot_in_IMU = ang_info * delta_time.reshape(-1, 1)
    # in roll, pitch, yield rate
    angular_rot_in_IMU = R.from_euler('yxz', angular_rot_in_IMU).as_matrix()

    offset_compensation = velo_in_IMU * delta_time.reshape(-1, 1)
    ptc0_compensated = np.einsum(
        'nd,ned->ne', ptc0_in_IMU, angular_rot_in_IMU) + offset_compensation.astype(np.float32)
    ptc0_compensated = transform_points(ptc0_compensated, np.linalg.inv(TL2I))
    if return_original:
        return ptc0, ptc0_compensated, delta_time
    else:
        return ptc0_compensated

def transform_points(pts_3d_ref, Tr):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]

def load_velo_scan(velo_filename, lidar_channels=5, dtype=np.float64):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, lidar_channels))
    return scan

def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    return pts_3d_hom

def labels_to_imu(box_3D, imu_info, ang_info, TL2I, return_original=False):
    

    ptc0_in_IMU = transform_points(ptc0, TL2I)
    
    velo_in_ENU = imu_info[-3:]
    
    R_ENU2IMU = R.from_euler(
        'yxz', [dofs2imu(imu_info).roll, dofs2imu(imu_info).pitch, dofs2imu(imu_info).yaw]).as_matrix()
    
    velo_in_IMU = R_ENU2IMU.T @ velo_in_ENU
    
    angular_rot_in_IMU = ang_info * delta_time.reshape(-1, 1)
    # in roll, pitch, yield rate
    angular_rot_in_IMU = R.from_euler('yxz', angular_rot_in_IMU).as_matrix()

    offset_compensation = velo_in_IMU * delta_time.reshape(-1, 1)
    
    ptc0_compensated = np.einsum(
        'nd,ned->ne', ptc0_in_IMU, angular_rot_in_IMU) + offset_compensation.astype(np.float32)

    ptc0_compensated = transform_points(ptc0_compensated, np.linalg.inv(TL2I))
    return box_3D