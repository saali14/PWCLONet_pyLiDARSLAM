from typing import Optional, List, Union, Tuple

import matplotlib
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2


import os
import sys

project_path = os.getenv('RELIDAR_SLAM_ABS_PATH')
if project_path is None:
    raise RuntimeError('Please set the following environment variable: `RELIDAR_SLAM_ABS_PATH`')
sys.path.insert(0, project_path)

from pyLiDAR_SLAM.slam.common.utils import assert_debug, check_tensor
from pyLiDAR_SLAM.slam.common.io import poses_to_df, delimiter


from enum import Enum
import matplotlib.animation as animation


class FrameState(Enum):
    NORMAL_FRAME = 0
    FIRST_FRAME = 1
    LAST_FRAME = 2
    SPECIAL_FRAME = 3


class TrajectoryPlotter:

    def __init__(self, output_file: str, labels: list = ['prediction', 'gt'],
                 figsize: Optional[Tuple] = None, font_size: int = 20,
                 palette: Optional[list] = None, with_gt: bool = False):
        self.xdata, self.ydata = [], []

        self.output_file = output_file
        self.labels = labels
        self.figsize = figsize
        self.font_size = font_size
        self.palette = palette
        if palette is None:
            self.palette = "tab10"

        self.with_gt = with_gt
        self.n_colors = 1
        if with_gt:
            self.n_colors = 2

        self.launch()


    def init(self):
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, 1)

        del self.xdata[:]
        del self.ydata[:]

        for i in range(self.n_colors):
            self.lines[i].set_data(self.xdata, self.ydata)

        return self.lines,


    def launch(self):
        sns.set_theme(style="darkgrid")

        #Set up plot
        self.figure, self.ax = plt.subplots(figsize=self.figsize if self.figsize is not None else (10., 10.), dpi=1000)#, clear=True)
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-1.1, 1.1)
        #Other stuff
        #self.ax.grid()

        matplotlib.rcParams.update({'font.size': self.font_size})
        plt.rc('font', size=self.font_size)
        plt.rc('axes', labelsize=self.font_size)
        plt.rc('axes', titlesize=self.font_size)
        plt.rc('xtick', labelsize=self.font_size)
        plt.rc('ytick', labelsize=self.font_size)
        plt.rc('xtick', labelsize=self.font_size)
        plt.rc('legend', fontsize=self.font_size)
        plt.rc('legend', title_fontsize=self.font_size)

        color_palette = sns.color_palette(self.palette, n_colors=self.n_colors, as_cmap=True)
        
        self.lines = []
        for i in range(self.n_colors):
            lines, = self.ax.plot([],[], linewidth=4, label=self.labels[i], color=color_palette.colors[i])
            self.lines.append(lines)

        self.ax.set_xlabel("x[m]")
        self.ax.set_ylabel("y[m]")

        leg = self.ax.legend(loc="lower left")
        for line in leg.get_lines():
            line.set_linewidth(4.0)

        plt.axis("equal")
        self.figure.set_dpi(100)


    def animation(self, data_gen: callable, interval: int, save_count: int, **args):
        ani = animation.FuncAnimation(self.figure, self.run, data_gen, fargs=args, interval=interval, init_func=self.init, save_count=save_count)


    def run(self, data):
        # update the data
        x, y = data
        append_axis = 0
        if self.with_gt:
            append_axis = 1
        self.xdata = np.append(self.xdata, x, axis=append_axis)
        self.ydata = np.append(self.ydata, y, axis=append_axis)
        """xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()

        if x >= xmax:
            self.ax.set_xlim(xmin, 2*xmax)
            self.ax.figure.canvas.draw()"""
        for i, (x, y) in enumerate(zip(self.xdata, self.ydata)):
            self.lines[i].set_data(x, y)

        # rescale
        self.ax.relim()
        self.ax.autoscale_view()
        # draw and flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        self.figure.savefig(self.output_file)

        return self.lines,


    def __del__(self):
        plt.close(self.figure)



def draw_trajectory_files(xs: list, ys: list,
                          output_file: str, labels: list = None,
                          figsize: Optional[Tuple] = None, font_size: int = 20,
                          palette: Optional[list] = None,
                          interactive_mode: bool = False):
    """
    Draws multiple 2D trajectories in matplotlib plots and saves the plots as png images

    Parameters
    ----------
    xs : list of arrays
        The list of xs arrays
    ys : list of arrays
        The list of ys arrays
    output_file :
        The output file
    labels : Optional[list]
        An optional list of labels to be displayed in the trajectory
    figsize : Optional[Tuple]
    font_size : int
        The font size of the legend

    """
    if interactive_mode:
        plt.ion()
        lines = []
    else:
        plt.ioff()
    sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=figsize if figsize is not None else (10., 10.), dpi=1000, clear=True)

    matplotlib.rcParams.update({'font.size': font_size})
    plt.rc('font', size=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('legend', title_fontsize=font_size)

    axes = plt.gca()

    color_palette = sns.color_palette(palette="tab10" if palette is None else palette, n_colors=len(xs), as_cmap=True)
    for i, (x, y) in enumerate(zip(xs, ys)):
        line, = axes.plot(x, y, linewidth=4, label=labels[i], color=color_palette.colors[i])
        if interactive_mode:
            lines.append(line)

    axes.set_xlabel("x[m]")
    axes.set_ylabel("y[m]")

    leg = axes.legend(loc="lower left")
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    plt.axis("equal")
    fig.set_dpi(100)
    plt.savefig(output_file)
    if not interactive_mode:
        plt.close(fig)
        return None, None, None
    return fig, lines, axes


def update_trajectory(xs: list, ys: list, fig, lines, axes,
            output_file: str, labels: list = None,
            figsize: Optional[Tuple] = None, font_size: int = 20,
            palette: Optional[list] = None):

    #axes.clear()
    for i, (x, y) in enumerate(zip(xs, ys)):
        lines[i].set_xdata(np.append(lines[i].get_xdata(), x))
        lines[i].set_ydata(np.append(lines[i].get_ydata(), y))
        #axes.plot(x, y, linewidth=4)#, label=labels[i], color=color_palette.colors[i])
            
    fig.gca().relim()
    fig.gca().autoscale_view() 

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.savefig(output_file)


def list_poses_to_poses_array(poses_list: list):
    return np.concatenate([np.expand_dims(pose, axis=0) for pose in poses_list], axis=0)


def shift_poses(poses: np.ndarray):
    """
        shifts the poses by one frame
    """
    shifted = poses[:-1, :4, :4] # -- ME -- remove the last pose
    shifted = np.concatenate([np.expand_dims(np.eye(4), axis=0), shifted], axis=0) # -- ME -- add I as a first pose
    return shifted


def compute_relative_poses(poses: np.ndarray):
    shifted = shift_poses(poses)
    # -- ME -- (n, m, k) @ (n, k, p) -> (n, m, p)
    # here (n, 4, 4) @ (n, 4, 4) -> (n, 4, 4)
    # abs_pose(t-1) @ abs_pose(t) -> rel_pose(t)
    relative_poses = np.linalg.inv(shifted) @ poses
    return relative_poses


def compute_absolute_poses_(relative_poses: np.ndarray, absolute_poses: np.ndarray):
    for i in range(absolute_poses.shape[0] - 1):
        pose_i = absolute_poses[i, :, :].copy()
        r_pose_ip = relative_poses[i + 1, :, :].copy()
        absolute_poses[i + 1, :] = np.dot(pose_i, r_pose_ip)


def compute_absolute_poses(relative_poses: np.ndarray):
    absolute_poses = relative_poses.copy()
    compute_absolute_poses_(relative_poses, absolute_poses)
    return absolute_poses


def compute_cumulative_trajectory_length(trajectory: np.ndarray):
    """
        the cumulative distance traveled between each two consecutive frames
    """
    shifted = shift_poses(trajectory) # -- ME -- shifts the trajectory by one frame: (n, 4, 4) where n = len(trajectory)
    lengths = np.linalg.norm(shifted[:, :3, 3] - trajectory[:, :3, 3], axis=1) # -- ME -- lengths between each two consecutive frames (n,)
    lengths = np.cumsum(lengths) # -- ME -- np.cumsum([1, 2, 3]) = [1, 3, 6]
    return lengths


def rotation_error(pose_err: np.ndarray):
    _slice = []
    if len(pose_err.shape) == 3:
        _slice.append(slice(pose_err.shape[0]))

    a = pose_err[tuple(_slice + [slice(0, 1), slice(0, 1)])]
    b = pose_err[tuple(_slice + [slice(1, 2), slice(1, 2)])]
    c = pose_err[tuple(_slice + [slice(2, 3), slice(2, 3)])]
    d = 0.5 * (a + b + c - 1.0)
    error = np.arccos(np.maximum(np.minimum(d, 1.0), -1.0))
    error = error.reshape(error.shape[0])
    return error


def translation_error(pose_err: np.ndarray):
    _slice = []
    axis = 0
    if len(pose_err.shape) == 3:
        _slice.append(slice(pose_err.shape[0]))
        axis = 1

    return np.linalg.norm(pose_err[tuple(_slice + [slice(3), slice(3, 4)])], axis=axis)


def lastFrameFromSegmentLength(dist: list, first_frame: int, segment: float) -> int:
    """
        ### -- ME -- 
        starting from first_frame, we search for the frame
        when we would have traveled a distance=segment.
    """
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + segment:
            return i
    return -1


__default_segments = [100, 200, 300, 400, 500, 600, 700, 800]


def calcSequenceErrors(trajectory, ground_truth, all_segments=__default_segments, step_size: int = 10) -> list:
    """
        ### -- ME --
        calculate pose errors between frames.\n
        * we iterate on frames by a step equal to `step_size`.\n
        * for each frame, we consider cumulative pose error
        for a distance of each `segment` in `all_segments`.\n\n
        * return `tr_err`, `r_err`, `segment`, `speed`, `first_frame`, `last_frame`
    """
    dist = compute_cumulative_trajectory_length(ground_truth) # -- ME -- (n_pose,)
    n_poses = ground_truth.shape[0]

    errors = []
    for first_frame in range(0, n_poses, step_size):
        for segment_len in all_segments:
            last_frame = lastFrameFromSegmentLength(dist, first_frame, segment_len)

            if last_frame == -1:
                continue

            # -- ME --
            # calculate the pose error between two frames 
            # where the lidar traveled a distance equal to segment
            pose_delta_gt = np.linalg.inv(ground_truth[first_frame]).dot(ground_truth[last_frame])
            pose_delta_traj = np.linalg.inv(trajectory[first_frame]).dot(trajectory[last_frame])
            pose_err = np.linalg.inv(pose_delta_traj).dot(pose_delta_gt)

            r_err = rotation_error(pose_err)
            t_err = translation_error(pose_err)

            num_frames = last_frame - first_frame + 1
            speed = segment_len / (0.1 * num_frames)

            errors.append({"tr_err": t_err / segment_len,
                           "r_err": r_err / segment_len,
                           "segment": segment_len,
                           "speed": speed,
                           "first_frame": first_frame,
                           "last_frame": last_frame})

    return errors


def compute_kitti_metrics(trajectory, ground_truth, segments_sizes=__default_segments) -> tuple:
    """
        ### -- ME --
        calculate pose errors between frames.\n
        * we iterate on frames by a step equal to `step_size`.\n
        * for each frame, we consider cumulative pose error
        for a distance of each `segment` in `all_segments`.\n\n
        * return `avg_tr_err`, `avg_rot_err`, `errors`
        where errors' keys are: `tr_err`, `r_err`, `segment`, `speed`, `first_frame`, `last_frame`
    """
    errors = calcSequenceErrors(trajectory, ground_truth, segments_sizes)

    if len(errors) > 0:
        # Compute averaged errors
        tr_err = sum([error["tr_err"] for error in errors])[0]
        rot_err = sum([error["r_err"] for error in errors])[0]
        tr_err /= len(errors)
        rot_err /= len(errors)
        return tr_err, rot_err, errors
    return None, None


def compute_ate(relative_predicted, relative_ground_truth) -> Tuple[float, float]:
    pred_xyz = relative_predicted[:, :3, 3]
    gt_xyz = relative_ground_truth[:, :3, 3]

    tr_err = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    ate = tr_err.mean()
    std_dev = np.sqrt(np.power(tr_err - ate, 2).mean())

    return ate, std_dev


def compute_are(relative_trajectory, relative_ground_truth) -> Tuple[float, float]:
    # -- ME --
    # np.linalg.inv(rot_gt) @ rot_pred should be equal 
    # to np.eye(3) if the prediction was perfect
    diff = np.linalg.inv(relative_ground_truth[:, :3, :3]) @ relative_trajectory[:, :3, :3] - np.eye(3)
    r_err = np.linalg.norm(diff, axis=(1, 2))
    are = r_err.mean()
    std_dev = np.sqrt(np.power(r_err - are, 2).mean())
    return are, std_dev


def compute_2d_tr_err(relative_predicted, relative_ground_truth) -> Tuple[float, float]:
    pred_xy = relative_predicted[:, :2, 3]
    gt_xy = relative_ground_truth[:, :2, 3]

    tr_err = np.linalg.norm(pred_xy - gt_xy, axis=1)
    ate = tr_err.mean()
    std_dev = np.sqrt(np.power(tr_err - ate, 2).mean())

    return ate, std_dev


def compute_2d_rot_err(relative_trajectory, relative_ground_truth) -> Tuple[float, float]:
    # -- ME --
    # np.linalg.inv(rot_gt) @ rot_pred should be equal 
    # to np.eye(3) if the prediction was perfect
    diff = np.linalg.inv(relative_ground_truth[:, 3, :3]) @ relative_trajectory[:, 3, :3] - np.eye(3)[3,:]
    r_err = np.linalg.norm(diff, axis=1)
    are = r_err.mean()
    std_dev = np.sqrt(np.power(r_err - are, 2).mean())
    return are, std_dev


def compute_average(curr, last, curr_avg, last_avg):
    if last_avg == 0:
        return float(curr_avg)
    
    curr_nb = curr - last + 1
    last_nb = last - 1
    total_nb = curr + 1

    #total_avg = (curr_avg + last_avg) * (curr_nb * last_nb) - (last_nb * (curr_nb - 1) * last_avg + curr_nb * (last_nb - 1) * curr_avg)
    total_avg = curr_avg * curr_nb + last_avg * last_nb
    total_avg = total_avg / total_nb

    return float(total_avg)


def rescale_prediction(sequence_pred: np.ndarray, sequence_gt: np.ndarray) -> np.ndarray:
    check_tensor(sequence_pred, [-1, 4, 4])
    check_tensor(sequence_gt, [-1, 4, 4])
    rescaled_pred = []
    for i in range(len(sequence_pred)):
        poses_pred = sequence_pred[i]
        poses_gt = sequence_gt[i]
        norm_pred = np.linalg.norm(poses_pred[:3, -1])
        norm_gt = np.linalg.norm(poses_gt[:3, -1])
        scale = 1.0
        if norm_pred > 1e-6:
            scale = np.linalg.norm(norm_gt) / norm_pred
        new_poses = poses_pred.copy()
        new_poses[:3, -1] *= scale
        rescaled_pred.append(new_poses)

    return list_poses_to_poses_array(rescaled_pred)


class OdometryResults(object):
    """
    An object which aggregrates the results of an Odometry benchmark
    """

    _FRAME_GAP = 100 # 100 meters

    _PRED_FIG = None
    _PRED_LINES = None
    _PRED_AXES = None

    _GT_PRED_FIG = None
    _GT_PRED_LINES = None
    _GT_PRED_AXES = None

    def __init__(self, log_dir: str):
        self.log_dir_path = Path(log_dir)
        if not self.log_dir_path.exists():
            self.log_dir_path.mkdir()

        self.metrics = {}

        self.frame_gaps = {}


    def compute_poses(self, sequence_id: str, 
                    relative_prediction: Union[np.ndarray, List],
                    relative_ground_truth: Optional[Union[np.ndarray, List]],
                    mode: str = "normal", 
                    currFrame: int = -1,
                    lastFrame: int = -1):
        
        assert(currFrame >= lastFrame)

        absolute_gt = None
        with_ground_truth = relative_ground_truth is not None
        if isinstance(relative_prediction, list):
            relative_prediction = list_poses_to_poses_array(relative_prediction)

        absolute_pred = compute_absolute_poses(relative_prediction)

        if (currFrame >= 0) and (lastFrame >= 0):
            assert(relative_prediction.shape[0] == (currFrame + 1))
            relative_prediction = relative_prediction[lastFrame:currFrame+1, :, :]
            absolute_pred = absolute_pred[lastFrame:currFrame+1, :, :]

        if with_ground_truth:
            if isinstance(relative_ground_truth, list):
                relative_ground_truth = list_poses_to_poses_array(relative_ground_truth)

            absolute_gt = compute_absolute_poses(relative_ground_truth)

            # -- ME -- frame == -1 for `add_sequence``
            if (currFrame >= 0) and (lastFrame >= 0):
                relative_ground_truth = relative_ground_truth[lastFrame:currFrame+1, :, :]
                absolute_gt = absolute_gt[lastFrame:currFrame+1, :, :]

            if mode == "rescale_simple":
                relative_prediction = rescale_prediction(relative_prediction, relative_ground_truth)
            elif mode == "eval_rotation":
                relative_prediction[:, :3, 3] = relative_ground_truth[:, :3, 3]
            elif mode == "eval_translation":
                relative_prediction[:, :3, :3] = relative_ground_truth[:, :3, :3]

            assert_debug(list(relative_ground_truth.shape) == list(relative_prediction.shape))

        return relative_prediction, relative_ground_truth, absolute_pred, absolute_gt
    

    def save_poses(self, poses: np.ndarray, filePath: str, frame_state: FrameState = FrameState.NORMAL_FRAME):
        df_poses = poses_to_df(poses)
        if frame_state == FrameState.FIRST_FRAME:
            df_poses.to_csv(filePath, sep=delimiter(), index=False, header=False)
        elif frame_state == FrameState.SPECIAL_FRAME:
            df_poses.to_csv(filePath, mode='a', sep=delimiter(), header=False, index=False)


    # -- ME --
    def add_frames(self, sequence_id: str,
                     currFrame: int,
                     lastFrame: int,
                     relative_prediction: Union[np.ndarray, List],
                     relative_ground_truth: Optional[Union[np.ndarray, List]],
                     elapsed: Optional[float] = None,
                     mode: str = "normal",
                     additional_metrics_filename: Optional[str] = None,
                     frame_state: FrameState = FrameState.NORMAL_FRAME,
                     plot_trajectory: bool = False):
        
        assert(currFrame >= lastFrame)

        if not sequence_id in self.metrics.keys():
            self.frame_gaps[sequence_id] = [currFrame - lastFrame + 1]
        else:
            self.frame_gaps[sequence_id].append(currFrame - lastFrame + 1)
        
        relative_pred, relative_gt, absolute_pred, absolute_gt = self.compute_poses(sequence_id, relative_prediction, relative_ground_truth, mode=mode, currFrame=currFrame, lastFrame=lastFrame)
        with_ground_truth = (absolute_gt is not None)

        if plot_trajectory:
            pred_outputfile = str(self.log_dir_path / f"trajectory_{sequence_id}.png")
            if frame_state == FrameState.FIRST_FRAME:
                OdometryResults._PRED_FIG, OdometryResults._PRED_LINES, OdometryResults._PRED_AXES = draw_trajectory_files([absolute_pred[:, 0, 3]],
                                    [absolute_pred[:, 1, 3]],
                                    output_file=pred_outputfile,
                                    labels=["prediction"], interactive_mode=True)
            elif frame_state == FrameState.SPECIAL_FRAME:
                update_trajectory([absolute_pred[:, 0, 3]], [absolute_pred[:, 1, 3]], 
                                OdometryResults._PRED_FIG, OdometryResults._PRED_LINES, OdometryResults._PRED_AXES,
                                output_file=pred_outputfile, 
                                labels=["prediction"])
        
        self.save_poses(absolute_pred, str(self.log_dir_path / f"{sequence_id}.poses.txt"), frame_state=frame_state)
        

        if with_ground_truth:
            # Save the files

            if plot_trajectory:
                gt_outputfile = str(self.log_dir_path / f"trajectory_{sequence_id}_with_gt.png")
                if frame_state == FrameState.FIRST_FRAME:
                    OdometryResults._GT_PRED_FIG, OdometryResults._GT_PRED_LINES, OdometryResults._GT_PRED_AXES = draw_trajectory_files([absolute_pred[:, 0, 3], absolute_gt[:, 0, 3]],
                                                    [absolute_pred[:, 1, 3], absolute_gt[:, 1, 3]],
                                                    output_file=gt_outputfile,
                                                    labels=["prediction", "GT"], interactive_mode=True)
                elif frame_state == FrameState.SPECIAL_FRAME:
                    update_trajectory([absolute_pred[:, 0, 3], absolute_gt[:, 0, 3]],
                                    [absolute_pred[:, 1, 3], absolute_gt[:, 1, 3]], 
                                    OdometryResults._GT_PRED_FIG, OdometryResults._GT_PRED_LINES, OdometryResults._GT_PRED_AXES,
                                    output_file=gt_outputfile, 
                                    labels=["prediction", "GT"])
            
            self.save_poses(absolute_gt, str(self.log_dir_path / f"{sequence_id}_gt.poses.txt"), frame_state=frame_state)

            # Save the metrics dict
            rel_tr_err, rel_tr_std = compute_ate(relative_pred, relative_gt)
            rel_rot_err, rel_rot_std = compute_are(relative_pred, relative_gt)

            abs_tr_err, abs_tr_std = compute_ate(absolute_pred, absolute_gt)
            abs_rot_err, abs_rot_std = compute_are(absolute_pred, absolute_gt)

            avg_rel_tr_err, avg_rel_tr_std, avg_rel_rot_err, avg_rel_rot_std = 0, 0, 0, 0
            avg_abs_tr_err, avg_abs_tr_std, avg_abs_rot_err, avg_abs_rot_std = 0, 0, 0, 0
                
            nsecs_per_frame = 0

            if sequence_id in self.metrics.keys():
                if '0000_average' in self.metrics[sequence_id].keys():
                    avg_rel_tr_err = self.metrics[sequence_id]['0000_average']['rel_tr_err']
                    avg_rel_tr_std = self.metrics[sequence_id]['0000_average']['rel_tr_std']
                    avg_rel_rot_err = self.metrics[sequence_id]['0000_average']['rel_rot_err']
                    avg_rel_rot_std = self.metrics[sequence_id]['0000_average']['rel_rot_std']
                    
                    avg_abs_tr_err = self.metrics[sequence_id]['0000_average']['abs_tr_err']
                    avg_abs_tr_std = self.metrics[sequence_id]['0000_average']['abs_tr_std']
                    avg_abs_rot_err = self.metrics[sequence_id]['0000_average']['abs_rot_err']
                    avg_abs_rot_std = self.metrics[sequence_id]['0000_average']['abs_rot_std']

                    if 'nsecs_per_frame' in self.metrics[sequence_id]['0000_average'].keys():
                        nsecs_per_frame = self.metrics[sequence_id]['0000_average']['nsecs_per_frame']
            else:
                self.metrics[sequence_id] = {}       

            self.metrics[sequence_id]['0000_average'] = {
                "rel_tr_err":   compute_average(currFrame, lastFrame, rel_tr_err, avg_rel_tr_err),
                "rel_tr_std":   compute_average(currFrame, lastFrame, rel_tr_std, avg_rel_tr_std),
                "rel_rot_err":  compute_average(currFrame, lastFrame, rel_rot_err, avg_rel_rot_err),
                "rel_rot_std":  compute_average(currFrame, lastFrame, rel_rot_std, avg_rel_rot_std),
                "abs_tr_err":   compute_average(currFrame, lastFrame, abs_tr_err, avg_abs_tr_err),
                "abs_tr_std":   compute_average(currFrame, lastFrame, abs_tr_std, avg_abs_tr_std),
                "abs_rot_err":  compute_average(currFrame, lastFrame, abs_rot_err, avg_abs_rot_err),
                "abs_rot_std":  compute_average(currFrame, lastFrame, abs_rot_std, avg_abs_rot_std)
            }

            self.metrics[sequence_id][str(currFrame).zfill(4)] = {
                "rel_tr_err":   float(rel_tr_err),
                "rel_tr_std":   float(rel_tr_std),
                "rel_rot_err":  float(rel_rot_err),
                "rel_rot_std":  float(rel_rot_std),
                "abs_tr_err":   float(abs_tr_err),
                "abs_tr_std":   float(abs_tr_std),
                "abs_rot_err":  float(abs_rot_err),
                "abs_rot_std":  float(abs_rot_std),
            }

            if (elapsed is not None):
                elapsed_time = float(elapsed / absolute_gt.shape[0])
                self.metrics[sequence_id]['0000_average']['nsecs_per_frame'] = compute_average(currFrame, lastFrame, elapsed_time, nsecs_per_frame)
                self.metrics[sequence_id][str(currFrame).zfill(4)]['nsecs_per_frame'] = elapsed_time
                
            self.save_metrics()

            if additional_metrics_filename is not None:
                self.save_metrics(additional_metrics_filename)


    def add_sequence(self, sequence_id: str,
                     relative_prediction: Union[np.ndarray, List],
                     relative_ground_truth: Optional[Union[np.ndarray, List]],
                     elapsed: Optional[float] = None,
                     mode: str = "normal",
                     additional_metrics_filename: Optional[str] = None):
        """
        Computes the odometry metrics ATE, ARE, tr_err, rot_err for the sequence sequence_id,
        Saves the result in the log_dir, the trajectories and the projected images

        Parameters
        ----------
        sequence_id : str
            The id of the sequence, will be used as key / prefix for the metrics computed
        relative_prediction : Union[list, np.ndarray]
            The prediction of the relative poses
        relative_ground_truth : Optional[Union[list, np.ndarray]]
            The ground truth of the relative poses
        elapsed : Optional[float]
            The optional number of seconds elapsed for the acquisition of the sequence
        mode : str
            The mode of evaluation accepted modes are :
            normal : the poses are evaluated against the ground truth
            rescale_simple : the poses are rescaled with respect to the ground truth using a 5 frame snippet
            eval_rotation : the translation are set to the ground truth, the rotations are set by the gt
            eval_translation : the rotation are set to the ground truth, the translations by the gt

        additional_metrics_filename: Optional[str] = None
            An optional path to a metrics file to which the metrics should be appended
        """
        _, _, absolute_pred, absolute_gt = self.compute_poses(sequence_id, relative_prediction, relative_ground_truth, mode=mode)
        with_ground_truth = (absolute_gt is not None)

        draw_trajectory_files([absolute_pred[:, 0, 3]],
                              [absolute_pred[:, 1, 3]],
                              output_file=str(self.log_dir_path / f"trajectory_{sequence_id}.png"),
                              labels=["prediction"])

        if with_ground_truth:
            # Save the files
            draw_trajectory_files([absolute_pred[:, 0, 3], absolute_gt[:, 0, 3]],
                                  [absolute_pred[:, 1, 3], absolute_gt[:, 1, 3]],
                                  output_file=str(self.log_dir_path / f"trajectory_{sequence_id}_with_gt.png"),
                                  labels=["prediction", "GT"])
            
            # Save the metrics dict
            tr_err, rot_err, errors = compute_kitti_metrics(absolute_pred, absolute_gt)
            if tr_err and rot_err:
                ate, std_ate = compute_ate(relative_prediction, relative_ground_truth)
                are, std_are = compute_are(relative_prediction, relative_ground_truth)

                self.metrics[sequence_id]['0000_avg'] = {
                    "tr_err":   float(tr_err),
                    "rot_err":  float(rot_err),
                    "ATE":      float(ate),
                    "STD_ATE":  float(std_ate),
                    "ARE":      float(are),
                    "STD_ARE":  float(std_are),
                }

                self.metrics[sequence_id]['0000_errors'] = errors
                if elapsed is not None:
                    self.metrics[sequence_id]['0000_avg']["nsecs_per_frame"] = float(elapsed / absolute_gt.shape[0])
                self.save_metrics()

            if additional_metrics_filename is not None:
                self.save_metrics(additional_metrics_filename)

            # TODO Add Average translation error as simple metric over all sequences (to have one number)
            

    def __add_mean_metrics(self):
        avg_metrics = {
            "tr_err": 0.0,
            "rot_err": 0.0,
            "ATE": 0.0,
            "STD_ATE": 0.0,
            "ARE": 0.0,
            "STD_ARE": 0.0,
            "nsecs_per_frame": 0.0
        }
        count = 0
        for seq_id, metrics_dict in self.metrics.items():
            if seq_id != "0000_AVG":
                for key, metric in metrics_dict.items():
                    avg_metrics[key] += metric
                count += 1
        if count > 0:
            for key, metric in avg_metrics.items():
                avg_metrics[key] = metric / count
            self.metrics["0000_AVG"] = avg_metrics


    def save_metrics(self, filename: str = "metrics.yaml"):
        """
        Saves the metrics dictionary saved as a yaml file
        """
        assert_debug(self.log_dir_path.exists() and self.log_dir_path.is_dir())
        open_file_mode = "w"
        file_path: Path = self.log_dir_path / filename

        with open(str(file_path), open_file_mode) as metrics_file:
            yaml.safe_dump(self.metrics, metrics_file)

    def close(self):
        """
        Close the metrics file
        ### -- ME --
        * calculate mean metrics and save the metrics
        """
        self.__add_mean_metrics()
        self.save_metrics()

    def __del__(self):
        self.close()
