# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/8/2
# @Time        : 15:07
# @Description :
import os
import sys
import matplotlib.pyplot as plt

from core.utils.message import push

plt.rcParams['figure.figsize'] = [8, 8]

from core.test.analysis.plot_results import plot_results
from core.test.evaluation import get_dataset, trackerlist
def plot_error_line():
    pass



dataset_names = ['clust','ndth']
for dataset_name in dataset_names:

    dataset = get_dataset(dataset_name)

    # For distance curve

    trackers = []
    trackers.extend(trackerlist(name='custrack', parameter_name='base', dataset_name=dataset_name, display_name='CUSTrack'))
    plot_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('frames_dist'),force_evaluation=True)
    #
    # For sequence success rate (Causal Graph)
    # trackers = []
    # trackers.extend(trackerlist(name='custrack', parameter_name='base', dataset_name=dataset_name, display_name='Learnable'))
    #
    # trackers.extend(trackerlist(name='custrack', parameter_name='base_uniform', dataset_name=dataset_name, display_name='Uniform'))
    #
    # trackers.extend(trackerlist(name='custrack', parameter_name='base_random', dataset_name=dataset_name, display_name='Random'))
    # plot_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('sequence_sr'),force_evaluation=True)

    # For sequence success rate (Causal Graph)
    # trackers = []
    # trackers.extend(trackerlist(name='custrack', parameter_name='base', dataset_name=dataset_name, display_name='CUSTrack'))
    # trackers.extend(trackerlist(name='custrack', parameter_name='base_simple_all', dataset_name=dataset_name, display_name='V1'))
    #
    # trackers.extend(trackerlist(name='custrack', parameter_name='base_simple_ip', dataset_name=dataset_name, display_name='V3'))
    # #
    # trackers.extend(trackerlist(name='custrack', parameter_name='base_simple_sub', dataset_name=dataset_name, display_name='V2'))
    #
    #
    # plot_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('sequence_sr'),
    #              force_evaluation=True)

    # For visual
    # trackers = []

    # trackers.extend(trackerlist(name='custrack', parameter_name='base', dataset_name=dataset_name, display_name='CUSTrack'))
    # trackers.extend(
    #     trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_got10k_ep100', dataset_name=dataset_name, display_name='OSTrack', path_prefix='got10k'))
    # trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_b256_got', dataset_name=dataset_name, display_name='SeqTrack', path_prefix='got10k'))
    # trackers.extend(trackerlist(name='evptrack', parameter_name='EVPTrack-got-224_3', dataset_name=dataset_name, display_name='EVPTrack'))

    # plot_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('visual'),force_evaluation=True)