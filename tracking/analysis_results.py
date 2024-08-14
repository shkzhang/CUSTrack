import argparse

import _init_paths
import matplotlib.pyplot as plt

from core.utils.message import push

plt.rcParams['figure.figsize'] = [8, 8]

from core.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from core.test.evaluation import get_dataset, trackerlist

trackers = []
def parse_args():
    """
    args for evaluation.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, help='yaml configure file name')

    args = parser.parse_args()

    return args
def main():
    """CUSTrack"""
    dataset_names = ['clust', 'ndth']

    for dataset_name in dataset_names:
        trackers = []

        trackers.extend(trackerlist(name='SiamFC', parameter_name=None, dataset_name=dataset_name, display_name='*SiamFC'))
        trackers.extend(trackerlist(name='SiamDW', parameter_name=None, dataset_name=dataset_name, display_name='*SiamDW'))

        trackers.extend(trackerlist(name='Ocean', parameter_name=None, dataset_name=dataset_name, display_name='*Ocean'))
        trackers.extend(trackerlist(name='transt', parameter_name='base', dataset_name=dataset_name, display_name='*TransT'))



        trackers.extend(trackerlist(name='stark', parameter_name='baseline_got10k_only', dataset_name=dataset_name, display_name='*Stark',path_prefix='got10k'))


        trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_got10k_ep100', dataset_name=dataset_name, display_name='*OSTrack',path_prefix='got10k'))
        trackers.extend(trackerlist(name='seqtrack', parameter_name='seqtrack_b256_got', dataset_name=dataset_name, display_name='SeqTrack',path_prefix='got10k'))


        trackers.extend(trackerlist(name='aqatrack', parameter_name='AQATrack-ep100-got-256', dataset_name=dataset_name, display_name='AQATrack'))

        trackers.extend(trackerlist(name='evptrack', parameter_name='EVPTrack-got-224_3', dataset_name=dataset_name, display_name='*EVPTrack'))
        trackers.extend(trackerlist(name='custrack', parameter_name='base', dataset_name=dataset_name, display_name='CUSTrack256'))
        dataset = get_dataset(dataset_name)
        print_results(trackers, dataset, dataset_name, merge_results=False, plot_types=('ACE', 'SR', 'AO'),
                      force_evaluation=True)

if __name__ == '__main__':
    main()


