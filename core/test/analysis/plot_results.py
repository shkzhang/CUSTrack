import os
import torch
import pickle
import json
import matplotlib.pyplot as plt

from core.test.analysis.plot_figure import plot_draw_save, plot_draw_dist_save, get_plot_draw_styles, \
    get_tracker_display_name, plot_sequence_sr_draw_save, plot_visual_draw_save
from core.test.evaluation.environment import env_settings
from core.test.analysis.extract_results import extract_results




def check_eval_data_is_valid(eval_data, trackers, dataset):
    """ Checks if the pre-computed results are valid"""
    seq_names = [s.name for s in dataset]
    seq_names_saved = eval_data['sequences']

    tracker_names_f = [(t.name, t.parameter_name, t.run_id) for t in trackers]
    tracker_names_f_saved = [(t['name'], t['param'], t['run_id']) for t in eval_data['trackers']]

    return seq_names == seq_names_saved and tracker_names_f == tracker_names_f_saved


def merge_multiple_runs(eval_data):
    new_tracker_names = []
    ave_success_rate_plot_overlap_merged = []
    ave_success_rate_plot_center_merged = []
    ave_success_rate_plot_center_norm_merged = []
    avg_overlap_all_merged = []

    ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])
    ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
    ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all'])

    trackers = eval_data['trackers']
    merged = torch.zeros(len(trackers), dtype=torch.uint8)
    for i in range(len(trackers)):
        if merged[i]:
            continue
        base_tracker = trackers[i]
        new_tracker_names.append(base_tracker)

        match = [t['name'] == base_tracker['name'] and t['param'] == base_tracker['param'] for t in trackers]
        match = torch.tensor(match)

        ave_success_rate_plot_overlap_merged.append(ave_success_rate_plot_overlap[:, match, :].mean(1))
        ave_success_rate_plot_center_merged.append(ave_success_rate_plot_center[:, match, :].mean(1))
        ave_success_rate_plot_center_norm_merged.append(ave_success_rate_plot_center_norm[:, match, :].mean(1))
        avg_overlap_all_merged.append(avg_overlap_all[:, match].mean(1))

        merged[match] = 1

    ave_success_rate_plot_overlap_merged = torch.stack(ave_success_rate_plot_overlap_merged, dim=1)
    ave_success_rate_plot_center_merged = torch.stack(ave_success_rate_plot_center_merged, dim=1)
    ave_success_rate_plot_center_norm_merged = torch.stack(ave_success_rate_plot_center_norm_merged, dim=1)
    avg_overlap_all_merged = torch.stack(avg_overlap_all_merged, dim=1)

    eval_data['trackers'] = new_tracker_names
    eval_data['ave_success_rate_plot_overlap'] = ave_success_rate_plot_overlap_merged.tolist()
    eval_data['ave_success_rate_plot_center'] = ave_success_rate_plot_center_merged.tolist()
    eval_data['ave_success_rate_plot_center_norm'] = ave_success_rate_plot_center_norm_merged.tolist()
    eval_data['avg_overlap_all'] = avg_overlap_all_merged.tolist()

    return eval_data





def check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation=False, **kwargs):
    # Load data
    settings = env_settings()

    # Load pre-computed results
    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data_path = os.path.join(result_plot_path, 'eval_data.pkl')

    if os.path.isfile(eval_data_path) and not force_evaluation:
        with open(eval_data_path, 'rb') as fh:
            eval_data = pickle.load(fh)
    else:
        # print('Pre-computed evaluation data not found. Computing results!')
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)

    if not check_eval_data_is_valid(eval_data, trackers, dataset):
        # print('Pre-computed evaluation data invalid. Re-computing results!')
        eval_data = extract_results(trackers, dataset, report_name, **kwargs)
        # pass
    else:
        # Update display names
        tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                         for t in trackers]
        eval_data['trackers'] = tracker_names
    with open(eval_data_path, 'wb') as fh:
        pickle.dump(eval_data, fh)
    return eval_data


def get_auc_curve(ave_success_rate_plot_overlap, valid_sequence):
    ave_success_rate_plot_overlap = ave_success_rate_plot_overlap[valid_sequence, :, :]
    auc_curve = ave_success_rate_plot_overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)

    return auc_curve, auc


def get_prec_curve(ave_success_rate_plot_center, valid_sequence, thresh=20):
    ave_success_rate_plot_center = ave_success_rate_plot_center[valid_sequence, :, :]
    prec_curve = ave_success_rate_plot_center.mean(0) * 100.0
    prec_score = prec_curve[:, thresh]

    return prec_curve, prec_score


def plot_results(trackers, dataset, report_name, merge_results=False,
                 plot_types=('success'), force_evaluation=False, **kwargs):
    """
    Plot results for the given trackers

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success',
                    'prec' (precision), and 'norm_prec' (normalized precision)
    """
    # Load data

    settings = env_settings()

    plot_draw_styles = get_plot_draw_styles()

    # Load pre-computed results

    result_plot_path = os.path.join(settings.result_plot_path, report_name)
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, force_evaluation,return_frame_results=True, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']

    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print(
        '\nPlotting results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    print('\nGenerating plots for: {}'.format(report_name))

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])

        success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                             'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 88), 'title': 'Success'}
        plot_draw_save(auc_curve, threshold_set_overlap, auc, tracker_names, plot_draw_styles, result_plot_path,
                       success_plot_opts)

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        threshold_set_center = torch.tensor(eval_data['threshold_set_center'])

        precision_plot_opts = {'plot_type': 'precision', 'legend_loc': 'lower right',
                               'xlabel': 'Location error threshold [pixels]', 'ylabel': 'Distance Precision [%]',
                               'xlim': (0, 50), 'ylim': (0, 100), 'title': 'Precision plot'}
        plot_draw_save(prec_curve, threshold_set_center, prec_score, tracker_names, plot_draw_styles, result_plot_path,
                       precision_plot_opts)

    # ********************************  Norm Precision Plot **************************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        threshold_set_center_norm = torch.tensor(eval_data['threshold_set_center_norm'])

        norm_precision_plot_opts = {'plot_type': 'norm_precision', 'legend_loc': 'lower right',
                                    'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                                    'xlim': (0, 0.5), 'ylim': (0, 85), 'title': 'Normalized Precision'}
        plot_draw_save(prec_curve, threshold_set_center_norm, prec_score, tracker_names, plot_draw_styles,
                       result_plot_path,
                       norm_precision_plot_opts)
    if 'frames_dist' in plot_types:
        frames_dists = eval_data['frames_dist']
        gt_dists = eval_data['ground_truth_dist']
        for i, seq_name in enumerate(eval_data['sequences']):
            frames_xy = frames_dists[i]
            frames_xy = frames_xy[:,:, :2] + 0.5 * (frames_xy[:,:, 2:] - 1.0)
            gt_xy  = gt_dists[i]
            gt_xy = gt_xy[:, :2] + 0.5 * (gt_xy[:, 2:] - 1.0)
            norm_precision_plot_opts = {'plot_type': f'frames_dist_{seq_name}', 'legend_loc': 'lower right',
                                        'xlim': (0, 0.5), 'ylim': (0, 85), 'title': '','select_frames':5}
            os.makedirs(os.path.join(str(result_plot_path), 'frames-distance'), exist_ok=True)

            plot_draw_dist_save(frames_xy[0,:,:].unsqueeze(0), gt_xy,[tracker_names[0]], get_plot_draw_styles('dist_xy'),
                                os.path.join(str(result_plot_path),'frames-distance'),norm_precision_plot_opts,dataset[i])
    if 'sequence_sr' in plot_types:
        seq_scores = []
        seq_names = []
        for i, seq_name in enumerate(eval_data['sequences']):
            if not valid_sequence[i]: continue
            seq_names.append(seq_name)
            ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
            # Index out valid sequences
            prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center[i:i + 1, :, :],
                                                    valid_sequence[i:i + 1], 5)
            seq_scores.append(prec_score)
        seq_scores = torch.stack(seq_scores,dim=1)
        norm_precision_plot_opts = {'plot_type': 'sequence precision', 'legend_loc': 'lower right',
                                    'xlabel': 'Location error threshold', 'ylabel': 'Distance Precision [%]',
                                    'xlim': (0, 0.5), 'ylim': (0, 85), 'title': 'Normalized Precision'}
        plot_sequence_sr_draw_save(seq_scores, tracker_names, plot_draw_styles,
                       result_plot_path,
                       norm_precision_plot_opts,seq_names)
    if 'visual' in plot_types:
        frames_dists = eval_data['frames_dist']
        gt_dists = eval_data['ground_truth_dist']

        for i in range(20):
            try:
                norm_precision_plot_opts = {'plot_type': f'visual_{i}', 'legend_loc': 'lower right',
                                            'xlim': (0, 0.5), 'ylim': (0, 85), 'title': '', 'select_frames': 5}
                os.makedirs(os.path.join(str(result_plot_path),'visual'),exist_ok=True)
                plot_visual_draw_save(frames_dists, gt_dists,tracker_names, get_plot_draw_styles('visual'),
                                        os.path.join(str(result_plot_path),'visual'),norm_precision_plot_opts,dataset)
            except Exception as e:
                print(e)

def sort_result(row_labels, scores,sort_key='ACE'):
    values = scores[sort_key]

    sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)

    sorted_row_labels = [row_labels[i] for i in sorted_indices]
    sorted_scores = {k: [v[i] for i in sorted_indices] for k, v in scores.items()}

    return sorted_row_labels, sorted_scores

def generate_formatted_report(row_labels, scores, table_name='',sort_key=None):
    if sort_key is not None:
        row_labels,scores = sort_result(row_labels,scores,sort_key)
    name_width = max([len(d) for d in row_labels] + [len(table_name)]) + 5
    min_score_width = 10

    report_text = '\n{label: <{width}} |'.format(label=table_name, width=name_width)

    score_widths = [max(min_score_width, len(k) + 3) for k in scores.keys()]

    for s, s_w in zip(scores.keys(), score_widths):
        report_text = '{prev} {s: <{width}} |'.format(prev=report_text, s=s, width=s_w)

    report_text = '{prev}\n'.format(prev=report_text)

    for trk_id, d_name in enumerate(row_labels):
        # display name
        report_text = '{prev}{tracker: <{width}} |'.format(prev=report_text, tracker=d_name,
                                                           width=name_width)
        for (score_type, score_value), s_w in zip(scores.items(), score_widths):
            report_text = '{prev} {score: <{width}} |'.format(prev=report_text,
                                                              score='{:0.2f}'.format(score_value[trk_id].item()),
                                                              width=s_w)
        report_text = '{prev}\n'.format(prev=report_text)

    return report_text


def print_results(trackers, dataset, report_name, test_id=None, merge_results=False,
                  plot_types=('success'),print_group=False, print_seq=False,return_value=False,sort_key=None,skip_missing_tracker = False, **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    if test_id is not None:
        report_name = '{}_{:03d}'.format(report_name, test_id)

    available_trackers = []

    if skip_missing_tracker:
        for tracker in trackers:
            if os.path.exists(tracker.results_dir):
                available_trackers.append(tracker)
            else:
                pass
                # print(tracker.results_dir,'is not exists, skip it.')
        trackers = available_trackers
    assert len(trackers)>0, "Need at least one tracker"
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(),
                                                              valid_sequence.shape[0]))

    scores = {}
    sr_thresholds = [3, 5, 6,7,9,10]
    # ********************************  ACE Plot **************************************
    if 'ACE' in plot_types:
        ave_center_error = torch.tensor(eval_data['ave_center_error'])
        scores['ACE'] = ave_center_error.mean(0)

    if 'SR' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        for threshold in sr_thresholds:
            prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence, threshold)
            scores[f'SR{threshold}'] = prec_score

    if 'AO' in plot_types:
        ave_iou = torch.tensor(eval_data['ave_iou'])
        scores['AO'] = ave_iou.mean(0) * 100.0

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]

    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name,sort_key = sort_key)
    print(report_text)

    if print_seq:
        seq_scores = {}
        seq_names = []
        for i, seq_name in enumerate(eval_data['sequences']):
            if not valid_sequence[i]:continue
            seq_names.append(seq_name)
            if 'ACE' in plot_types:
                ave_center_error = torch.tensor(eval_data['ave_center_error'])
                if 'ACE' not in seq_scores:
                    seq_scores['ACE'] = []
                seq_scores['ACE'].append(ave_center_error[i, 0, :])
            if 'SR' in plot_types:
                ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])
                # Index out valid sequences
                for threshold in sr_thresholds:
                    prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center[i:i + 1, 0:1, :],
                                                            valid_sequence[i:i + 1], threshold)
                    if f'SR{threshold}' not in seq_scores:
                        seq_scores[f'SR{threshold}'] = []
                    seq_scores[f'SR{threshold}'].append(prec_score)

            if 'AO' in plot_types:
                ave_iou = torch.tensor(eval_data['ave_iou'])
                if 'AO' not in seq_scores:
                    seq_scores['AO'] = []
                seq_scores['AO'].append(ave_iou[i, 0, :] * 100.0)
            report_text = generate_formatted_report(seq_names, seq_scores, table_name=report_name,sort_key='ACE')
            print(report_text)

    if return_value:
        ave_center_error = torch.tensor(eval_data['ave_center_error'])

        return ave_center_error.mean(0)

def plot_got_success(trackers, report_name):
    """ Plot success plot for GOT-10k dataset using the json reports.
    Save the json reports from http://got-10k.aitestunion.com/leaderboard in the directory set to
    env_settings.got_reports_path

    The tracker name in the experiment file should be set to the name of the report file for that tracker,
    e.g. DiMP50_report_2019_09_02_15_44_25 if the report is name DiMP50_report_2019_09_02_15_44_25.json

    args:
        trackers - List of trackers to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
    """
    # Load data
    settings = env_settings()
    plot_draw_styles = get_plot_draw_styles()

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    auc_curve = torch.zeros((len(trackers), 101))
    scores = torch.zeros(len(trackers))

    # Load results
    tracker_names = []
    for trk_id, trk in enumerate(trackers):
        json_path = '{}/{}.json'.format(settings.got_reports_path, trk.name)

        if os.path.isfile(json_path):
            with open(json_path, 'r') as f:
                eval_data = json.load(f)
        else:
            raise Exception('Report not found {}'.format(json_path))

        if len(eval_data.keys()) > 1:
            raise Exception

        # First field is the tracker name. Index it out
        eval_data = eval_data[list(eval_data.keys())[0]]
        if 'succ_curve' in eval_data.keys():
            curve = eval_data['succ_curve']
            ao = eval_data['ao']
        elif 'overall' in eval_data.keys() and 'succ_curve' in eval_data['overall'].keys():
            curve = eval_data['overall']['succ_curve']
            ao = eval_data['overall']['ao']
        else:
            raise Exception('Invalid JSON file {}'.format(json_path))

        auc_curve[trk_id, :] = torch.tensor(curve) * 100.0
        scores[trk_id] = ao * 100.0

        tracker_names.append({'name': trk.name, 'param': trk.parameter_name, 'run_id': trk.run_id,
                              'disp_name': trk.display_name})

    threshold_set_overlap = torch.arange(0.0, 1.01, 0.01, dtype=torch.float64)

    success_plot_opts = {'plot_type': 'success', 'legend_loc': 'lower left', 'xlabel': 'Overlap threshold',
                         'ylabel': 'Overlap Precision [%]', 'xlim': (0, 1.0), 'ylim': (0, 100), 'title': 'Success plot'}
    plot_draw_save(auc_curve, threshold_set_overlap, scores, tracker_names, plot_draw_styles, result_plot_path,
                   success_plot_opts)
    plt.show()


def print_per_sequence_results(trackers, dataset, report_name, merge_results=False,
                               filter_criteria=None, **kwargs):
    """ Print per-sequence results for the given trackers. Additionally, the sequences to list can be filtered using
    the filter criteria.

    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        filter_criteria - Filter sequence results which are reported. Following modes are supported
                        None: No filtering. Display results for all sequences in dataset
                        'ao_min': Only display sequences for which the minimum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences where at least one tracker performs poorly.
                        'ao_max': Only display sequences for which the maximum average overlap (AO) score over the
                                  trackers is less than a threshold filter_criteria['threshold']. This mode can
                                  be used to select sequences all tracker performs poorly.
                        'delta_ao': Only display sequences for which the performance of different trackers vary by at
                                    least filter_criteria['threshold'] in average overlap (AO) score. This mode can
                                    be used to select sequences where the behaviour of the trackers greatly differ
                                    between each other.
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
    sequence_names = eval_data['sequences']
    avg_overlap_all = torch.tensor(eval_data['avg_overlap_all']) * 100.0

    # Filter sequences
    if filter_criteria is not None:
        if filter_criteria['mode'] == 'ao_min':
            min_ao = avg_overlap_all.min(dim=1)[0]
            valid_sequence = valid_sequence & (min_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'ao_max':
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & (max_ao < filter_criteria['threshold'])
        elif filter_criteria['mode'] == 'delta_ao':
            min_ao = avg_overlap_all.min(dim=1)[0]
            max_ao = avg_overlap_all.max(dim=1)[0]
            valid_sequence = valid_sequence & ((max_ao - min_ao) > filter_criteria['threshold'])
        else:
            raise Exception

    avg_overlap_all = avg_overlap_all[valid_sequence, :]
    sequence_names = [s + ' (ID={})'.format(i) for i, (s, v) in enumerate(zip(sequence_names, valid_sequence.tolist()))
                      if v]

    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]

    scores_per_tracker = {k: avg_overlap_all[:, i] for i, k in enumerate(tracker_disp_names)}
    report_text = generate_formatted_report(sequence_names, scores_per_tracker)

    print(report_text)


def print_results_per_video(trackers, dataset, report_name, merge_results=False,
                            plot_types=('success'), per_video=False, **kwargs):
    """ Print the results for the given trackers in a formatted table
    args:
        trackers - List of trackers to evaluate
        dataset - List of sequences to evaluate
        report_name - Name of the folder in env_settings.perm_mat_path where the computed results and plots are saved
        merge_results - If True, multiple random runs for a non-deterministic trackers are averaged
        plot_types - List of scores to display. Can contain 'success' (prints AUC, OP50, and OP75 scores),
                    'prec' (prints precision score), and 'norm_prec' (prints normalized precision score)
    """
    # Load pre-computed results
    eval_data = check_and_load_precomputed_results(trackers, dataset, report_name, **kwargs)

    # Merge results from multiple runs
    if merge_results:
        eval_data = merge_multiple_runs(eval_data)

    seq_lens = len(eval_data['sequences'])
    eval_datas = [{} for _ in range(seq_lens)]
    if per_video:
        for key, value in eval_data.items():
            if len(value) == seq_lens:
                for i in range(seq_lens):
                    eval_datas[i][key] = [value[i]]
            else:
                for i in range(seq_lens):
                    eval_datas[i][key] = value

    tracker_names = eval_data['trackers']
    valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

    print('\nReporting results over {} / {} sequences'.format(valid_sequence.long().sum().item(),
                                                              valid_sequence.shape[0]))

    scores = {}

    # ********************************  Success Plot **************************************
    if 'success' in plot_types:
        threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
        ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

        # Index out valid sequences
        auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
        scores['AUC'] = auc
        scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
        scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

    # ********************************  Precision Plot **************************************
    if 'prec' in plot_types:
        ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

        # Index out valid sequences
        prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
        scores['Precision'] = prec_score

    # ********************************  Norm Precision Plot *********************************
    if 'norm_prec' in plot_types:
        ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

        # Index out valid sequences
        norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
        scores['Norm Precision'] = norm_prec_score

    # Print
    tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
    report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
    print(report_text)

    if per_video:
        for i in range(seq_lens):
            eval_data = eval_datas[i]

            print('\n{} sequences'.format(eval_data['sequences'][0]))

            scores = {}
            valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)

            # ********************************  Success Plot **************************************
            if 'success' in plot_types:
                threshold_set_overlap = torch.tensor(eval_data['threshold_set_overlap'])
                ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

                # Index out valid sequences
                auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
                scores['AUC'] = auc
                scores['OP50'] = auc_curve[:, threshold_set_overlap == 0.50]
                scores['OP75'] = auc_curve[:, threshold_set_overlap == 0.75]

            # ********************************  Precision Plot **************************************
            if 'prec' in plot_types:
                ave_success_rate_plot_center = torch.tensor(eval_data['ave_success_rate_plot_center'])

                # Index out valid sequences
                prec_curve, prec_score = get_prec_curve(ave_success_rate_plot_center, valid_sequence)
                scores['Precision'] = prec_score

            # ********************************  Norm Precision Plot *********************************
            if 'norm_prec' in plot_types:
                ave_success_rate_plot_center_norm = torch.tensor(eval_data['ave_success_rate_plot_center_norm'])

                # Index out valid sequences
                norm_prec_curve, norm_prec_score = get_prec_curve(ave_success_rate_plot_center_norm, valid_sequence)
                scores['Norm Precision'] = norm_prec_score

            # Print
            tracker_disp_names = [get_tracker_display_name(trk) for trk in tracker_names]
            report_text = generate_formatted_report(tracker_disp_names, scores, table_name=report_name)
            print(report_text)
