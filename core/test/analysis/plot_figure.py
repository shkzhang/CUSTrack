import copy
import cv2
import numpy as np
import tikzplotlib
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import os
import torch
import matplotlib.ticker as mticker
from sympy.core.random import random
import matplotlib.patches as patches


def get_tracker_display_name(tracker):
    if tracker['disp_name'] is None:
        if tracker['run_id'] is None:
            disp_name = '{}_{}'.format(tracker['name'], tracker['param'])
        else:
            disp_name = '{}_{}_{:03d}'.format(tracker['name'], tracker['param'],
                                              tracker['run_id'])
    else:
        disp_name = tracker['disp_name']

    return disp_name


def get_plot_draw_styles(type=None):
    if type == 'dist_xy':
        plot_draw_style=[
            {'color': (1.0, 0.5, 0.2), 'line_style': '-.'},
            {'color': (0.4, 0.7, 0.1), 'line_style': '--'},
            {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'}
            ]
    elif type == 'visual':
        plot_draw_style = [
            {'color': (0.0, 1.0, 0.0), 'line_style': '-'},
            {'color': (1.0, 0.0, 0.0), 'line_style': '-'},
            {'color': (1.0, 0.5, 0.2), 'line_style': '-.'},
            {'color': (0.0, 1.0, 1.0), 'line_style': '-.'},
            {'color': (0.6, 0.3, 0.9), 'line_style': '-.'}
        ]
    else:
        plot_draw_style = [{'color': (1.0, 0.0, 0.0), 'line_style': '-'},
                           {'color': (0.0, 1.0, 0.0), 'line_style': '--'},
                           {'color': (0.0, 0.0, 1.0), 'line_style': '--'},
                           {'color': (1.0, 0.0, 1.0), 'line_style': '--'},
                           {'color': (0.0, 0.0, 0.0), 'line_style': '--'},
                           {'color': (0.0, 1.0, 1.0), 'line_style': '--'},
                           {'color': (0.5, 0.5, 0.5), 'line_style': '-'},
                           {'color': (136.0 / 255.0, 0.0, 21.0 / 255.0), 'line_style': '-'},
                           {'color': (1.0, 127.0 / 255.0, 39.0 / 255.0), 'line_style': '-'},
                           {'color': (0.0, 162.0 / 255.0, 232.0 / 255.0), 'line_style': '-'},
                           {'color': (0.0, 0.5, 0.0), 'line_style': '-'},
                           {'color': (1.0, 0.5, 0.2), 'line_style': '-'},
                           {'color': (0.1, 0.4, 0.0), 'line_style': '-'},
                           {'color': (0.6, 0.3, 0.9), 'line_style': '-'},
                           {'color': (0.4, 0.7, 0.1), 'line_style': '-'},
                           {'color': (0.2, 0.1, 0.7), 'line_style': '-'},
                           {'color': (0.7, 0.6, 0.2), 'line_style': '-'}]

    return plot_draw_style


def plot_draw_save(y, x, scores, trackers, plot_draw_styles, result_plot_path, plot_opts):
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    # Plot settings
    font_size = plot_opts.get('font_size', 20)
    font_size_axis = plot_opts.get('font_size_axis', 20)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 20)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    ylabel = "%s" % (ylabel.replace('%', '\%'))
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = r"$\bf{%s}$" % (plot_opts['title'])

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots()

    index_sort = scores.argsort(descending=False)

    plotted_lines = []
    legend_text = []

    for id, id_sort in enumerate(index_sort):
        line = ax.plot(x.tolist(), y[id_sort, :].tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'])

        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, scores[id_sort]))

    try:
        # add bold to our method
        for i in range(1, 2):
            legend_text[-i] = r'\textbf{%s}' % (legend_text[-i])

        ax.legend(plotted_lines[::-1], legend_text[::-1], loc=legend_loc, fancybox=False, edgecolor='black',
                  fontsize=font_size_legend, framealpha=1.0)
    except:
        pass

    ax.set(xlabel=xlabel,
           ylabel=ylabel,
           xlim=xlim, ylim=ylim,
           title=title)

    ax.grid(True, linestyle='-.')
    fig.tight_layout()

    tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    plt.draw()

def plot_draw_dist_save(frames_xy,gt_xy, trackers, plot_draw_styles, result_plot_path, plot_opts,seq,show_select_frame=False):

    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['figure.figsize'] = (9, 6)
    # Plot settings
    font_size = plot_opts.get('font_size', 20)
    frames_range = plot_opts.get('frames_range', (0,40))
    font_size_axis = plot_opts.get('font_size_axis', 20)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 20)
    select_frames = plot_opts.get('select_frames', 2)
    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']


    title = r"$\bf{%s}$" % (plot_opts['title'])

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig = plt.figure()
    fig.tight_layout()
    if show_select_frame:
        gs = GridSpec(7,6)
    else:
        gs = GridSpec(4,6)

    # draw x error
    def draw_error(_ax,index='X'):
        index_name = index
        index = 1 if index=='Y' else 0

        index_sort = torch.arange(0,len(trackers))
        plotted_lines = []
        legend_text = []
        x = torch.arange(max(frames_range[0],0),min(gt_xy.shape[0],frames_range[1]))

        # No Track
        no_tracking_xy = gt_xy[0,:].unsqueeze(0).repeat(x.shape[0],1)
        y = no_tracking_xy[:,index]
        line = _ax.plot(x.tolist(), y.tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[0]['color'],
                       linestyle=plot_draw_styles[0]['line_style'])
        plotted_lines.append(line[0])
        disp_name = 'No Tracking'
        legend_text.append('{}'.format(disp_name))

        # GT
        y = gt_xy[:,index]
        gt_y = torch.index_select(y, 0, x)
        line = _ax.plot(x.tolist(), gt_y.tolist(),
                       linewidth=line_width,
                       color=plot_draw_styles[1]['color'],
                       linestyle=plot_draw_styles[1]['line_style'])
        plotted_lines.append(line[0])
        disp_name = 'Ground Truth'
        legend_text.append('{}'.format(disp_name))

        # Selected
        select_frame_list = []
        if show_select_frame:
            pred_xy = frames_xy[:, x, :]
            anno_xy = gt_xy[x, :].unsqueeze(0).repeat(frames_xy.shape[0], 1,1)
            error =((pred_xy - anno_xy)**2).sum(2).sqrt() # (Tracker,Frames)
            error_index = torch.argsort(error[:,((no_tracking_xy - gt_xy[x, :])**2).sum(1).sqrt()>10], dim=1)+1
            select_frame_list = []
            for i in range(min(error_index.shape[1], select_frames)):
                select_index = error_index[0, i]  # (Tracker,1)
                select_frame_list.append(select_index)
                for id, id_sort in enumerate(index_sort):
                    _x = select_index
                    y = frames_xy[id, _x, index]
                    line = _ax.scatter(_x.tolist(), y.tolist(), s=40, marker='x',color='red', linewidths=2,zorder=4)
            plotted_lines.insert(0,line)
            legend_text.insert(0,'Selected Frames')



        for id, id_sort in enumerate(index_sort):
            y = frames_xy[id_sort, :,index]
            y = torch.index_select(y,0,x)
            line = _ax.plot(x.tolist(), y.tolist(),
                           linewidth=line_width,
                           color=plot_draw_styles[(id+2)%len(plot_draw_styles)]['color'],
                           linestyle=plot_draw_styles[(id+2)%len(plot_draw_styles)]['line_style'], marker = "o", mfc = "white", ms = 5)
            _ax.fill_between(x, y, gt_y,color='red',alpha = 0.1)
            plotted_lines.append(line[0])
            tracker = trackers[id_sort]
            disp_name = get_tracker_display_name(tracker)
            if id==0:
                disp_name = r'\textbf{%s}' % (disp_name)
            legend_text.append('{}'.format(disp_name))

        try:
            # add bold to our method
            if index==1:
                fig.legend(plotted_lines[::-1], legend_text[::-1],loc='upper right', bbox_to_anchor=(1,0.8), fancybox=False, edgecolor='black',
                          fontsize=font_size_legend, framealpha=1.0)
        except Exception as e:
            print(e)
        _ax.set(xlabel='Frames',
               ylabel=r'Distance %s (Pixel)'%(index_name),
               xlim=(torch.min(x),torch.max(x)), ylim=(max(torch.min(frames_xy[:,:,index]).item()-10,0),torch.max(frames_xy[:,:,index]).item()+10),
               title=title)

        _ax.grid(True, linestyle='-.')
        return select_frame_list
    ax = plt.subplot(gs[0:2, 0:])
    draw_error(ax,'X')
    ax = plt.subplot(gs[2:4, 0:])
    select_frame_list = draw_error(ax,'Y')

    # Draw frame image
    if show_select_frame:

        select_frame_list.sort()
        crop_size = 64
        plot_size = 512
        max_plot_size = plot_size//2
        max_crop_size=64
        scale = plot_size/crop_size

        plotted_lines = []
        legend_text = []
        ax = plt.subplot(gs[4:7, 0])
        anno_xy = gt_xy[0].to(int)
        no_track_xy = copy.deepcopy(anno_xy)

        frames_path = seq.frames[0]
        frames_image = cv2.imread(frames_path)
        frames_image = frames_image[anno_xy[1] - crop_size // 2:anno_xy[1] + crop_size // 2,
                       anno_xy[0] - crop_size // 2:anno_xy[0] + crop_size // 2]
        frames_image = cv2.resize(frames_image,(plot_size,plot_size))
        ax.imshow(frames_image)
        anno_xy = np.array(anno_xy)
        anno_xy = (anno_xy-(anno_xy-crop_size//2))*scale

        line = ax.scatter([anno_xy[1]], [anno_xy[0]], s=40, marker='x', color=plot_draw_styles[1]['color'], linewidths=2,
                          zorder=3)
        plotted_lines.append(line)
        legend_text.append('Ground Truth')
        plt.axis('off')
        ax.set_title('Initial frame')

        for i,select_frame in enumerate(select_frame_list):
            ax = plt.subplot(gs[4:7, i+1])
            frames_path = seq.frames[select_frame]
            pred_xy = frames_xy[0,select_frame].to(int).tolist()
            anno_xy = gt_xy[select_frame].to(int).tolist()
            frames_image = cv2.imread(frames_path)
            frames_image = frames_image[anno_xy[1]-crop_size//2:anno_xy[1]+crop_size//2,anno_xy[0]-crop_size//2:anno_xy[0]+crop_size//2]
            frames_image = cv2.resize(frames_image, (plot_size, plot_size))
            anno_xy = np.array(anno_xy)
            offset_xy = anno_xy-crop_size//2
            anno_xy = (anno_xy - offset_xy) * scale
            ax.imshow(frames_image)
            ax.scatter([anno_xy[1]],[anno_xy[0]], s=40, marker='x', color=plot_draw_styles[1]['color'], linewidths=2, zorder=3)

            _no_track_xy = (no_track_xy - offset_xy) * scale

            line = ax.scatter([_no_track_xy[1]], [_no_track_xy[0]], s=40,c='none', marker='o', edgecolors=plot_draw_styles[0]['color'],
                              linewidths=1, zorder=3)
            if 'No Tracking' not in legend_text:
                plotted_lines.insert(0,line)
                legend_text.insert(0,'No Tracking')

            pred_xy = np.array(pred_xy)
            pred_xy = (pred_xy - offset_xy) * scale
            line = ax.scatter([pred_xy[1]], [pred_xy[0]], s=40, marker='.', color=plot_draw_styles[2]['color'],
                              linewidths=2, zorder=3)
            if get_tracker_display_name(trackers[0]) not in legend_text:
                plotted_lines.append(line)
                legend_text.append(get_tracker_display_name(trackers[0]))

            plt.axis('off')
            ax.set_title('Frame {}'.format(select_frame))

        legend_text[-1] = r'\textbf{%s}' % (legend_text[-1])

        fig.legend(plotted_lines[::-1], legend_text[::-1],loc='lower right' ,fancybox=False, edgecolor='black',
                   fontsize=font_size_legend, framealpha=1.0)
        # cv2.circle(frames_image,frames_xy[0,0].to(int).tolist(),2,(255,0,0))
    fig.tight_layout()
    resources_dir = os.path.join(result_plot_path,'resources')
    os.makedirs(resources_dir,exist_ok=True)
    tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type),tex_relative_path_to_data='resources')
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    fig.savefig('{}/{}_plot.png'.format(result_plot_path, plot_type), dpi=300, format='png', transparent=False)

    plt.draw()

def plot_sequence_sr_draw_save(sr, trackers, plot_draw_styles, result_plot_path, plot_opts,seq_names):
    seq_names = [i.replace('_','-') for i in seq_names]
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['figure.figsize'] = (8, 8)
    # Plot settings
    font_size = plot_opts.get('font_size', 20)
    font_size_axis = plot_opts.get('font_size_axis', 20)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 20)

    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    xlabel = plot_opts['xlabel']
    ylabel = plot_opts['ylabel']
    ylabel = "%s" % (ylabel.replace('%', '\%'))
    xlim = plot_opts['xlim']
    ylim = plot_opts['ylim']

    title = r"$\bf{%s}$" % (plot_opts['title'])

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    index_sort = sr.mean(1).argsort(descending=False)

    plotted_lines = []
    legend_text = []
    angles = np.linspace(0, 2 * np.pi, len(seq_names), endpoint=False).tolist()
    angles += angles[:1]
    ax.set_theta_offset(np.pi+np.pi/3)
    ax.set_theta_direction(-1)
    y_ticks = [40,80,90,100]
    for id, id_sort in enumerate(index_sort):
        x = angles
        y = sr[id_sort,:].tolist()
        y = y+ y[:1]
        # ax.fill(x, y, color=plot_draw_styles[index_sort.numel() - id - 1]['color'], alpha=0.25)

        line = ax.plot(x, y,
                       linewidth=line_width,
                       color=plot_draw_styles[index_sort.numel() - id - 1]['color'],
                       linestyle=plot_draw_styles[index_sort.numel() - id - 1]['line_style'],marker='o')
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([])
        # ax.set_yticklabels([r'%.0f'%i for i in y_ticks])



        plotted_lines.append(line[0])

        tracker = trackers[id_sort]
        disp_name = get_tracker_display_name(tracker)

        legend_text.append('{} [{:.1f}]'.format(disp_name, sr.mean(1)[id_sort]))

    try:
        # add bold to our method
        for i in range(1, 2):
            legend_text[-i] = r'\textbf{%s}' % (legend_text[-i])

        ax.legend(plotted_lines[::-1], legend_text[::-1],loc='lower center',fancybox=False, ncol=2,frameon=False,
                  fontsize=font_size_legend, framealpha=1.0, bbox_to_anchor=(0.5, -0.4))
    except:
        pass
    ax.grid(True, linestyle='-.')

    ax.set(ylim=[min(y_ticks),100])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(seq_names)
    ax.xaxis.set_tick_params(pad=30)
    ax.set_frame_on(False)
    fig.tight_layout()

    # tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type))
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    fig.savefig('{}/{}_plot.png'.format(result_plot_path, plot_type), dpi=300, format='png', transparent=False)

    plt.draw()


def plot_visual_draw_save(pred_bbox,gt_bbox, trackers, plot_draw_styles, result_plot_path, plot_opts,dataset,visual_num=4,frame_num=4):
    plt.rcParams['text.usetex'] = True
    plt.rcParams["font.family"] = "Times New Roman"
    padding_size = 50
    plot_size = 256
    t_padding_size = 15
    t_plot_size = plot_size
    plt.rcParams['figure.figsize'] = ((plot_size*frame_num+t_plot_size)/100, (plot_size*visual_num+50)/100)
    # Plot settings
    font_size = plot_opts.get('font_size', 20)
    frames_range = plot_opts.get('frames_range', (0, 40))
    font_size_axis = plot_opts.get('font_size_axis', 20)
    line_width = plot_opts.get('line_width', 2)
    font_size_legend = plot_opts.get('font_size_legend', 20)
    select_frames = plot_opts.get('select_frames', 2)
    plot_type = plot_opts['plot_type']
    legend_loc = plot_opts['legend_loc']

    title = r"$\bf{%s}$" % (plot_opts['title'])

    matplotlib.rcParams.update({'font.size': font_size})
    matplotlib.rcParams.update({'axes.titlesize': font_size_axis})
    matplotlib.rcParams.update({'axes.titleweight': 'black'})
    matplotlib.rcParams.update({'axes.labelsize': font_size_axis})

    fig = plt.figure()
    fig.tight_layout()
    visual_seq = np.random.choice(range(len(dataset)),min(visual_num,len(dataset)),replace=False)
    gs = GridSpec(len(visual_seq), frame_num*2+2)
    plotted_lines = []
    legend_text = []

    def transformer(image,_gt_bbox,_bbox=None,_padding_size=None,_plot_size=None):
        if _padding_size is None:
            _padding_size = padding_size
        if _plot_size is None:
            _plot_size  = plot_size
        _gt_bbox = _gt_bbox.numpy()
        center = _gt_bbox[:2] + 0.5 * (_gt_bbox[ 2:] - 1.0)
        crop_size = max(_padding_size+_gt_bbox[2],_padding_size+_gt_bbox[3])
        center = center.astype(int)
        crop_size = int(crop_size*2)
        if center[1] - crop_size//2<0:
            crop_size = center[1] - crop_size//2 + center[1]
        if center[0] - crop_size//2<0:
            crop_size = center[0] - crop_size//2 + center[0]
        image = image[center[1] - crop_size//2:center[1] + crop_size//2,
                       center[0] - crop_size//2:center[0]  + crop_size//2]
        image = cv2.resize(image, (_plot_size, _plot_size))
        _gt_bbox = np.array(_gt_bbox,dtype=int)
        offset = center[:2] - crop_size//2
        _gt_bbox[:2] = _gt_bbox[:2] - offset
        scale = _plot_size / (crop_size)

        _gt_bbox = _gt_bbox * scale
        if _bbox is not None:
            _bbox = _bbox.numpy()
            _bbox[:,:2] = _bbox[:,:2] - offset
            _bbox = _bbox * scale
            return image,_gt_bbox.astype(int).tolist(),_bbox.astype(int).tolist()

        return image,_gt_bbox.astype(int).tolist()

    def select_frames_from_dataset(seq_id,frame_num):
        gt_center = gt_bbox[seq_id][:,:2] + 0.5 * (gt_bbox[seq_id][:,2:] - 1.0) # Seq_len, 2
        pred_center = pred_bbox[seq_id][:,:,:2] + 0.5 * (pred_bbox[seq_id][:,:,2:] - 1.0) # Tracker_num,Seq_Len, 2
        error_center = pred_center - torch.unsqueeze(gt_center,dim=0).repeat(pred_center.shape[0],1,1) # Tracker_num,Seq_Len, 2
        error_center = (error_center ** 2).sum(2).sqrt()  # (Tracker,Frames)
        error_center_main = error_center[0].unsqueeze(0) # (1,Frames)
        error_dist = torch.abs(error_center_main - error_center) # (Tracker,Frames)
        error_dist_mean = error_dist.mean(0) # (1,Frames)
        return torch.argsort(error_dist_mean,descending=True)[0:frame_num]


    for visual_id,seq_id in enumerate(visual_seq):
        seq = dataset[int(seq_id)]

        ax = plt.subplot(gs[visual_id, 0:2])
        frames_path = seq.frames[0]

        anno_bbox = gt_bbox[seq_id][0,:].to(int)
        frames_image = cv2.imread(frames_path)
        # cv2.rectangle(frames_image,(anno_bbox.tolist()[0], anno_bbox.tolist()[1]),(anno_bbox.tolist()[0]+anno_bbox.tolist()[2], anno_bbox.tolist()[1]+anno_bbox.tolist()[3]),(0,255,0),1)
        frames_image,anno_bbox = transformer(frames_image,anno_bbox,_padding_size=t_padding_size, _plot_size=t_plot_size)
        # cv2.rectangle(frames_image,(anno_bbox[0], anno_bbox[1]),(anno_bbox[0]+anno_bbox[2], anno_bbox[1]+anno_bbox[3]),(0,255,0),1)
        ax.imshow(frames_image)
        rect = patches.Rectangle((anno_bbox[0], anno_bbox[1]), anno_bbox[2], anno_bbox[3], linewidth=2,edgecolor=plot_draw_styles[0]['color'],
                                         linestyle=plot_draw_styles[0]['line_style'], facecolor='none')

        p = ax.add_patch(rect)
        # if visual_id==0:
        #     ax.set_title(r'\textbf{Template}')




        for label in ax.get_yticklabels():
            label.set_visible(False)
        for label in ax.get_xticklabels():
            label.set_visible(False)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

        name = seq.name
        name = name.replace('_', '-')
        name = f'G{visual_id+1}'
        ax.set_ylabel(r'\textbf{%s}' % name, rotation=90, labelpad=20)

        if visual_id == 0:
            plotted_lines.append(p)
            legend_text.append('Ground Truth')
        frames_id = select_frames_from_dataset(int(seq_id),frame_num)


        for index,frame_id in enumerate(frames_id):
            ax = plt.subplot(gs[visual_id, index*2 + 2:index*2 + 4])
            # if visual_id == 0:
            #     ax.set_title(r'\textbf{Frame %s}' % (index+1))
            for label in ax.get_yticklabels():
                label.set_visible(False)
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            frames_path = seq.frames[frame_id]
            frames_image = cv2.imread(frames_path)
            anno_bbox = gt_bbox[seq_id][frame_id, :].to(int)
            _pred_bbox = pred_bbox[seq_id][:,frame_id,:].to(int)
            frames_image, anno_bbox,_pred_bbox = transformer(frames_image, anno_bbox,_pred_bbox)
            ax.imshow(frames_image)
            rect = patches.Rectangle((anno_bbox[0], anno_bbox[1]), anno_bbox[2], anno_bbox[3], linewidth=2,edgecolor=plot_draw_styles[0]['color'],
                                         linestyle=plot_draw_styles[0]['line_style'], facecolor='none')

            p = ax.add_patch(rect)
            for id,(track_name,track_pred) in enumerate(zip(trackers,_pred_bbox)):
                rect = patches.Rectangle((track_pred[0], track_pred[1]), track_pred[2], track_pred[3], linewidth=2,
                                         edgecolor=plot_draw_styles[(id+1) % len(plot_draw_styles)]['color'],
                                         linestyle=plot_draw_styles[(id+1) % len(plot_draw_styles)]['line_style'], facecolor='none')

                p = ax.add_patch(rect)
                track_name = get_tracker_display_name(track_name)
                if track_name not in legend_text:
                    plotted_lines.insert(0,p)
                    legend_text.insert(0,track_name)
            # plt.axis('off')

    # legend_text[-1] = r'\textbf{%s}' % (legend_text[-1])
    #
    fig.legend(plotted_lines[::-1], legend_text[::-1],loc='lower center', bbox_to_anchor=(0.5,0),fancybox=False,ncol=len(trackers)+1,
               fontsize=font_size_legend, framealpha=1.0)
        # cv2.circle(frames_image,frames_xy[0,0].to(int).tolist(),2,(255,0,0))

    fig.tight_layout()
    resources_dir = os.path.join(result_plot_path,'resources')
    os.makedirs(resources_dir,exist_ok=True)
    tikzplotlib.save('{}/{}_plot.tex'.format(result_plot_path, plot_type),tex_relative_path_to_data='resources')
    fig.savefig('{}/{}_plot.pdf'.format(result_plot_path, plot_type), dpi=300, format='pdf', transparent=True)
    fig.savefig('{}/{}_plot.png'.format(result_plot_path, plot_type), dpi=300, format='png', transparent=False)

    plt.draw()