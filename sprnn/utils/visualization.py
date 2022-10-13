# ------------------------------------------------------------------------------
# @file:    visualizaton.py
# @brief:   This file contains the implementation of visualization utils
# ------------------------------------------------------------------------------
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import random
import torch

logger = logging.getLogger(__name__)

# NOTE: this code is a huge mess, inefficient and lots of repeated functions. 

def plot_trajectories(
    config, hist, fut, pred, seq_start_end, best_sample_idx, filename='val_', 
    **kwargs):
    """ Trajectory plot wrapper.
    Inputs:
    -------
    config[dict]: visualization configuration parameters
    hist[torch.tensor(hist_len, dim, batch_size)]: trajectories history
    fut[torch.tensor(fut_len, dim, batch_size)]: trajectories future
    pred[torch.tensor(num_samples, fut_len, dim, batch_size)]: trajectories predictions
    dataset_name[str]: name of trajectory dataset
    seq_start_end[int]: agents' sequences to plot
    filename[str]: filename
    """
    patterns = kwargs.get('patterns')
    for i, (s, e) in enumerate(seq_start_end):
        idx = best_sample_idx[s:e]
        fn = filename+ f"_seq-{i}"
        h, f, p_multi = hist[:, :, s:e], fut[:, :, s:e], pred[:, :, :, s:e]
        
        # TODO: fix this
        n = e - s
        p = np.empty_like(f)
        for j in range(n):
            p[:, :, j] = p_multi[best_sample_idx[j], :, :, j]
            
        plot_2d_trajectories(
            config=config, hist=h, fut=f, pred=p, max_agents=n, filename=fn)
        
        plot_2d_trajectories_mml(
            config=config, hist=h, fut=f, pred=p_multi, best_idx=idx, 
            max_agents=n, filename=fn)
        
        if isinstance(patterns, np.ndarray):
            pat = patterns[:, :, :, s:e]
            plot_patterns(
                config=config, hist=h, fut=f, pred=p, pat=pat, filename=fn)
            
        if config.animation:
            animate_2d_trajectories(
                config=config, hist=h, fut=f, pred=p, max_agents=n, filename=fn)
    
    # TODO: add plot patterns
    
    # if "bsk" in dataset_name:
    #     plot_trajectories_bsk(
    #         config=config, hist=hist, fut=fut, pred=pred, 
    #         max_agents=max_agents, filename=filename)
    # # traj-air plots
    # elif "days" in dataset_name:
    # elif "sdd" in dataset_name:
    #     plot2d_trajectories(
    #         config, hist, fut, pred, max_agents=max_agents, filename=filename)
    # else:
    #     logger.info(f"Dataset {dataset_name} is not supported!")
    
def plot_trajectories_animated(
    config, hist, fut, pred, seq_start_end, best_sample_idx, filename='val_', 
    **kwargs):
    """ Trajectory plot wrapper.
    Inputs:
    -------
    config[dict]: visualization configuration parameters
    hist[torch.tensor(hist_len, dim, batch_size)]: trajectories history
    fut[torch.tensor(fut_len, dim, batch_size)]: trajectories future
    pred[torch.tensor(num_samples, fut_len, dim, batch_size)]: trajectories predictions
    dataset_name[str]: name of trajectory dataset
    seq_start_end[int]: agents' sequences to plot
    filename[str]: filename
    """
    patterns = kwargs.get('patterns')
    for i, (s, e) in enumerate(seq_start_end):
        idx = best_sample_idx[s:e]
        fn = filename+ f"_seq-{i}"
        h, f, p_multi = hist[:, :, s:e], fut[:, :, s:e], pred[:, :, :, s:e]
        
        # TODO: fix this
        n = e - s
        p = np.empty_like(f)
        for j in range(n):
            p[:, :, j] = p_multi[best_sample_idx[j], :, :, j]
            
        # plot_2d_trajectories(
        #     config=config, hist=h, fut=f, pred=p, max_agents=n, filename=fn)
        
        # plot_3d_trajectories(
        #     config=config, hist=h, fut=f, pred=p, max_agents=n, filename=fn)
        
        plot_2d_trajectories_mml(
            config=config, hist=h, fut=f, pred=p_multi, best_idx=idx, 
            max_agents=n, filename=fn)
        
        # if isinstance(patterns, np.ndarray):
        #     pat = patterns[:, :, :, s:e]
        #     plot_patterns(
        #         config=config, hist=h, fut=f, pred=p, pat=pat, filename=fn)
            
        # if config.animation:
        #     animate_2d_trajectories(
        #         config=config, hist=h, fut=f, pred=p, max_agents=n, filename=fn)
            
        
def plot_2d_trajectories(
    config, hist, fut, pred, max_agents = 100, filename='val_', plot=False):
    """ Plots full trajectories from trajair dataset, i.e. hist + ground truth 
    and predictions.
    Inputs
    ------
    config[dict]: visualization configuration parameters
    hist[torch.tensor]: trajectories history
    fut[torch.tensor]: trajectories future
    pred[torch.tensor]: trajectories predictions
    max_agent[int]: max number of agents to plot
    filename[str]: filename
    """
    # set the limits 
    ax = plt.axes()
    if config.use_limits:
        ax = plt.axes(
            xlim=(config.x_lim[0], config.x_lim[1]),
            ylim=(config.y_lim[0], config.y_lim[1]))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)
    
    _, _, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    mhist, mfut, mpred, alpha = '--', '--', '-', 1.0
    lw = config.lw
    
    # plot agent trajectoreis (history and future)
    for agent in range(num_agents):
        # append legent to first agent
        if agent == 0:
            # 0.0 landmark
            plt.plot(
                [0], [0.1], 's', color='C'+str(num_agents % 10), label=config.center_label, 
                markersize=8, alpha=0.7)
            
            # history start landmark
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='History Start')
            ax.annotate(str(agent+1), xy=(x, y), fontsize=8, color='w' )
            
            # prediction start landmark
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='Prediction Start')
            
            # goal landmark
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='Goal')
            
            # history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Ground Truth Trajectory')
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Prediction')
        else:
            # a circle will denote the start location along with agent number
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            ax.annotate(str(agent+1), xy=(x, y), fontsize=8, color='w')
            
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            
            # a star will denote the goal 
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            
            # agent history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
    # add background
    if os.path.exists(config.background):
        background = plt.imread(config.background)
        plt.imshow(background, zorder=0,
            extent=[config.x_lim[0], config.x_lim[1] - config.diff,
                    config.x_lim[0], config.y_lim[1]], alpha=0.3)
    
    lgnd = plt.legend(
        fontsize=6, loc="upper center", ncol=3, labelspacing = 1, handletextpad=0.3)
    
    if plot:
        plt.show()
        plt.close()
    else:
        out_file = os.path.join(config.plot_path, f"{filename}.png")
        logger.debug(f"Saving plot to {out_file}")
        plt.savefig(
            out_file, format=config.format, dpi=config.dpi, bbox_inches='tight')
        plt.close()

def plot_3d_trajectories(
    config, hist, fut, pred, max_agents = 100, filename='val_', plot=False):
    """ Plots full trajectories from trajair dataset, i.e. hist + ground truth 
    and predictions.
    Inputs
    ------
    config[dict]: visualization configuration parameters
    hist[torch.tensor]: trajectories history
    fut[torch.tensor]: trajectories future
    pred[torch.tensor]: trajectories predictions
    max_agent[int]: max number of agents to plot
    filename[str]: filename
    """
    # set the limits 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # ax.set_axis_off()
    
    _, _, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    mhist, mfut, mpred, alpha = '--', '--', '-', 0.6
    lw = config.lw


    # ax.set_facecolor('green') 
    ax.grid(False) 
    # ax.w_xaxis.pane.fill = False
    # ax.w_yaxis.pane.fill = False
    # ax.w_zaxis.pane.fill = False
    
    # plot agent trajectoreis (history and future)
    for agent in range(num_agents):
        # append legent to first agent
        if agent == 0:
            # 0.0 landmark
            ax.plot3D(
                [0], [0.0], [0.0], 's', color='C'+str(num_agents % 10), label=config.center_label, 
                markersize=8, alpha=0.7)
            
            # history start landmark
            x, y, z = hist[0, 0, agent], hist[0, 1, agent], hist[0, 2, agent]
            ax.plot3D(x, y, z, '8', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='History Start')
            # ax.annotate(str(agent+1), xy=(x, y), fontsize=8, color='w' )
            
            # prediction start landmark
            x, y, z = hist[-1, 0, agent], hist[-1, 1, agent], hist[-1, 2, agent]
            ax.plot3D(x, y, z, 'P', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='Prediction Start')
            
            # goal landmark
            x, y, z = fut[-1, 0, agent], fut[-1, 1, agent], fut[-1, 2, agent]
            ax.plot3D(x, y, z, '*', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='Goal')
            
            # history
            hx, hy, hz = hist[:, 0, agent], hist[:, 1, agent], hist[:, 2, agent]
            ax.plot3D(hx, hy, hz, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Ground Truth Trajectory')
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            fz = np.append(hist[-1, 2, agent], fut[:, 2, agent])
            plt.plot(fx, fy, fz, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            pz = np.append(hist[-1, 2, agent], pred[:, 2, agent])
            plt.plot(px, py, pz, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=0.1, label='Prediction')  
        else:
            # a circle will denote the start location along with agent number
            x, y, z = hist[0, 0, agent], hist[0, 1, agent], hist[0, 2, agent]
            plt.plot(x, y, z, '8', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            # ax.annotate(str(agent+1), xy=(x, y), fontsize=8, color='w')
            
            x, y, z = hist[-1, 0, agent], hist[-1, 1, agent], hist[-1, 2, agent]
            plt.plot(x, y, z, 'P', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            
            # a star will denote the goal 
            x, y, z = fut[-1, 0, agent], fut[-1, 1, agent], fut[-1, 2, agent]
            plt.plot(x, y, z, '*', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            
            # agent history
            hx, hy, hz = hist[:, 0, agent], hist[:, 1, agent], hist[:, 2, agent]
            plt.plot(hx, hy, hz, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            fz = np.append(hist[-1, 2, agent], fut[:, 2, agent])
            plt.plot(fx, fy, fz, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            pz = np.append(hist[-1, 2, agent], pred[:, 2, agent])
            plt.plot(px, py, pz, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=0.1)
        
        init_view = -90
        init_elev = 80
        ax.view_init(elev=init_elev, azim=init_view)
        def update(i):
            # x, y, z = fut[i, 0], fut[i, 1], fut[i, 2]
            # ax.scatter(x, y, z, mfut, c='C'+str(agent % 10), alpha=1.0, s=1, )
            for j in range(num_agents):
                xx0, yy0, zz0 = pred[i, 0, j], pred[i, 1, j], pred[i, 2, j]
                ax.scatter(xx0, yy0, zz0, c='C'+str(j % 10), alpha=1.0, s=4)
            ax.view_init(elev=init_elev-2*i, azim=init_view+2*i)
                
            # xx1, yy1, zz1 = pred[i, 0, 1], pred[i, 1, 1], pred[i, 2, 1]
            # ax.scatter(xx1, yy1, zz1, c='C'+str(1 % 10), alpha=1.0, s=3)
            # xx2, yy2, zz2 = pred[i, 0, 1], pred[i, 1, 1], pred[i, 2, 1]
            # ax.scatter(xx1, yy1, zz1, c='C'+str(1 % 10), alpha=1.0, s=3)
            # xx1, yy1, zz1 = pred[i, 0, 1], pred[i, 1, 1], pred[i, 2, 1]
            # ax.scatter(xx1, yy1, zz1, c='C'+str(1 % 10), alpha=1.0, s=3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ani = animation.FuncAnimation(fig, update, frames=fut.shape[0], interval=200, blit=False, repeat=True)
        
    # add background
    # if os.path.exists(config.background):
    #     background = plt.imread(config.background)
    #     plt.imshow(background, zorder=0,
    #         extent=[config.x_lim[0], config.x_lim[1] - config.diff,
    #                 config.x_lim[0], config.y_lim[1]], alpha=0.3)
    
    lgnd = plt.legend(
        fontsize=6, loc="upper center", ncol=3, labelspacing = 1, handletextpad=0.3)
    
    if plot:
        plt.show()
        plt.close()
    else:
        out_file = os.path.join(config.plot_path, f"{filename}.gif")
        logger.debug(f"Saving plot to {out_file}")
        ani.save(out_file, writer='pillow')#, dpi=config.dpi, bbox_inches='tight')
        # plt.savefig(
        #     out_file, format=config.format, dpi=config.dpi, bbox_inches='tight')
        plt.close()
    # plt.show()   



def plot_patterns(
    config, hist, fut, pred, pat, filename='val_'):
    """ Plots full trajectories from trajair dataset, i.e. hist + ground truth 
    and predictions.
    Inputs
    ------
    config[dict]: visualization configuration parameters
    hist[torch.tensor]: trajectories history
    fut[torch.tensor]: trajectories future
    pred[torch.tensor]: trajectories predictions
    max_agent[int]: max number of agents to plot
    filename[str]: filename
    """
    # set the limits 
    
    _, _, num_agents = hist.shape  # n_channel is 2
    
    mtraj, mpred, alpha = '--', '-', 1.0
    lw = config.lw
    
    # will just randomly choose an agent 
    agent = np.random.choice(num_agents, 1)[0]
    hist, fut = hist[:, :, agent], fut[:, :, agent]
    
    # ground truth trajectory 
    traj = np.concatenate((hist, fut))
    
    ax = plt.axes()
    ax = plt.axes(
        xlim=(traj[:, 0].min()-5, traj[:, 0].max()+5),
        ylim=(traj[:, 1].min()-5, traj[:, 1].max()+5))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)
    
    pred, pat = pred[:, :, agent], pat[:, :, :, agent]
    
   
    
    # prediction start landmark
    plt.plot(hist[-1, 0], hist[-1, 1], 'P', markersize=8, color='C'+str(agent % 10), 
        alpha=alpha, label='Prediction Start')
    
    # goal landmark
    plt.plot(fut[-1, 0], fut[-1, 1], '*', markersize=10, color='C'+str(agent % 10), 
        alpha=alpha, label='Goal')
    
    plt.plot(traj[:, 0], traj[:, 1], mtraj, color='C'+str(agent % 10), 
        linewidth=lw, markersize=1, alpha=alpha, label='Ground Truth Trajectory')
    
    # predicted patterns
    x = np.append(pred[1, 0], pat[0, :, 0])
    y = np.append(pred[1, 1], pat[0, :, 1])
    plt.plot(x, y, mpred, color='C'+str((agent+2) % 10), linewidth=lw, 
        markersize=1, alpha=0.4, label='Patterns')
    for i in range(1, pat.shape[0]-1):
        x = np.append(pred[i+1, 0], pat[i, :, 0])
        y = np.append(pred[i+1, 1], pat[i, :, 1])
        
        plt.plot(x, y, mpred, color='C'+str((agent+2) % 10), linewidth=lw, 
            markersize=1, alpha=0.4)
        
    # predicted trajectory
    fx = np.append(hist[-1, 0], pred[:, 0])
    fy = np.append(hist[-1, 1], pred[:, 1])
    plt.plot(fx, fy, mpred, color='C'+str((agent+1) % 10), linewidth=lw+1, 
        markersize=1, alpha=alpha, label='Prediction')
    
    # add background
    if os.path.exists(config.background):
        background = plt.imread(config.background)
        plt.imshow(background, zorder=0,
            extent=[config.x_lim[0], config.x_lim[1] - config.diff,
                    config.x_lim[0], config.y_lim[1]], alpha=0.8)
    
    # history start landmark
    plt.plot(
        hist[0, 0], hist[0, 1], '8', markersize=10, color='C'+str(agent % 10), 
        alpha=alpha, label='History Start')
    
    lgnd = plt.legend(
        fontsize=6, loc="best", ncol=3, labelspacing = 1, handletextpad=0.3)

    out_file = os.path.join(config.plot_path, f"{filename}_patterns.png")
    logger.debug(f"Saving plot to {out_file}")
    plt.savefig(out_file, format=config.format, dpi=config.dpi, bbox_inches='tight')
    plt.close()


def plot_2d_trajectories_mml(
    config, hist, fut, pred, best_idx, max_agents = 100, filename='val_'):
    """ Plots full trajectories from trajair dataset, i.e. hist + ground truth 
    and predictions.
    Inputs
    ------
    config[dict]: visualization configuration parameters
    hist[torch.tensor]: trajectories history
    fut[torch.tensor]: trajectories future
    pred[torch.tensor]: trajectories predictions
    max_agent[int]: max number of agents to plot
    filename[str]: filename
    """
    # set the limits 
    ax = plt.axes()
    if config.use_limits:
        ax = plt.axes(
            xlim=(config.x_lim[0], config.x_lim[1]),
            ylim=(config.y_lim[0], config.y_lim[1]))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)
    
    _, _, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    mhist, mfut, mpred, alpha = '--', '--', '-', 1.0
    lw = config.lw
    
    # plot agent trajectoreis (history and future)
    for agent in range(num_agents):
        alpha = 1.0
        # append legent to first agent
        if agent == 0:
            # center landmark
            plt.plot(
                [0], [0.1], 's', color='C'+str(num_agents % 10), 
                label=config.center_label, markersize=8, alpha=0.9)
            
            # history start landmark
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='History Start')
            ax.annotate(str(agent+1), xy=(x, y), fontsize=9, color='k' )
            
            # prediction start landmark
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='Prediction Start')
            
            # goal landmark
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='Goal')
            
            # history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Ground Truth Trajectory')
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            for i in range(pred.shape[0]):
                if not i == best_idx[agent]:
                    alpha = 0.1
                    mpred = '-'
                    label = ''
                else:
                    alpha = 1.0
                    mpred = '-'
                    label = 'Prediction'
                px = np.append(hist[-1, 0, agent], pred[i, :, 0, agent])
                py = np.append(hist[-1, 1, agent], pred[i, :, 1, agent])
                plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                    markersize=1, alpha=alpha, label=label)

        else:
            # a circle will denote the start location along with agent number
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            ax.annotate(str(agent+1), xy=(x, y), fontsize=9, color='k')
            
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            
            # a star will denote the goal 
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            
            # agent history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (pred)
            for i in range(pred.shape[0]):
                if not i == best_idx[agent]:
                    alpha = 0.05
                    mpred = '-'
                else:
                    alpha = 1.0
                    mpred = '-'
                px = np.append(hist[-1, 0, agent], pred[i, :, 0, agent])
                py = np.append(hist[-1, 1, agent], pred[i, :, 1, agent])
                plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                    markersize=1, alpha=alpha)

    # add background
    if os.path.exists(config.background):
        background = plt.imread(config.background)
        plt.imshow(background, zorder=0,
            extent=[config.x_lim[0], config.x_lim[1] - config.diff,
                    config.x_lim[0], config.y_lim[1]], alpha=0.2)
    
    lgnd = plt.legend(
        fontsize=6, loc="upper center", ncol=3, labelspacing = 1, handletextpad=0.3)
 
    out_file = os.path.join(config.plot_path, f"{filename}_multi.png")
    logger.debug(f"Saving plot to {out_file}")
    plt.savefig(out_file, format=config.format, dpi=config.dpi, bbox_inches='tight')
    # plt.show()
    plt.close()

def animate_2d_trajectories(
    config, hist, fut, pred, max_agents = 100, filename='val_'):
    """ Plots full trajectories from trajair dataset, i.e. hist + ground truth 
    and predictions.
    Inputs
    ------
    config[dict]: visualization configuration parameters
    hist[torch.tensor]: trajectories history
    fut[torch.tensor]: trajectories future
    pred[torch.tensor]: trajectories predictions
    max_agent[int]: max number of agents to plot
    filename[str]: filename
    """
    # set the limits 
    ax = plt.axes(
        xlim=(config.x_lim[0], config.x_lim[1]),
        ylim=(config.y_lim[0], config.y_lim[1]))
    ax.axis('off')
    fig = plt.gcf()
    ax.grid(False)
    
    _, _, num_agents = hist.shape  # n_channel is 2
    num_agents = min(num_agents, max_agents)
    
    mhist, mfut, mpred, alpha = '--', '--', '-', 1.0
    lw = config.lw
    
    # plot agent trajectoreis (history and future)
    agent_trajs_x, agent_trajs_y = [], []
    for agent in range(num_agents):
        # append legent to first agent
        if agent == 0:
            # center landmark
            plt.plot(
                [0], [0.1], 's', color='C'+str(num_agents % 10), 
                label=config.center_label, markersize=8, alpha=0.7)
            
            # history start landmark
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='History Start')
            ax.annotate(str(agent+1), xy=(x, y), fontsize=10, color='w' )
            
            # prediction start landmark
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), 
                alpha=alpha, label='Prediction Start')
            
            # goal landmark
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), 
                alpha=alpha, label='Goal')
            
            # history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha, label='Ground Truth Trajectory')
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            agent_trajs_x.append(px)
            agent_trajs_y.append(py)
            plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=0.2, label='Prediction')
        else:
            # a circle will denote the start location along with agent number
            x, y = hist[0, 0, agent], hist[0, 1, agent]
            plt.plot(x, y, '8', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            ax.annotate(str(agent+1), xy=(x, y), fontsize=10, color='w')
            
            x, y = hist[-1, 0, agent], hist[-1, 1, agent]
            plt.plot(x, y, 'P', markersize=8, color='C'+str(agent % 10), alpha=alpha)
            
            # a star will denote the goal 
            x, y = fut[-1, 0, agent], fut[-1, 1, agent]
            plt.plot(x, y, '*', markersize=10, color='C'+str(agent % 10), alpha=alpha)
            
            # agent history
            hx, hy = hist[:, 0, agent], hist[:, 1, agent]
            plt.plot(hx, hy, mhist, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            # agent future (ground truth)
            fx = np.append(hist[-1, 0, agent], fut[:, 0, agent])
            fy = np.append(hist[-1, 1, agent], fut[:, 1, agent])
            plt.plot(fx, fy, mfut, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=alpha)
            
            px = np.append(hist[-1, 0, agent], pred[:, 0, agent])
            py = np.append(hist[-1, 1, agent], pred[:, 1, agent])
            agent_trajs_x.append(px)
            agent_trajs_y.append(py)
            plt.plot(px, py, mpred, color='C'+str(agent % 10), linewidth=lw, 
                markersize=1, alpha=0.2)
    
    plt.legend(fontsize=6, loc="upper center", ncol=3, labelspacing = 1, 
        handletextpad=0.3)
    
    # add background
    if os.path.exists(config.background):
        background = plt.imread(config.background)
        plt.imshow(background, zorder=0,
            extent=[config.x_lim[0], config.x_lim[1] - config.diff,
                    config.x_lim[0], config.y_lim[1]], alpha=0.8)
    
    # animate predictions 
    agents_x = [[] for _ in range(num_agents)]
    agents_y = [[] for _ in range(num_agents)]
        
    def animate(i):
        for agent in range(num_agents):
            agents_x[agent].append(agent_trajs_x[agent][i])
            agents_y[agent].append(agent_trajs_y[agent][i])
            plt.plot(
                agents_x[agent], agents_y[agent], mpred, color='C'+str(agent % 10), 
                linewidth=lw, markersize=1)
    
    anim = animation.FuncAnimation(
        fig, animate, frames=agent_trajs_y[0].shape[0], interval=300)
    
    out_file = os.path.join(config.video_path, f"{filename}.gif")
    logger.debug(f"Saving animation to {out_file}")
    anim.save(out_file, dpi=config.dpi)