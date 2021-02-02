import numpy as np
import matplotlib.pyplot as plt

plt1_legend_dict = {"td_agent": "approximate values learned by\n TD with neural network", 
                    "td_agent_5000_episodes": "approximate values learned by\n TD with neural network",
                    "td_agent_tilecoding": "approximate values learned by\n TD with tile-coding"}


plt2_legend_dict = {"td_agent": "TD with neural network", 
                    "td_agent_5000_episodes": "TD with neural network",
                    "td_agent_tilecoding": "TD with tile-coding"}


plt2_label_dict = {"td_agent": "RMSVE\n averaged\n over\n 20 runs", 
                   "td_agent_5000_episodes": "RMSVE\n averaged\n over\n 20 runs",
                   "td_agent_tilecoding": "RMSVE\n averaged\n over\n 20 runs"}


# Function to plot result
def plot_result(agent_parameters, directory):
    
    true_V = np.load('data/true_V.npy')

    for num_g in agent_parameters["num_groups"]:
        plt1_agent_sweeps = []
        plt2_agent_sweeps = []
        
        # two plots: learned state-value and learning curve (RMSVE)
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
            
        for step_size in agent_parameters["step_size"]:
            
            # plot1
            filename = 'V_TD_agent_agg_states_{}_step_size_{}'.format(num_g, step_size).replace('.','')
            current_agent_V = np.load('{}/{}.npy'.format(directory, filename))

            plt1_x_legend = range(1,len(current_agent_V[:]) + 1)
            graph_current_agent_V, = ax[0].plot(plt1_x_legend, current_agent_V[:], label="approximate values: state aggregation: {}, step-size: {}".format(num_g, step_size))
            plt1_agent_sweeps.append(graph_current_agent_V)
            
            # plot2
            filename = 'RMSVE_TD_agent_agg_states_{}_step_size_{}'.format(num_g, step_size).replace('.','')
            current_agent_RMSVE = np.load('{}/{}.npy'.format(directory, filename))

            plt2_x_legend = range(1,len(current_agent_RMSVE[:]) + 1)
            graph_current_agent_RMSVE, = ax[1].plot(plt2_x_legend, current_agent_RMSVE[:], label="approximate values: state aggregation: {}, step-size: {}".format(num_g, step_size))
            plt2_agent_sweeps.append(graph_current_agent_RMSVE)
            
          
        # plot1: 
        # add True V
        plt1_x_legend = range(1,len(true_V[:]) + 1)
        graph_true_V, = ax[0].plot(plt1_x_legend, true_V[:], label="$v_\pi$")
        
        ax[0].legend(handles=[*plt1_agent_sweeps, graph_true_V])
        
        ax[0].set_title("Learned State Value after 2000 episodes")
        ax[0].set_xlabel('State')
        ax[0].set_ylabel('Value\n scale', rotation=0, labelpad=15)

        plt1_xticks = [1, 100, 200, 300, 400, 500]#, 600, 700, 800, 900, 1000]
        plt1_yticks = [-1.0, 0.0, 1.0]
        ax[0].set_xticks(plt1_xticks)
        ax[0].set_xticklabels(plt1_xticks)
        ax[0].set_yticks(plt1_yticks)
        ax[0].set_yticklabels(plt1_yticks)
        
        
        # plot2:
        ax[1].legend(handles=plt2_agent_sweeps)
        
        ax[1].set_title("Learning Curve")
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('RMSVE\n averaged over 50 runs', rotation=0, labelpad=40)

        plt2_xticks = range(0, 210, 20) # [0, 10, 20, 30, 40, 50, 60, 70, 80]
        plt2_xticklabels = range(0, 2100, 200) # [0, 100, 200, 300, 400, 500, 600, 700, 800]
        plt2_yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        ax[1].set_xticks(plt2_xticks)
        ax[1].set_xticklabels(plt2_xticklabels)
        ax[1].set_yticks(plt2_yticks)
        ax[1].set_yticklabels(plt2_yticks)
        
        plt.tight_layout()
        plt.suptitle("{}-State Aggregation".format(num_g),fontsize=16, fontweight='bold', y=1.03)
        plt.show()      

# Function to plot result
def plot_result_arr(data_name_array):
    
    true_V = np.load('data/true_V.npy')

    plt1_agent_sweeps = []
    plt2_agent_sweeps = []
    
    # two plots: learned state-value and learning curve (RMSVE)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    
    for data_name in data_name_array:

        # plot1
        filename = 'V_{}'.format(data_name).replace('.','')
        current_agent_V = np.load('results/{}.npy'.format(filename))
        current_agent_V = current_agent_V[-1, :]


        plt1_x_legend = range(1,len(current_agent_V[:]) + 1)
        graph_current_agent_V, = ax[0].plot(plt1_x_legend, current_agent_V[:], label=plt1_legend_dict[data_name])
        plt1_agent_sweeps.append(graph_current_agent_V)
        
        # plot2
        filename = 'RMSVE_{}'.format(data_name).replace('.','')
        RMSVE_data = np.load('results/{}.npz'.format(filename))
        current_agent_RMSVE = np.mean(RMSVE_data["rmsve"], axis = 0)

        plt2_x_legend = np.arange(0, RMSVE_data["num_episodes"]+1, RMSVE_data["eval_freq"])
        graph_current_agent_RMSVE, = ax[1].plot(plt2_x_legend, current_agent_RMSVE[:], label=plt2_legend_dict[data_name])
        plt2_agent_sweeps.append(graph_current_agent_RMSVE)
                
          
    # plot1: 
    # add True V
    plt1_x_legend = range(1,len(true_V[:]) + 1)
    graph_true_V, = ax[0].plot(plt1_x_legend, true_V[:], label="$v_{\pi}$")
    
    ax[0].legend(handles=[*plt1_agent_sweeps, graph_true_V], fontsize = 13)
    
    ax[0].set_title("State Value", fontsize = 15)
    ax[0].set_xlabel('State', fontsize = 14)
    ax[0].set_ylabel('Value\n scale', rotation=0, labelpad=15, fontsize = 14)

    plt1_xticks = [1, 100, 200, 300, 400, 500]
    plt1_yticks = [-1.0, 0.0, 1.0]
    ax[0].set_xticks(plt1_xticks)
    ax[0].set_xticklabels(plt1_xticks, fontsize=13)
    ax[0].set_yticks(plt1_yticks)
    ax[0].set_yticklabels(plt1_yticks, fontsize=13)
    
    
    # plot2:
    ax[1].legend(handles=plt2_agent_sweeps, fontsize = 13)
    
    ax[1].set_title("Learning Curve", fontsize = 15)
    ax[1].set_xlabel('Episodes', fontsize = 14)
    ax[1].set_ylabel(plt2_label_dict[data_name_array[0]], rotation=0, labelpad=40, fontsize = 14)

    plt2_yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    ax[1].tick_params(axis="x", labelsize=13)
    ax[1].set_yticks(plt2_yticks)
    ax[1].set_yticklabels(plt2_yticks, fontsize = 13)

    plt.tight_layout()
    plt.show()   
    
# Function to plot result
def plot_result_ac(agent_parameters, directory):
    
    plt1_agent_sweeps = []
    plt2_agent_sweeps = []
    
    x_range = 20000
    plt_xticks = [0, 4999, 9999, 14999, 19999]
    plt_xlabels = [1, 5000, 10000, 15000, 20000]
    plt1_yticks = range(0, -6001, -2000)
    plt2_yticks = range(-3, 1, 1)
    
        
    # single plots: Exp Avg reward 
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,14))

    for num_tilings in agent_parameters["num_tilings"]:
        for num_tiles in agent_parameters["num_tiles"]:
            for actor_ss in agent_parameters["actor_step_size"]:
                for critic_ss in agent_parameters["critic_step_size"]:
                    for avg_reward_ss in agent_parameters["avg_reward_step_size"]:

                        load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_avg_reward_ss_{}'.format(num_tilings, num_tiles, actor_ss, critic_ss, avg_reward_ss)
                        
                        ### plot1
                        file_type1 = "total_return"
                        data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type1))

                        data_mean = np.mean(data, axis=0)
                        data_std_err = np.std(data, axis=0)/np.sqrt(len(data))

                        data_mean = data_mean[:x_range]
                        data_std_err = data_std_err[:x_range]

                        plt_x_legend = range(0,len(data_mean))[:x_range]

                        ax[0].fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha = 0.2)
                        graph_current_data, = ax[0].plot(plt_x_legend, data_mean, linewidth=1.0, label="actor_ss: {}/32, critic_ss: {}/32, avg reward step_size: {}".format(actor_ss, critic_ss, avg_reward_ss))
                        plt1_agent_sweeps.append(graph_current_data)

    
                        ### plot2
                        file_type2 = "exp_avg_reward"
                        data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type2))

                        data_mean = np.mean(data, axis=0)
                        data_std_err = np.std(data, axis=0)/np.sqrt(len(data))

                        data_mean = data_mean[:x_range]
                        data_std_err = data_std_err[:x_range]

                        plt_x_legend = range(1,len(data_mean) + 1)[:x_range]

                        ax[1].fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha = 0.2)
                        graph_current_data, = ax[1].plot(plt_x_legend, data_mean, linewidth=1.0, label="actor: {}/32, critic: {}/32, avg reward: {}".format(actor_ss, critic_ss, avg_reward_ss))
                        plt2_agent_sweeps.append(graph_current_data)

    # plot 1
    ax[0].legend(handles=[*plt1_agent_sweeps])
    ax[0].set_xticks(plt_xticks)
    ax[0].set_yticks(plt1_yticks)
    ax[0].set_xticklabels(plt_xlabels)
    ax[0].set_yticklabels(plt1_yticks)
                        
    ax[0].set_title("Return per Step")
    ax[0].set_xlabel('Training steps')
    ax[0].set_ylabel('Total Return', rotation=90)
    ax[0].set_xlim([0,20000])
    
    # plot 2
    ax[1].legend(handles=[*plt2_agent_sweeps])
    ax[1].set_xticks(plt_xticks)
    ax[1].set_yticks(plt2_yticks)

    ax[1].set_title("Exponential Average Reward per Step")
    ax[1].set_xlabel('Training steps')
    ax[1].set_ylabel('Exponential Average Reward', rotation=90)
    ax[1].set_xticklabels(plt_xlabels)
    ax[1].set_yticklabels(plt2_yticks)
    ax[1].set_xlim([0,20000])
    ax[1].set_ylim([-3, 0.16])

    plt.suptitle("Average Reward Softmax Actor-Critic ({} Runs)".format(len(data)),fontsize=16, fontweight='bold', y=1.03)
                    
    # ax[1].legend(handles=plt2_agent_sweeps)

    # ax[1].set_title("Softmax policy Actor-Critic: Average Reward per Step ({} Runs)".format(len(avg_reward)))
    # ax[1].set_xlabel('Training steps')
    # ax[1].set_ylabel('Average Reward', rotation=0, labelpad=40)
    # ax[1].set_xticklabels(plt_xticks)
    # ax[1].set_yticklabels(plt_yticks)
    # ax.axhline(y=0.1, linestyle='dashed', linewidth=1.0, color='black')

    plt.tight_layout()
    # plt.suptitle("{}-State Aggregation".format(num_agg_states),fontsize=16, fontweight='bold', y=1.03)
    # plt.suptitle("Average Reward ActorCritic",fontsize=16, fontweight='bold', y=1.03)
    plt.show()   
        
def plot_sweep_result_ac(directory):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,7))
    
    plt_agent_sweeps = []
    
    x_range = 20000
    plt_xticks = [0, 4999, 9999, 14999, 19999]
    plt_xlabels = [1, 5000, 10000, 15000, 20000]
    plt2_yticks = range(-3, 1, 1)

    top_results = [{"actor_ss": 0.25, "critic_ss": 2, "avg_reward_ss": 0.03125},
                  {"actor_ss": 0.25, "critic_ss": 2, "avg_reward_ss": 0.015625},
                  {"actor_ss": 0.5, "critic_ss": 2, "avg_reward_ss": 0.0625},
                  {"actor_ss": 1, "critic_ss": 2, "avg_reward_ss": 0.0625},
                  {"actor_ss": 0.25, "critic_ss": 1, "avg_reward_ss": 0.015625}]
    
    for setting in top_results:
        
        num_tilings = 32
        num_tiles = 8
        actor_ss = setting["actor_ss"]
        critic_ss = setting["critic_ss"]
        avg_reward_ss = setting["avg_reward_ss"]
        
        load_name = 'ActorCriticSoftmax_tilings_{}_tiledim_{}_actor_ss_{}_critic_ss_{}_avg_reward_ss_{}'.format(num_tilings, num_tiles, actor_ss, critic_ss, avg_reward_ss)

        file_type2 = "exp_avg_reward"
        data = np.load('{}/{}_{}.npy'.format(directory, load_name, file_type2))

        data_mean = np.mean(data, axis=0)
        data_std_err = np.std(data, axis=0)/np.sqrt(len(data))

        data_mean = data_mean[:x_range]
        data_std_err = data_std_err[:x_range]

        plt_x_legend = range(1,len(data_mean) + 1)[:x_range]

        ax.fill_between(plt_x_legend, data_mean - data_std_err, data_mean + data_std_err, alpha = 0.2)
        graph_current_data, = ax.plot(plt_x_legend, data_mean, linewidth=1.0, label="actor: {}/32, critic: {}/32, avg reward: {}".format(actor_ss, critic_ss, avg_reward_ss))
        plt_agent_sweeps.append(graph_current_data)
        
    ax.legend(handles=[*plt_agent_sweeps])
    ax.set_xticks(plt_xticks)
    ax.set_yticks(plt2_yticks)

    ax.set_title("Exponential Average Reward per Step ({} Runs)".format(len(data)))
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Exponential Average Reward', rotation=90)
    ax.set_xticklabels(plt_xlabels)
    ax.set_yticklabels(plt2_yticks)
    ax.set_xlim([0, 20000])
    ax.set_ylim([-3.5, 0.16])