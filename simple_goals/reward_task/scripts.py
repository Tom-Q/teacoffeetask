from reward_task import reward
import rdm
import utils
import tensorflow as tf


def performance_test():
    for i in range(10):
        print(i)
        model = reward.train_with_goals(nonlinearity=tf.nn.sigmoid, learning_rate=0.1)
        # utils.save_object("reward_task_with_goals", model)
        utils.save_object("reward_task_reproduction", model)
    cond = "none"
    nets = "reward_task_reproduction"
    # nets = "reward_task_with_4goals"
    num_nets = 15
    for type in [rdm.EUCLIDIAN]:
        cond = type
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0011" + cond, gain=[0, 0, 1, 1])  # , skips=[0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1100" + cond, gain=[1, 1, 0, 0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1100" + cond, gain=[1, 1, 1, 1])
        #reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_2200" + cond, gain=[2, 2, 0, 0])
        #reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0044" + cond, gain=[0, 0, 4, 4])
        #reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_4400" + cond, gain=[4, 4, 0, 0])
        """
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0000"+cond, gain=[0,0,0,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0011"+cond, gain=[0,0,1,1])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1100"+cond, gain=[1,1,0,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1122"+cond, gain=[1,1,2,2])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1110"+cond, gain=[1,1,1,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0001"+cond, gain=[0,0,0,1])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1112"+cond, gain=[1,1,1,2])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_5_5_52"+cond, gain=[.5,.5,.5,2])
        """

def goal_multiplier_tests():
    for i in range(5):
        print(i)
        model = reward.train_with_goals(nonlinearity=tf.nn.relu, learning_rate=0.02)
        # utils.save_object("reward_task_with_goals", model)
        utils.save_object("reward_task_with_goals_sigmoid", model)
    cond = "none"
    nets = "reward_task_with_goals_sigmoid"
    # nets = "reward_task_with_4goals"
    num_nets = 5
    for type in [rdm.SPEARMAN, rdm.EUCLIDIAN]:
        cond = type
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1111" + cond,
                                      gain=[1, 1, 1, 1])  # , skips=[0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0022" + cond, gain=[0, 0, 2, 2])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_2200" + cond, gain=[2, 2, 0, 0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0044" + cond, gain=[0, 0, 4, 4])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_4400" + cond, gain=[4, 4, 0, 0])
        """
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0000"+cond, gain=[0,0,0,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0011"+cond, gain=[0,0,1,1])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1100"+cond, gain=[1,1,0,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1122"+cond, gain=[1,1,2,2])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1110"+cond, gain=[1,1,1,0])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_0001"+cond, gain=[0,0,0,1])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_1112"+cond, gain=[1,1,1,2])
        reward.make_rdm_multiple_gain(nets, num_nets, rdm_type=type, save_name="reward_5_5_52"+cond, gain=[.5,.5,.5,2])
        """