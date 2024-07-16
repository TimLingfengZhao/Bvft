from Env_change_util import *

class Bvft_(Hopper_edi):
    def select_Q(self,q_list,r_plus_vfsp,policy_namei):
        resolution_list = np.array([2, 4, 5,  7, 8,  10, 11, 12, 16, 19, 22,23])
        rmax, rmin = self.env_list[self.true_env_num].reward_range[0], self.env_list[self.true_env_num].reward_range[1]
        result_list = []
        for resolution in resolution_list:
            record = BvftRecord()
            bvft_instance = BVFT_(q_list,r_plus_vfsp, self.data, self.gamma, rmax, rmin, policy_namei, record,
                                 "torch_actor_critic_cont", verbose=True, data_size=self.data.size,
                                 trajectory_num=self.trajectory_num)
            bvft_instance.run(resolution=resolution)
            result_list.append(record.losses[0])
        list_to_rank = result_list[0]
        less_index_list = self.rank_elements_lower_higher(list_to_rank)
        return less_index_list