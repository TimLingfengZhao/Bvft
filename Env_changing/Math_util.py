from Env_change_util import *

class Bvft_(Hopper_edi):
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99):
        r_plus_vfsp = self.calculate_r_plus_vfsp(q_sa=q_list,q_prime=q_prime,dataset=dataset,gamma=gamma)

        data = CustomDataLoader(dataset)
        resolution_list = np.array([2, 4, 5,  7, 8,  10, 11, 12, 16, 19, 22,23])
        rmax, rmin = env.reward_range[0], env.reward_range[1]
        result_list = []
        # print(q_list[0])
        # print(len(q_list))
        # print(len(q_list[0]))
        # sys.exit()
        for resolution in resolution_list:
            record = BvftRecord()
            bvft_instance = BVFT_(q_list,r_plus_vfsp, data, gamma, rmax, rmin, policy_namei, record,
                                 "torch_actor_critic_cont", verbose=True, data_size=data.size,
                                 trajectory_num=len(dataset))
            bvft_instance.run(resolution=resolution)
            result_list.append(record.losses[0])
        list_to_rank = result_list[0]
        less_index_list = self.rank_elements_lower_higher(list_to_rank)
        return less_index_list