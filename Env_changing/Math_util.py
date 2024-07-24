from Env_change_util import *
from BvftUtil import *
class Bvft_(Hopper_edi):
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99,line_name_list = [],res_plot_save_path= "e"):
        r_plus_vfsp = self.calculate_r_plus_vfsp(q_sa=q_list,q_prime=q_prime,dataset=dataset,gamma=gamma)

        data = CustomDataLoader(dataset)
        resolution_list = np.array([2, 4, 5,  7, 8,  10, 11, 12, 16, 19, 22,23])
        rmax, rmin = env.reward_range[0], env.reward_range[1]
        result_list = []
        # print(q_list[0])
        # print(len(q_list))
        # print(len(q_list[0]))
        # sys.exit()
        Bvft_final_resolution_loss = []
        for i in range(len(q_list) ):
            current_list = []
            Bvft_final_resolution_loss.append(current_list)
        group_list = []
        for resolution in resolution_list:
            record = BvftRecord()
            bvft_instance = BVFT_(q_list,r_plus_vfsp, data, gamma, rmax, rmin, policy_namei, record,
                                 "torch_actor_critic_cont", verbose=True, data_size=data.size,
                                 trajectory_num=len(dataset))
            bvft_instance.run(resolution=resolution)
            result_list.append(record.losses[0])
            group_list.append(record.group_counts[0])
            for i in range(len(record.losses[0])):
                Bvft_final_resolution_loss[i].append(record.losses[0][i])
        list_to_rank = result_list[0]
        less_index_list = self.rank_elements_lower_higher(list_to_rank)

        self.draw_Bvft_resolution_loss_graph( Bvft_final_resolution_loss,
                                            resolution_list,
                                            line_name_list, group_list, plot_saving_path=res_plot_save_path)
        return less_index_list
class Bvft_zero(Hopper_edi):
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99,line_name_list = [],res_plot_save_path= "e"):
        r_plus_vfsp = self.calculate_r_plus_vfsp(q_sa=q_list,q_prime=q_prime,dataset=dataset,gamma=gamma)

        data = CustomDataLoader(dataset)
        resolution_list = np.array([0.00001])
        rmax, rmin = env.reward_range[0], env.reward_range[1]
        result_list = []


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


class Env_zero(Hopper_edi):
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99,line_name_list = [],res_plot_save_path= "e"):
        loss = []
        for i in range(len(q_list)):
            if i % 4 == 0:
                loss.append(1)
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

class Env_one(Hopper_edi):
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99,line_name_list = [],res_plot_save_path= "e"):
        loss = []
        for i in range(len(q_list)):
            if i % 4 == 1:
                loss.append(1)
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

class Env_two(Hopper_edi):
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99,line_name_list = [],res_plot_save_path= "e"):
        loss = []
        for i in range(len(q_list)):
            if i % 4 == 2:
                loss.append(1)
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

# class FQE_three(Hopper_edi):
#     def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99):
#         loss = []
#         for i in range(len(q_list)):
#             if i % 4 == 3:
#                 loss.append(1)
#             else:
#                 loss.append(-9999)
#         result = self.rank_elements_larger_higher(loss)
#         return result
class Bvft_abs(Hopper_edi):
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99,line_name_list = [],res_plot_save_path= "e"):
        loss_function = []
        r_plus_vfsp = self.calculate_r_plus_vfsp(q_sa=q_list, q_prime=q_prime, dataset=dataset, gamma=gamma)
        for i in range(len(q_list)):
            diff = q_list[i]-r_plus_vfsp[i]
            loss_function.append(np.abs(np.sum(diff)/len(diff)))
        less_index_list = self.rank_elements_lower_higher(loss_function)
        return less_index_list
class arg_i_max_j(Hopper_edi):
    def select_Q(self,q_list,q_prime,policy_namei,dataset,env,gamma=0.99,line_name_list = [],res_plot_save_path= "e"):
        max_j_list = []
        r_plus_vfsp = self.calculate_r_plus_vfsp(q_sa=q_list, q_prime=q_prime, dataset=dataset, gamma=gamma)
        length = len(q_list[0])
        for i in range(len(q_list)):
            loss_list = []
            for j in range(len(q_list)):
                inner_value = 0
                loss = np.abs(np.mean(q_list[j]*(q_list[i]-r_plus_vfsp[i])))
                loss_list.append(loss)
            max_j_list.append(max(loss_list))
        return self.rank_elements_lower_higher(max_j_list)