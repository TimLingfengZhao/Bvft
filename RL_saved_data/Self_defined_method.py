from Policy_selection import *
class Bvft_poli(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        # resolution_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        resolution_list = [2,  4,  8,   12, 16, 19, 23]
        rmax, rmin = self.env.reward_range[0], self.env.reward_range[1]
        result_list = []
        Bvft_final_resolution_loss = []
        for i in range(len(self.FQE_saving_step_list) * 4):
            current_list = []
            Bvft_final_resolution_loss.append(current_list)
        group_list = []
        for resolution in resolution_list:
            record = BvftRecord()
            bvft_instance = BVFT(q_functions, self.test_data, self.gamma, rmax, rmin, policy_name_listi, record,
                                 "torch_actor_critic_cont", verbose=True, data_size=self.data_size,
                                 trajectory_num=self.trajectory_num)
            bvft_instance.run(resolution=resolution)
            group_list.append(record.group_counts[0])
            for i in range(len(record.losses[0])):
                Bvft_final_resolution_loss[i].append(record.losses[0][i])

            result_list.append(record.losses[0])
        min_loss_list = self.get_min_loss(result_list)
        less_index_list = self.rank_elements_lower_higher(min_loss_list)

        FQE_lr_list = [1e-4, 2e-5]
        FQE_hl_list = [[128, 256], [128, 1024]]
        line_name_list = []
        for i in range(len(self.FQE_saving_step_list)):
            for j in range(len(FQE_lr_list)):
                for k in range(len(FQE_hl_list)):
                    line_name_list.append('FQE_' + str(FQE_lr_list[j]) + '_' + str(FQE_hl_list[k]) + '_' + str(
                        self.FQE_saving_step_list[i]) + "step")
        Policy_ranking_saving_place = "Policy_ranking_saving_place"
        Policy_k_saving_place = "Policy_k_saving_place"
        Res_plot_folder = "Bvft_res_data_"+str(resolution_list)

        Policy_k_saving_path = os.path.join(Policy_ranking_saving_place,Policy_k_saving_place)
        Res_plot_saving_folder = os.path.join(Policy_k_saving_path,Res_plot_folder)
        save_folder_name = policy_name_listi
        if not os.path.exists(Res_plot_saving_folder):
            os.makedirs(Res_plot_saving_folder)
        policy_saving_path = os.path.join(Res_plot_saving_folder,policy_name_listi)
        if not os.path.exists(policy_saving_path):
            os.makedirs(policy_saving_path)
        self.save_as_pkl(policy_saving_path,Bvft_final_resolution_loss)
        self.save_as_txt(policy_saving_path,Bvft_final_resolution_loss)
        self.draw_Bvft_resolution_loss_graph(Bvft_final_resolution_loss, self.FQE_saving_step_list, resolution_list,
                                        save_folder_name, line_name_list, group_list)
        return less_index_list
class Bvft_zero(policy_select):
    def select_Q(self,q_functions,q_name_functions,policy_name_listi,q_sa,r_plus_vfsp):
        resolution_list = np.array([0.00001])
        rmax, rmin = self.env.reward_range[0], self.env.reward_range[1]
        result_list = []
        for resolution in resolution_list:
            record = BvftRecord()
            bvft_instance = BVFT(q_functions, self.test_data, self.gamma, rmax, rmin, policy_name_listi, record,
                                 "torch_actor_critic_cont", verbose=True, data_size=self.data_size,
                                 trajectory_num=self.trajectory_num)
            bvft_instance.run(resolution=resolution)
            result_list.append(record.losses[0])
        list_to_rank = result_list[0]
        less_index_list = self.rank_elements_lower_higher(list_to_rank)
        return less_index_list


class FQE_zero(policy_select):
    def select_Q(self,q_functions,q_name_functions,policy_name_listi,q_sa,r_plus_vfsp):
        loss = []
        for i in range(len(q_functions)):
            if i % 4 == 0:
                loss.append(self.load_FQE_performance(q_name_functions[i]))
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

class FQE_one(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        loss = []
        for i in range(len(q_functions)):
            if i % 4 == 1:
                loss.append(self.load_FQE_performance(q_name_functions[i]))
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

class FQE_two(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        loss = []
        for i in range(len(q_functions)):
            if i % 4 == 2:
                loss.append(self.load_FQE_performance(q_name_functions[i]))
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result

class FQE_three(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        loss = []
        for i in range(len(q_functions)):
            if i % 4 == 3:
                loss.append(self.load_FQE_performance(q_name_functions[i]))
            else:
                loss.append(-9999)
        result = self.rank_elements_larger_higher(loss)
        return result
class Bvft_abs(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        loss_function = []
        for i in range(len(q_functions)):
            diff = q_sa[i]-r_plus_vfsp[i]
            loss_function.append(np.abs(np.sum(diff)/len(diff)))
        less_index_list = self.rank_elements_lower_higher(loss_function)
        return less_index_list
class arg_i_max_j(policy_select):
    def select_Q(self, q_functions, q_name_functions, policy_name_listi, q_sa, r_plus_vfsp):
        max_j_list = []
        length = len(q_sa[0])
        for i in range(len(q_functions)):
            loss_list = []
            for j in range(len(q_functions)):
                inner_value = 0
                loss = np.abs(np.mean(q_sa[j]*(q_sa[i]-r_plus_vfsp[i])))
                print("loss : ",loss)
                loss_list.append(loss)
            max_j_list.append(max(loss_list))
        return self.rank_elements_lower_higher(max_j_list)