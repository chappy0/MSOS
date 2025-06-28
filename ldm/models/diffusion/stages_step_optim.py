import json
import random
import torch
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint, differential_evolution
import logging
import time
import os # 确保 os 已导入

# --- 日志记录设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Robust NoiseScheduleVP Definition (与您提供的代码相同，保持不变) ---
def interpolate_fn_for_ns(x_query, x_pts, y_pts, device=torch.device('cpu'), dtype=torch.float32):
    x_query_orig_shape = x_query.shape
    x_query_flat = x_query.reshape(-1).to(device, dtype=dtype)
    x_pts_flat = x_pts.reshape(-1).to(device, dtype=dtype)
    y_pts_flat = y_pts.reshape(-1).to(device, dtype=dtype)

    indices = torch.searchsorted(x_pts_flat, x_query_flat)
    indices = torch.clamp(indices, 1, len(x_pts_flat) - 1)

    x_prev, x_next = x_pts_flat[indices - 1], x_pts_flat[indices]
    y_prev, y_next = y_pts_flat[indices - 1], y_pts_flat[indices]

    denom = x_next - x_prev
    weight = torch.where(denom.abs() < 1e-12, torch.zeros_like(denom), (x_query_flat - x_prev) / denom)
    interpolated_y_flat = y_prev + weight * (y_next - y_prev)
    
    try:
        if len(x_query_orig_shape) > 1 and interpolated_y_flat.numel() == x_query.numel():
             return interpolated_y_flat.reshape(x_query_orig_shape)
        return interpolated_y_flat.reshape(x_query_flat.shape[0], -1)
    except RuntimeError:
        logging.warning("Interpolate_fn_for_ns: Reshape failed, returning flat tensor.")
        return interpolated_y_flat

class NoiseScheduleVP:
    def __init__(self, schedule_name='discrete', betas=None, alphas_cumprod=None,
                 continuous_beta_0=0.1, continuous_beta_1=20., # For 'linear_vp' if used
                 num_trained_timesteps=1000, # Relevant for discrete if others not given
                 dtype=torch.float64, device=torch.device('cpu')): # Default to float64 for precision
        
        self.schedule_name = schedule_name
        self._device = device
        self.dtype = dtype
        self.num_trained_timesteps = int(num_trained_timesteps)

        if self.schedule_name == 'discrete':
            if alphas_cumprod is None:
                if betas is None:
                    logging.info("Discrete schedule: No betas/alphas_cumprod. Using default DDPM linear betas for 1000 steps.")
                    self.num_trained_timesteps = 1000 # Override if using default betas
                    betas = torch.linspace(1e-4, 0.02, self.num_trained_timesteps, dtype=self.dtype) # Typical DDPM
                if not isinstance(betas, torch.Tensor): betas = torch.tensor(betas, dtype=self.dtype)
                self.betas = betas.to(self._device)
                self.alphas = 1.0 - self.betas
                self.alphas_cumprod_model = torch.cumprod(self.alphas, dim=0)
            else:
                if not isinstance(alphas_cumprod, torch.Tensor): alphas_cumprod = torch.tensor(alphas_cumprod, dtype=self.dtype)
                self.alphas_cumprod_model = alphas_cumprod.to(self._device)
            
            if len(self.alphas_cumprod_model) != self.num_trained_timesteps:
                logging.warning(f"Discrete schedule: Length of alphas_cumprod ({len(self.alphas_cumprod_model)}) "
                                f"differs from num_trained_timesteps ({self.num_trained_timesteps}). "
                                f"Using length of alphas_cumprod as num_trained_timesteps.")
                self.num_trained_timesteps = len(self.alphas_cumprod_model)

            self.total_N = self.num_trained_timesteps
            self.T = 1.0 # Normalized max time
            self.min_t_schedule = 1.0 / self.num_trained_timesteps # Smallest discrete t
            
            self._t_discrete_map = (torch.arange(end=self.num_trained_timesteps, device=self._device, dtype=self.dtype) + 1.) / self.num_trained_timesteps
            self._log_alpha_discrete_map = 0.5 * torch.log(self.alphas_cumprod_model.clamp(min=1e-40)) # log(sqrt(bar_alpha_k))
            
            self._sorted_log_alphas_for_inverse, self._sort_indices_for_inverse = torch.sort(self._log_alpha_discrete_map)
            self._sorted_ts_for_inverse = self._t_discrete_map[self._sort_indices_for_inverse]

        elif self.schedule_name == 'linear_vp': # VP SDE linear beta schedule
            self.beta_0 = float(continuous_beta_0)
            self.beta_1 = float(continuous_beta_1)
            self.T = 1.0
            self.min_t_schedule = 1e-5 # Smaller epsilon for continuous VP
            self.total_N = self.num_trained_timesteps # Reference, not directly used in formulas
        else:
            raise ValueError(f"Unsupported schedule_name: {self.schedule_name}")
        logging.info(f"NoiseScheduleVP initialized: type={self.schedule_name}, T={self.T:.4f}, min_t={self.min_t_schedule:.4e}, N_model={self.num_trained_timesteps}")


    def to(self, device):
        self._device = device
        if self.schedule_name == 'discrete':
            if hasattr(self, 'alphas_cumprod_model'): self.alphas_cumprod_model = self.alphas_cumprod_model.to(device)
            if hasattr(self, '_log_alpha_discrete_map'): self._log_alpha_discrete_map = self._log_alpha_discrete_map.to(device)
            if hasattr(self, '_t_discrete_map'): self._t_discrete_map = self._t_discrete_map.to(device)
            if hasattr(self, '_sorted_log_alphas_for_inverse'): self._sorted_log_alphas_for_inverse = self._sorted_log_alphas_for_inverse.to(device)
            if hasattr(self, '_sorted_ts_for_inverse'): self._sorted_ts_for_inverse = self._sorted_ts_for_inverse.to(device)
        return self

    def marginal_log_mean_coeff(self, t): # This is log(alpha_t)
        t_tensor = torch.as_tensor(t, device=self._device, dtype=self.dtype).clamp(min=self.min_t_schedule, max=self.T)
        if self.schedule_name == 'discrete':
            return interpolate_fn_for_ns(t_tensor.reshape(-1, 1), 
                                         self._t_discrete_map.reshape(1, -1), 
                                         self._log_alpha_discrete_map.reshape(1, -1), 
                                         device=self._device, dtype=self.dtype).reshape(t_tensor.shape)
        elif self.schedule_name == 'linear_vp':
            return -0.25 * t_tensor**2 * (self.beta_1 - self.beta_0) - 0.5 * t_tensor * self.beta_0
        else: raise NotImplementedError(self.schedule_name)

    def marginal_alpha(self, t): return torch.exp(self.marginal_log_mean_coeff(t))
    
    def marginal_std(self, t):
        log_alpha_t_sq = 2. * self.marginal_log_mean_coeff(t)
        return torch.sqrt((1. - torch.exp(log_alpha_t_sq)).clamp(min=1e-12)) # Increased clamp from 1e-7

    def marginal_lambda(self, t): # log(alpha_t / sigma_t)
        log_alpha_t = self.marginal_log_mean_coeff(t)
        log_sigma_t = 0.5 * torch.log((1. - torch.exp(2. * log_alpha_t)).clamp(min=1e-40))
        return log_alpha_t - log_sigma_t

    def inverse_lambda(self, lamb): # Given lambda_t, find t
        lamb_tensor = torch.as_tensor(lamb, device=self._device, dtype=self.dtype)
        log_alpha_target = -0.5 * torch.logaddexp(torch.zeros_like(lamb_tensor), -2. * lamb_tensor)
        
        if self.schedule_name == 'discrete':
            return interpolate_fn_for_ns(log_alpha_target.reshape(-1,1), 
                                         self._sorted_log_alphas_for_inverse.reshape(1,-1), # x_pts (sorted log_alphas)
                                         self._sorted_ts_for_inverse.reshape(1,-1),         # y_pts (corresponding t's)
                                         device=self._device, dtype=self.dtype).reshape(lamb_tensor.shape)
        elif self.schedule_name == 'linear_vp':
            if abs(self.beta_1 - self.beta_0) < 1e-9 : # Special case beta1 = beta0
                if abs(self.beta_0) < 1e-9: return torch.full_like(lamb, self.T) # All betas zero
                t_candidate = -2 * log_alpha_target / self.beta_0
            else:
                tmp_for_inv = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb_tensor, torch.zeros_like(lamb_tensor))
                delta_sqrt_term_inv = torch.sqrt((self.beta_0**2 + tmp_for_inv).clamp(min=0))
                t_candidate = tmp_for_inv / (delta_sqrt_term_inv + self.beta_0) / (self.beta_1 - self.beta_0)
            return t_candidate.clamp(min=self.min_t_schedule, max=self.T)
        else:
            raise NotImplementedError(f"inverse_lambda for schedule {self.schedule_name}")

    def edm_sigma(self, t): # sigma_edm = sigma_vp / alpha_vp
        t_tensor = torch.as_tensor(t, device=self._device, dtype=self.dtype)
        return self.marginal_std(t_tensor) / self.marginal_alpha(t_tensor).clamp(min=1e-9)

    def inverse_edm_sigma(self, edm_s): # Given sigma_edm, find t
        edm_s_tensor = torch.as_tensor(edm_s, device=self._device, dtype=self.dtype).clamp(min=0)
        alpha_vp_target = 1.0 / torch.sqrt(edm_s_tensor**2 + 1.0)
        log_alpha_vp_target = torch.log(alpha_vp_target.clamp(min=1e-40))

        if self.schedule_name == 'discrete':
            return interpolate_fn_for_ns(log_alpha_vp_target.reshape(-1,1), 
                                         self._sorted_log_alphas_for_inverse.reshape(1,-1), 
                                         self._sorted_ts_for_inverse.reshape(1,-1), 
                                         device=self._device, dtype=self.dtype).reshape(edm_s_tensor.shape)
        elif self.schedule_name == 'linear_vp':
            # Use a robust bisection method for the continuous case
            t_low = torch.full_like(edm_s_tensor, self.min_t_schedule)
            t_high = torch.full_like(edm_s_tensor, self.T)
            for _ in range(30): # Bisection steps
                t_mid = (t_low + t_high) / 2.0
                log_alpha_mid = self.marginal_log_mean_coeff(t_mid)
                t_low = torch.where(log_alpha_mid > log_alpha_vp_target, t_mid, t_low)
                t_high = torch.where(log_alpha_mid <= log_alpha_vp_target, t_mid, t_high)
            res_t = (t_low + t_high) / 2.0
            return res_t.clamp(min=self.min_t_schedule, max=self.T)
        else:
            raise NotImplementedError(f"inverse_edm_sigma for schedule {self.schedule_name} not implemented.")

# --- StepOptim Class (与您提供的代码相同，保持不变) ---
class StepOptim(object):
    def __init__(self, ns: NoiseScheduleVP,
                 p_fixed_val=2.0,
                 objective_type='paper_eq35',
                 p_is_dynamic=False,
                 p_val_at_T=3.775, 
                 p_val_at_eps=2.824, 
                 p_dynamic_transform='sigmoid',
                 p_dynamic_sigmoid_k=2.908,
                 p_min_clip=0.5, p_max_clip=6.3,
                 power_q_val=1.6, 
                 smooth_q_power_base_eps=1e-9,
                 use_pf_inspired_error_metric=True,
                 use_rho_scaling: bool = False,
                 rho_q_is_dynamic: bool = False,
                 rho_q_base_val: float = 0.195,
                 rho_q_nfe_factor: float = -0.036,
                 rho_q_min_clip: float = 0.05,
                 rho_q_max_clip: float = 0.5
                 ):
        self.ns = ns 
        self.T_val = ns.T 
        self.min_t_val = ns.min_t_schedule 
        logging.debug(f"StepOptim initialized with T_val={self.T_val}, min_t_val={self.min_t_val}")

        self.p_fixed_val = float(p_fixed_val)
        self.p_is_dynamic = p_is_dynamic
        self.p_val_at_T = float(p_val_at_T)
        self.p_val_at_eps = float(p_val_at_eps)
        self.p_dynamic_transform = p_dynamic_transform
        self.p_dynamic_sigmoid_k = float(p_dynamic_sigmoid_k)
        self.p_min_clip = float(p_min_clip)
        self.p_max_clip = float(p_max_clip)
        self.power_q_val = float(power_q_val)
        self.smooth_q_power_base_eps = float(smooth_q_power_base_eps)
        self.objective_type = objective_type
        valid_objectives = ['paper_empirical', 'sum_of_squares', 'sum_of_squares_mse', 'paper_power_q', 'paper_eq35']
        if self.objective_type not in valid_objectives:
            raise ValueError(f"Unknown objective_type: {self.objective_type}. Valid: {valid_objectives}")
        self.use_pf_inspired_error_metric = use_pf_inspired_error_metric
        self.use_rho_scaling = use_rho_scaling
        self.rho_q_is_dynamic = rho_q_is_dynamic
        self.rho_q_base_val = float(rho_q_base_val)
        self.rho_q_nfe_factor = float(rho_q_nfe_factor)
        self.rho_q_min_clip = float(rho_q_min_clip)
        self.rho_q_max_clip = float(rho_q_max_clip)

    def _get_current_rho_q_val(self, N_intervals):
        if not self.rho_q_is_dynamic: return self.rho_q_base_val
        current_q = self.rho_q_base_val + (N_intervals * self.rho_q_nfe_factor)
        return np.clip(current_q, self.rho_q_min_clip, self.rho_q_max_clip)

    def _to_numpy(self, x_tensor): return x_tensor.cpu().detach().numpy()
    def alpha(self, t_val_np): return self._to_numpy(self.ns.marginal_alpha(t_val_np))
    def sigma(self, t_val_np): return self._to_numpy(self.ns.marginal_std(t_val_np))
    def lambda_func(self, t_val_np): return self._to_numpy(self.ns.marginal_lambda(t_val_np))
    def inverse_lambda(self, lamb_val_np): return self._to_numpy(self.ns.inverse_lambda(lamb_val_np))
    
    def H0(self, h): return np.expm1(h)
    def H1(self, h): h_exp = np.exp(h); return h_exp * h - (h_exp - 1)
    def H2(self, h): h_exp = np.exp(h); return h_exp * h**2 - 2 * self.H1(h)
    def H3(self, h): h_exp = np.exp(h); return h_exp * h**3 - 3 * self.H2(h)

    def _calculate_epsilon_tilde_vec(self, lambda_points_for_eval, N_intervals, current_lambda_min_for_p_norm, current_lambda_max_for_p_norm, actual_min_lambda_for_rho):
        t_points = self.inverse_lambda(lambda_points_for_eval)
        alpha_vals = self.alpha(t_points)
        sigma_vals = self.sigma(t_points)
        p_values_for_epsilon_tilde = None
        if self.p_is_dynamic:
            lambda_range_for_p = current_lambda_max_for_p_norm - current_lambda_min_for_p_norm
            if abs(lambda_range_for_p) < 1e-9: norm_lambdas_p = np.full_like(lambda_points_for_eval, 0.5)
            else: norm_lambdas_p = (lambda_points_for_eval - current_lambda_min_for_p_norm) / lambda_range_for_p
            norm_lambdas_p = np.clip(norm_lambdas_p, 0.0, 1.0)
            p_start_interp, p_end_interp = self.p_val_at_T, self.p_val_at_eps
            if current_lambda_min_for_p_norm > current_lambda_max_for_p_norm : p_start_interp, p_end_interp = self.p_val_at_eps, self.p_val_at_T
            if self.p_dynamic_transform == 'linear':
                p_vals = p_start_interp + (p_end_interp - p_start_interp) * norm_lambdas_p
            elif self.p_dynamic_transform == 'sigmoid':
                sigmoid_arg = self.p_dynamic_sigmoid_k * (2 * norm_lambdas_p - 1)
                weight_for_p_end = 1 / (1 + np.exp(-np.clip(sigmoid_arg, -700, 700)))
                p_vals = p_start_interp * (1 - weight_for_p_end) + p_end_interp * weight_for_p_end
            else:
                logging.warning(f"Unknown p_dynamic_transform '{self.p_dynamic_transform}', defaulting to linear.")
                p_vals = p_start_interp + (p_end_interp - p_start_interp) * norm_lambdas_p
            p_values_for_epsilon_tilde = np.clip(p_vals, self.p_min_clip, self.p_max_clip)
        else:
            p_values_for_epsilon_tilde = np.full_like(lambda_points_for_eval, self.p_fixed_val)
        epsilon_tilde_base = (sigma_vals**p_values_for_epsilon_tilde) / alpha_vals.clip(min=1e-9)
        if self.use_rho_scaling:
            current_rho_q = self._get_current_rho_q_val(N_intervals)
            if lambda_points_for_eval.size > 0:
                rho_exponent = current_rho_q * (lambda_points_for_eval - actual_min_lambda_for_rho)
                rho_vals = np.exp(np.clip(rho_exponent, -700, 700)); rho_vals = np.maximum(rho_vals, 1e-6)
                return rho_vals * epsilon_tilde_base
        return epsilon_tilde_base

    def _sel_lambdas_obj_calculator(self, lambda_vec_opt_part, N_intervals, eps_t_0_val, trunc_num_setting, use_pf_inspired_error_metric_calc_for_lte):
        # This function is the core objective calculator and remains unchanged in its logic.
        lambda_eps_val = self.lambda_func(eps_t_0_val).item()
        lambda_T_val = self.lambda_func(self.T_val).item()
        _lambdas_boundary = sorted([lambda_T_val, lambda_eps_val]) 
        actual_min_lambda_for_rho_calc = _lambdas_boundary[0]
        lambda_vec_opt_part_np = np.array(lambda_vec_opt_part, dtype=np.float64)
        temp_concat = np.concatenate(([_lambdas_boundary[0]], lambda_vec_opt_part_np, [_lambdas_boundary[1]]))
        lambda_vec_ext_np = np.sort(np.unique(temp_concat))
        if len(lambda_vec_ext_np) != N_intervals + 1:
            logging.debug(f"Length mismatch in _sel_lambdas_obj_calculator: len(unique lambda_vec_ext_np)={len(lambda_vec_ext_np)} vs N_intervals+1={N_intervals+1}. Opt_part: {lambda_vec_opt_part_np}, Boundaries: {_lambdas_boundary}")
            return 1e12 
        hv_np = np.diff(lambda_vec_ext_np); hv_np = np.maximum(hv_np, 1e-9) 
        elv_np = np.exp(lambda_vec_ext_np)
        current_lambda_min_for_p_norm_calc = lambda_vec_ext_np[0]
        current_lambda_max_for_p_norm_calc = lambda_vec_ext_np[-1]

        if self.objective_type == 'paper_eq35':
            aggregated_W = np.zeros(N_intervals)
            for s_loop_idx in range(N_intervals): 
                if N_intervals == 1: n_orig, kp_orig = 0, 1
                elif s_loop_idx == 0: n_orig, kp_orig = 0, 1
                elif s_loop_idx == N_intervals - 1 and N_intervals > 1: n_orig, kp_orig = s_loop_idx, 1
                elif s_loop_idx == 1 or (s_loop_idx == N_intervals - 2 and N_intervals > 2): n_orig, kp_orig = s_loop_idx - 1, 2
                else: n_orig, kp_orig = s_loop_idx - 2, 3
                kp_orig = min(kp_orig, 3, N_intervals - n_orig if N_intervals - n_orig > 0 else 1)
                if n_orig < 0: n_orig = 0
                if kp_orig == 1:
                    if n_orig + 1 >= len(elv_np): continue
                    J0 = elv_np[n_orig+1] - elv_np[n_orig]
                    if n_orig < N_intervals: aggregated_W[n_orig] += J0
                elif kp_orig == 2:
                    if n_orig+1 >= len(hv_np) or n_orig+2 >= len(elv_np): continue
                    h_n, h_np1 = hv_np[n_orig], hv_np[n_orig+1]; denom_J = h_n.clip(min=1e-12)
                    J0 = -elv_np[n_orig+2] * self.H1(h_np1) / denom_J
                    J1 =  elv_np[n_orig+2] * (self.H1(h_np1) + h_n * self.H0(h_np1)) / denom_J
                    if n_orig < N_intervals: aggregated_W[n_orig] += J0
                    if n_orig + 1 < N_intervals: aggregated_W[n_orig+1] += J1
                elif kp_orig == 3:
                    if n_orig+2 >= len(hv_np) or n_orig+3 >= len(elv_np): continue
                    h_n,h_np1,h_np2 = hv_np[n_orig],hv_np[n_orig+1],hv_np[n_orig+2]
                    d0=(h_n*(h_n+h_np1)).clip(min=1e-12); d1=(h_n*h_np1).clip(min=1e-12); d2=(h_np1*(h_n+h_np1)).clip(min=1e-12)
                    J0=elv_np[n_orig+3]*(self.H2(h_np2)+h_np1*self.H1(h_np2))/d0
                    J1=-elv_np[n_orig+3]*(self.H2(h_np2)+(h_n+h_np1)*self.H1(h_np2))/d1
                    J2=elv_np[n_orig+3]*(self.H2(h_np2)+(2*h_np1+h_n)*self.H1(h_np2)+h_np1*(h_n+h_np1)*self.H0(h_np2))/d2
                    if n_orig < N_intervals: aggregated_W[n_orig] += J0
                    if n_orig + 1 < N_intervals: aggregated_W[n_orig+1] += J1
                    if n_orig + 2 < N_intervals: aggregated_W[n_orig+2] += J2
            lambda_points_for_f_eval = lambda_vec_ext_np[:-1]

            if use_pf_inspired_error_metric_calc_for_lte:
                if len(hv_np) == len(lambda_points_for_f_eval):
                    lambda_points_for_f_eval = lambda_points_for_f_eval + hv_np / 2.0

            epsilon_tilde_values = self._calculate_epsilon_tilde_vec(lambda_points_for_f_eval, N_intervals, current_lambda_min_for_p_norm_calc, current_lambda_max_for_p_norm_calc, actual_min_lambda_for_rho_calc)
            if len(epsilon_tilde_values) != len(aggregated_W): 
                logging.error(f"Size mismatch for objective: eps_tilde ({len(epsilon_tilde_values)}) vs W ({len(aggregated_W)})")
                return 1e12
            return np.sum(epsilon_tilde_values * np.abs(aggregated_W))
        else: # LTE objectives (kept for completeness, though 'paper_eq35' is primary)
            lambda_for_eps_tilde_lte = lambda_vec_ext_np[:-1]
            lambdas_for_metric_eval_lte = lambda_for_eps_tilde_lte
            if use_pf_inspired_error_metric_calc_for_lte:
                if len(hv_np) == len(lambda_for_eps_tilde_lte): lambdas_for_metric_eval_lte = lambda_for_eps_tilde_lte + hv_np / 2.0
            data_err_vec_for_lte = self._calculate_epsilon_tilde_vec(lambdas_for_metric_eval_lte, N_intervals, current_lambda_min_for_p_norm_calc, current_lambda_max_for_p_norm_calc, actual_min_lambda_for_rho_calc)
            res_total = 0.0; c_vec_accum = np.zeros(len(hv_np))
            for s_loop_idx in range(N_intervals):
                if N_intervals == 1: n_orig, kp_orig = 0, 1
                elif s_loop_idx == 0: n_orig, kp_orig = 0, 1
                elif s_loop_idx == N_intervals - 1 and N_intervals > 1: n_orig, kp_orig = s_loop_idx, 1
                elif s_loop_idx == 1 or (s_loop_idx == N_intervals - 2 and N_intervals > 2): n_orig, kp_orig = s_loop_idx - 1, 2
                else: n_orig, kp_orig = s_loop_idx - 2, 3
                kp_orig = min(kp_orig, 3, N_intervals - n_orig if N_intervals - n_orig > 0 else 1)
                if n_orig < 0: n_orig = 0; 
                if n_orig >= N_intervals or n_orig + kp_orig > len(data_err_vec_for_lte): continue
                local_J_eps_terms_for_this_group = []
                if kp_orig == 1:
                    if n_orig + 1 >= len(elv_np) or n_orig >= len(data_err_vec_for_lte): continue
                    J0 = elv_np[n_orig+1] - elv_np[n_orig]; term0_val = data_err_vec_for_lte[n_orig] * J0
                    local_J_eps_terms_for_this_group.append(term0_val)
                    if s_loop_idx >= trunc_num_setting: c_vec_accum[n_orig] += term0_val
                elif kp_orig == 2:
                    if n_orig+1 >= len(hv_np) or n_orig+1 >= len(data_err_vec_for_lte) or n_orig+2 >= len(elv_np): continue
                    h_n, h_np1 = hv_np[n_orig], hv_np[n_orig+1]; denom_J = h_n.clip(min=1e-12)
                    J0 = -elv_np[n_orig+2] * self.H1(h_np1) / denom_J; J1 =  elv_np[n_orig+2] * (self.H1(h_np1) + h_n * self.H0(h_np1)) / denom_J
                    term0_val = data_err_vec_for_lte[n_orig] * J0; term1_val = data_err_vec_for_lte[n_orig+1] * J1
                    local_J_eps_terms_for_this_group.extend([term0_val, term1_val])
                    if s_loop_idx >= trunc_num_setting: c_vec_accum[n_orig] += term0_val; c_vec_accum[n_orig+1] += term1_val
                elif kp_orig == 3:
                    if n_orig+2 >= len(hv_np) or n_orig+2 >= len(data_err_vec_for_lte) or n_orig+3 >= len(elv_np): continue
                    h_n,h_np1,h_np2 = hv_np[n_orig],hv_np[n_orig+1],hv_np[n_orig+2]
                    d0=(h_n*(h_n+h_np1)).clip(min=1e-12); d1=(h_n*h_np1).clip(min=1e-12); d2=(h_np1*(h_n+h_np1)).clip(min=1e-12)
                    J0=elv_np[n_orig+3]*(self.H2(h_np2)+h_np1*self.H1(h_np2))/d0; J1=-elv_np[n_orig+3]*(self.H2(h_np2)+(h_n+h_np1)*self.H1(h_np2))/d1
                    J2=elv_np[n_orig+3]*(self.H2(h_np2)+(2*h_np1+h_n)*self.H1(h_np2)+h_np1*(h_n+h_np1)*self.H0(h_np2))/d2
                    t0=data_err_vec_for_lte[n_orig]*J0; t1=data_err_vec_for_lte[n_orig+1]*J1; t2=data_err_vec_for_lte[n_orig+2]*J2
                    local_J_eps_terms_for_this_group.extend([t0,t1,t2])
                    if s_loop_idx>=trunc_num_setting: c_vec_accum[n_orig]+=t0; c_vec_accum[n_orig+1]+=t1; c_vec_accum[n_orig+2]+=t2
                if s_loop_idx < trunc_num_setting:
                    if not local_J_eps_terms_for_this_group: continue
                    curr_sum_sq = np.sum(np.array(local_J_eps_terms_for_this_group)**2)
                    if self.objective_type=='paper_empirical': res_total+=np.sqrt(curr_sum_sq).clip(min=0)
                    elif self.objective_type=='paper_power_q':
                        l2=np.sqrt(curr_sum_sq).clip(min=0); res_total+=(l2+self.smooth_q_power_base_eps)**self.power_q_val - self.smooth_q_power_base_eps**self.power_q_val
                    elif self.objective_type in ['sum_of_squares', 'sum_of_squares_mse']: res_total+=curr_sum_sq
            valid_c_acc_indices = np.where(c_vec_accum != 0)[0]
            if len(valid_c_acc_indices) > 0:
                valid_c_acc = c_vec_accum[valid_c_acc_indices]
                if self.objective_type=='paper_empirical': res_total+=np.sum(np.abs(valid_c_acc))
                elif self.objective_type=='paper_power_q':
                    abs_c=np.abs(valid_c_acc); res_total+=np.sum((abs_c+self.smooth_q_power_base_eps)**self.power_q_val - self.smooth_q_power_base_eps**self.power_q_val)
                elif self.objective_type in ['sum_of_squares', 'sum_of_squares_mse']: res_total+=np.sum(valid_c_acc**2)
            if self.objective_type=='sum_of_squares_mse': return res_total/N_intervals if N_intervals>0 else res_total
            return res_total

    def get_ts_lambdas(self, N_intervals, eps_t_0_val=None, initType='edm', init_rho=7.0, trunc_num_setting_input=None):
        if N_intervals <= 0:
            raise ValueError("N_intervals must be positive.")
        
        if trunc_num_setting_input is None:
            if N_intervals <= 5: effective_trunc_num_setting = 0
            elif N_intervals <= 7: effective_trunc_num_setting = 3
            else: effective_trunc_num_setting = 0
        else:
            effective_trunc_num_setting = min(trunc_num_setting_input, N_intervals)

        if eps_t_0_val is None:
            eps_t_0_val = self.min_t_val
        
        lambda_eps_val = self.lambda_func(eps_t_0_val).item()
        lambda_T_val = self.lambda_func(self.T_val).item()
        lambda_bounds = sorted([lambda_T_val, lambda_eps_val])
        actual_min_lambda, actual_max_lambda = lambda_bounds[0], lambda_bounds[1]
        
        num_opt_vars = N_intervals - 1
        if num_opt_vars <= 0:
            t_res_np = np.array([self.T_val, eps_t_0_val], dtype=np.float64)
            lambda_res_ext_np = np.array(lambda_bounds, dtype=np.float64)
            return torch.from_numpy(t_res_np).to(self.ns.dtype), torch.from_numpy(lambda_res_ext_np).to(self.ns.dtype)

        # Stage 1: Generate Initial Schedule
        if initType.startswith('edm'):
            sigma_min_edm = self.ns.edm_sigma(torch.tensor(self.T_val)).cpu().numpy()
            sigma_max_edm = self.ns.edm_sigma(torch.tensor(eps_t_0_val)).cpu().numpy()
            inv_rho = 1.0 / init_rho
            steps_lin = np.linspace(0.0, 1.0, N_intervals + 1)
            sigma_schedule = (sigma_max_edm**inv_rho * (1 - steps_lin) + sigma_min_edm**inv_rho * steps_lin)**init_rho
            # Convert sigma_edm to lambda via t
            t_from_sigma_edm = self.ns.inverse_edm_sigma(torch.from_numpy(sigma_schedule.clip(min=1e-9)).to(self.ns._device, self.ns.dtype))
            lambda_full_init = np.sort(self.lambda_func(t_from_sigma_edm.cpu().numpy()))
        else: # Default to uniform in lambda
             lambda_full_init = np.linspace(actual_min_lambda, actual_max_lambda, N_intervals + 1)
        
        lambda_opt_vars_init = lambda_full_init[1:-1]
        
        # Stage 2: Optimize
        lambda_res_opt_part_np = lambda_opt_vars_init
        if not initType.endswith("_origin"):
            opt_result = minimize(
                self._sel_lambdas_obj_calculator,
                lambda_opt_vars_init,
                args=(
                    N_intervals,
                    eps_t_0_val,
                    effective_trunc_num_setting,
                    self.use_pf_inspired_error_metric
                ),
                method='trust-constr',
                options={'maxiter': 80, 'xtol': 1e-6, 'gtol': 1e-6, 'finite_diff_rel_step': 1e-7}
            )
            if opt_result.success:
                lambda_res_opt_part_np = opt_result.x
            else:
                logging.warning(f"Lambda optimization failed for {N_intervals} steps. Using initial values. Message: {opt_result.message}")

        # Stage 3: Return Final Schedule
        lambda_res_ext_np = np.sort(np.concatenate(([actual_min_lambda], lambda_res_opt_part_np, [actual_max_lambda])))
        t_res_np = np.sort(self.inverse_lambda(lambda_res_ext_np))[::-1] # Descending order
        return torch.from_numpy(t_res_np.copy()).to(self.ns.dtype), torch.from_numpy(lambda_res_ext_np.copy()).to(self.ns.dtype)


# --- 核心优化与搜索流程 ---

# --- UniPC 专用流程 ---
# def find_optimal_schedule(
#     nfe: int, 
#     noise_schedule: NoiseScheduleVP, 
#     initial_rho: float, 
#     initial_epsilon: float,
#     return_fitness: bool
# ):
#     """
#     为 UniPC 执行核心的“二次优化”流程。
#     目标是为 NFE 步的 UniPC 生成 NFE+1 点的 t 调度。
#     """
#     logging.debug(f"Executing UniPC inner optimization for NFE={nfe}, rho={initial_rho:.2f}, epsilon={initial_epsilon:.5f}")

#     step_optimizer = StepOptim(
#         ns=noise_schedule,
#         objective_type='paper_eq35',
#         p_fixed_val=2.0,
#         use_pf_inspired_error_metric=True
#     )

#     try:
#         # 步骤 1: 调用 get_ts_lambdas 进行二次优化 (NFE 步对应 N_intervals)
#         optimized_t_steps, optimized_lambda_steps = step_optimizer.get_ts_lambdas(
#             N_intervals=nfe,
#             initType='edm',
#             init_rho=initial_rho,
#             eps_t_0_val=initial_epsilon
#         )
#     except Exception as e:
#         logging.warning(f"UniPC Inner optimization failed for rho={initial_rho:.2f}, eps={initial_epsilon:.4f}. Reason: {e}")
#         return None, float('inf') if return_fitness else None

#     if not return_fitness:
#         return optimized_t_steps

#     # 步骤 2: 计算这个最终优化调度的理论误差作为“分数”
#     final_lambda_for_eval = optimized_lambda_steps.cpu().numpy()[1:-1]
    
#     try:
#         fitness_score = step_optimizer._sel_lambdas_obj_calculator(
#             lambda_vec_opt_part=final_lambda_for_eval,
#             N_intervals=nfe,
#             eps_t_0_val=initial_epsilon,
#             trunc_num_setting=0,
#             use_pf_inspired_error_metric_calc_for_lte=True
#         )
#         t_schedule_np = optimized_t_steps.cpu().numpy()
#         t_diffs = np.abs(np.diff(np.sort(t_schedule_np)))
#         nfe_low, dist_at_low = 4.0, 0.15; nfe_high, dist_at_high = 20.0, 0.01
#         slope = (dist_at_high - dist_at_low) / (nfe_high - nfe_low)
#         min_t_distance = np.clip(dist_at_low + slope * (nfe - nfe_low), dist_at_high, dist_at_low)
#         violations = min_t_distance - t_diffs
#         spacing_penalty = 1e9 * np.sum(np.maximum(0, violations)**2)
        
#         return optimized_t_steps, fitness_score + spacing_penalty
#     except Exception as e:
#         logging.error(f"Error calculating fitness for optimized UniPC schedule: {e}")
#         return None, float('inf')

# def fitness_function_for_unipc_es(params, nfe, noise_schedule):
#     """ UniPC 进化搜索的全局适应度函数 """
#     rho, epsilon = params[0], params[1]
#     _, fitness_score = find_optimal_schedule(
#         nfe=nfe, noise_schedule=noise_schedule, initial_rho=rho,
#         initial_epsilon=epsilon, return_fitness=True
#     )
#     return fitness_score


def find_optimal_schedule(
    nfe: int, 
    noise_schedule: NoiseScheduleVP, 
    initial_rho: float, 
    initial_epsilon: float,
    t_max: float, # <-- ADDED: The upper bound 'T' is now a parameter
    return_fitness: bool
):
    """
    Core "inner" optimization for UniPC, now accepting a dynamic upper bound T.
    """
    logging.debug(f"Executing UniPC inner optimization for NFE={nfe}, rho={initial_rho:.2f}, epsilon={initial_epsilon:.5f}, t_max={t_max:.3f}")

    # --- KEY CHANGE: Temporarily set the upper bound on the noise schedule object ---
    original_T = noise_schedule.T
    noise_schedule.T = t_max
    
    # This StepOptim instance will now use the new T_max
    step_optimizer = StepOptim(
        ns=noise_schedule,
        objective_type='paper_eq35',
        p_fixed_val=2.0,
        use_pf_inspired_error_metric=True
    )

    try:
        optimized_t_steps, optimized_lambda_steps = step_optimizer.get_ts_lambdas(
            N_intervals=nfe,
            initType='edm',
            init_rho=initial_rho,
            eps_t_0_val=initial_epsilon
        )
        
        if not return_fitness:
            noise_schedule.T = original_T # Restore original T value
            return optimized_t_steps

        # Calculate fitness using the found schedule
        final_lambda_for_eval = optimized_lambda_steps.cpu().numpy()[1:-1]
        fitness_score = step_optimizer._sel_lambdas_obj_calculator(
            lambda_vec_opt_part=final_lambda_for_eval,
            N_intervals=nfe,
            eps_t_0_val=initial_epsilon,
            trunc_num_setting=0,
            use_pf_inspired_error_metric_calc_for_lte=True
        )
        
        # ... (Spacing penalty logic remains the same) ...
        t_schedule_np = optimized_t_steps.cpu().numpy()
        t_diffs = np.abs(np.diff(np.sort(t_schedule_np)))
        nfe_low, dist_at_low = 4.0, 0.15; nfe_high, dist_at_high = 20.0, 0.01
        slope = (dist_at_high - dist_at_low) / (nfe_high - nfe_low)
        min_t_distance = np.clip(dist_at_low + slope * (nfe - nfe_low), dist_at_high, dist_at_low)
        violations = min_t_distance - t_diffs
        spacing_penalty = 1e9 * np.sum(np.maximum(0, violations)**2)

        noise_schedule.T = original_T # Restore original T value
        return optimized_t_steps, fitness_score + spacing_penalty

    except Exception as e:
        noise_schedule.T = original_T # Ensure T is restored even on error
        logging.warning(f"UniPC Inner optimization failed. Reason: {e}")
        return None, float('inf')


def fitness_function_for_unipc_es(params, nfe, noise_schedule):
    """ UniPC evolutionary search fitness function, now for 3 parameters. """
    # --- KEY CHANGE: Unpack 3 parameters ---
    rho, epsilon, t_max = params[0], params[1], params[2]
    
    _, fitness_score = find_optimal_schedule(
        nfe=nfe, noise_schedule=noise_schedule, initial_rho=rho,
        initial_epsilon=epsilon, 
        t_max=t_max, # Pass the new t_max to the inner loop
        return_fitness=True
    )
    return fitness_score

def run_unipc_search(nfe: int, noise_schedule: NoiseScheduleVP,seed):
    """ Runs the EA to find the best rho, epsilon, AND T_max for UniPC. """
    logging.info(f"===== Starting UniPC Evolutionary Search for NFE={nfe} =====")
    
    # --- KEY CHANGE: Add bounds for the new T_max parameter ---
    param_bounds = [
        (5.0, 7.0),      # Bounds for rho
        (0.01, 0.03),    # Bounds for epsilon
        (0.96, 1.0)      # Bounds for T_max (the upper bound)
    ]
    
    result = differential_evolution(
        fitness_function_for_unipc_es,
        bounds=param_bounds, args=(nfe, noise_schedule),
        # maxiter=60, popsize=20, disp=True, tol=0.01, workers=-1, updating='deferred',
        maxiter=60, popsize=20, disp=True, tol=0.01, workers=-1, updating='deferred',
        seed=seed

    )

    # --- KEY CHANGE: Unpack 3 results ---
    best_rho, best_epsilon, best_t_max = result.x
    
    logging.info(f"✅ UniPC Search Complete! Optimal parameters: rho={best_rho:.4f}, epsilon={best_epsilon:.5f}, T_max={best_t_max:.4f}")
    
    min_error_score = result.fun # Get the optimal fitness (error) score
    
    logging.info(f"✅ UniPC Search Complete! Optimal parameters: rho={best_rho:.4f}, epsilon={best_epsilon:.5f}, T_max={best_t_max:.4f}")
    # --- ADD THIS LINE ---
    logging.info(f"   📉 Minimum Theoretical Error Score Found: {min_error_score:.6e}") # Use scientific notation for clarity

    # --- MODIFY THIS LINE ---
    # Return all three found parameters AND the score
    return best_rho, best_epsilon, best_t_max, min_error_score
    # Return all three found parameters
    # return best_rho, best_epsilon, best_t_max



# --- (新增) DDIM 专用流程 ---
def find_optimal_schedule_ddim(
    nfe: int, 
    noise_schedule: NoiseScheduleVP, 
    initial_rho: float, 
    initial_epsilon: float,
    t_max: float,
    return_fitness: bool
):
    """
    为 DDIM 执行核心的“二次优化”流程。
    目标是为 NFE 步的 DDIM 直接生成 NFE 点的 t 调度 (对应 NFE-1 个优化区间)。
    """
    logging.debug(f"Executing DDIM inner optimization for NFE={nfe}, rho={initial_rho:.2f}, epsilon={initial_epsilon:.5f}")

    # original_T = noise_schedule.T
    # noise_schedule.T = t_max

    if nfe <= 1:
        # 无法为1步或更少步数进行优化，直接返回线性t调度
        final_t = np.linspace(noise_schedule.T, initial_epsilon, nfe, dtype=np.float64)
        final_t_tensor = torch.from_numpy(final_t).to(noise_schedule.dtype, noise_schedule._device)
        return final_t_tensor, 0 if return_fitness else final_t_tensor

    ddim_intervals = nfe - 1

    step_optimizer = StepOptim(
        ns=noise_schedule, objective_type='paper_eq35', p_fixed_val=2.0, use_pf_inspired_error_metric=True
    )

    try:
        # 步骤 1: 为 nfe-1 个区间调用优化器，以获得 nfe 个点
        optimized_t_steps, optimized_lambda_steps = step_optimizer.get_ts_lambdas(
            N_intervals=ddim_intervals, initType='edm', init_rho=initial_rho, eps_t_0_val=initial_epsilon
        )
    except Exception as e:
        logging.warning(f"DDIM Inner optimization failed for rho={initial_rho:.2f}, eps={initial_epsilon:.4f}. Reason: {e}")
        return None, float('inf') if return_fitness else None

    if not return_fitness:
        return optimized_t_steps

    # 步骤 2: 为得到的 NFE 点调度计算适应度分数
    final_lambda_for_eval = optimized_lambda_steps.cpu().numpy()[1:-1]
    
    try:
        fitness_score = step_optimizer._sel_lambdas_obj_calculator(
            lambda_vec_opt_part=final_lambda_for_eval, N_intervals=ddim_intervals, eps_t_0_val=initial_epsilon,
            trunc_num_setting=0, use_pf_inspired_error_metric_calc_for_lte=True
        )
        t_schedule_np = optimized_t_steps.cpu().numpy()
        t_diffs = np.abs(np.diff(np.sort(t_schedule_np)))
        nfe_low, dist_at_low = 4.0, 0.15; nfe_high, dist_at_high = 20.0, 0.01
        slope = (dist_at_high - dist_at_low) / (nfe_high - nfe_low)
        min_t_distance = np.clip(dist_at_low + slope * (nfe - nfe_low), dist_at_high, dist_at_low)
        violations = min_t_distance - t_diffs
        spacing_penalty = 1e9 * np.sum(np.maximum(0, violations)**2)
        
        return optimized_t_steps, fitness_score + spacing_penalty
    except Exception as e:
        logging.error(f"Error calculating fitness for optimized DDIM schedule: {e}")
        return None, float('inf')

def fitness_function_for_ddim_es(params, nfe, noise_schedule):
    """ DDIM 进化搜索的全局适应度函数 """
    rho, epsilon = params[0], params[1]
    _, fitness_score = find_optimal_schedule_ddim(
        nfe=nfe, noise_schedule=noise_schedule, initial_rho=rho,
        initial_epsilon=epsilon, return_fitness=True
    )
    return fitness_score

def run_ddim_search(nfe: int, noise_schedule: NoiseScheduleVP):
    """ 使用进化算法，为 DDIM 自动搜索最优的 rho 和 epsilon """
    logging.info(f"===== 开始为 NFE={nfe} 进行 DDIM 专属进化搜索 =====")
    param_bounds = [(5.0, 7.0), (0.01, 0.03)] # rho, epsilon 的搜索范围
    result = differential_evolution(
        fitness_function_for_ddim_es,
        bounds=param_bounds, args=(nfe, noise_schedule),
        maxiter=60, popsize=20, disp=True, tol=0.01, workers=1, updating='deferred'
    )
    best_rho, best_epsilon = result.x
    logging.info(f"✅ DDIM 搜索完成！最优参数: rho = {best_rho:.4f}, epsilon = {best_epsilon:.5f}")
    return best_rho, best_epsilon


def set_global_seed(seed_value='None'):
    """Sets the seed for all major sources of randomness."""
    if seed_value is None:
        pass
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # For full reproducibility, you might need these, but they can slow down training.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logging.info(f"Global random seed set to {seed_value}")
    
# --- 主程序入口 ---
if __name__ == '__main__':
    # 初始化噪声调度器
    # seed = 66
    seed = None
    # set_global_seed(66)
    model_total_timesteps = 1000
    betas_ddpm = torch.linspace(0.00085, 0.0120, model_total_timesteps, dtype=torch.float64) 
    alphas_cumprod_sd = torch.cumprod(1.0 - betas_ddpm, dim=0)
    ns_instance = NoiseScheduleVP(
        schedule_name='discrete', 
        alphas_cumprod=alphas_cumprod_sd, 
        dtype=torch.float64
    )
    
    # 设定我们想要求解的目标NFE
    NFE_TO_SOLVE = 3

    # --- 1. UniPC 调度优化 ---
    logging.info(f"\n{'='*25}\n===== 1. 开始优化 UniPC 调度 =====\n{'='*25}")
    best_rho_unipc, best_epsilon_unipc,best_t_max_unipc, error_score_unipc = run_unipc_search(nfe=NFE_TO_SOLVE, noise_schedule=ns_instance,seed=seed)
    
    final_schedule_unipc = find_optimal_schedule(
        nfe=NFE_TO_SOLVE,
        noise_schedule=ns_instance,
        initial_rho=best_rho_unipc,
        initial_epsilon=best_epsilon_unipc,
        t_max=best_t_max_unipc,
        return_fitness=False
    )
    
    logging.info(f"\n--- 最终结果 (UniPC) ---")
    logging.info(f"使用自动搜索到的 UniPC 最优参数 (rho={best_rho_unipc:.4f}, epsilon={best_epsilon_unipc:.5f})")
    logging.info(f"📈 对应的最小总误差分数为: {error_score_unipc:.6e}")
    if final_schedule_unipc is not None:
        logging.info(f"生成的最终优化调度 (UniPC, {len(final_schedule_unipc)} 点):")
        print(np.array2string(final_schedule_unipc.cpu().numpy(), formatter={'float_kind':lambda x: "%.8f" % x}))
    else:
        logging.error("未能成功生成最终 UniPC 调度。")

    # # --- 2. DDIM 调度优化 ---
    # logging.info(f"\n{'='*25}\n===== 2. 开始优化 DDIM 调度 =====\n{'='*25}")
    # best_rho_ddim, best_epsilon_ddim = run_ddim_search(nfe=NFE_TO_SOLVE, noise_schedule=ns_instance)

    # final_schedule_ddim_t = find_optimal_schedule_ddim(
    #     nfe=NFE_TO_SOLVE,
    #     noise_schedule=ns_instance,
    #     initial_rho=best_rho_ddim,
    #     initial_epsilon=best_epsilon_ddim,
    #     return_fitness=False
    # )

    # logging.info(f"\n--- 最终结果 (DDIM) ---")
    # logging.info(f"使用自动搜索到的 DDIM 最优参数 (rho={best_rho_ddim:.4f}, epsilon={best_epsilon_ddim:.5f})")
    # if final_schedule_ddim_t is not None:
    #     logging.info(f"生成的优化连续时间调度 (DDIM, {len(final_schedule_ddim_t)} 点):")
    #     print(np.array2string(final_schedule_ddim_t.cpu().numpy(), formatter={'float_kind':lambda x: "%.8f" % x}))
        
    #     n_model_steps = ns_instance.num_trained_timesteps
    #     ddim_timesteps_indices = (final_schedule_ddim_t.cpu().to(torch.float64) * n_model_steps).round().long() - 1
    #     ddim_timesteps_indices = torch.clamp(ddim_timesteps_indices, min=0)
    #     ddim_timesteps = ddim_timesteps_indices.cpu().numpy()

    #     logging.info(f"\n转换后的离散 DDIM Timesteps ({len(ddim_timesteps)} 点，可直接用于采样器):")
    #     print(ddim_timesteps)
    # else:
    #     logging.error("未能成功生成最终 DDIM 调度。")