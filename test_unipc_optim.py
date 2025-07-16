
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import os
import json # For saving/loading schedule if you prefer JSON

# 假设 sampler.py 和 ldm 模块在您的 PYTHONPATH 中
from ldm.models.diffusion.uni_pc.sampler import UniPCSampler # 您修改后的 UniPCSampler

# from ldm.models.diffusion.uni_pc.uni_pc import model_wrapper, UniPC
# from ldm.models.diffusion.uni_pc.uni_pc import NoiseScheduleVP
from ldm.util import instantiate_from_config

# from ldm.models.diffusion.step_optim_search import StepOptim,NoiseScheduleVP

from ldm.models.diffusion.stages_step_optim import StepOptim,NoiseScheduleVP
# --- Configuration ---
prompts_folder = r"D:\paper\FCDiffusion_code-main\datasets\laion_aesthetics_6.5\laion-600k-aesthetic-6.5plus-768"
# prompts_folder = r"D:\paper\FCDiffusion_code-main\datasets\00001"
output_folder = r"D:\paper\FCDiffusion_code-main\datasets\unipc_test_laion" # 新的输出文件夹
config_path = "configs/stable-diffusion/v2-inference.yaml"
ckpt_path = r"D:\paper\FCDiffusion_code-main\models\v2-1_512-ema-pruned.ckpt"
n_steps = 4 # 您期望的采样步数

guidance_scale = 7.5
height = 512
width = 512
batch_size =1

# 定义优化时间表文件的路径
# optimized_schedule_dir = "./optimized_schedules_unipc" # 建议将调度文件统一存放
# os.makedirs(optimized_schedule_dir, exist_ok=True)
# # 基于模型、配置和步数生成一个独特的文件名
# # (更复杂可以加入 StepOptim 的核心参数摘要到文件名中)
# schedule_filename = f"sd_v2_1_{n_steps}steps_schedule_0612.txt"
# optimized_schedule_file = os.path.join(optimized_schedule_dir, schedule_filename)

optimized_schedule_file = r'D:\paper\stablediffusion-main\stablediffusion-main\optimized_schedules_search\sd_v2_1_5steps_schedule_search.txt'

os.makedirs(output_folder, exist_ok=True)
print(f"ℹ️  Prompts will be read from: {prompts_folder}") # 修正路径显示
print(f"ℹ️  Generated images will be saved to: {output_folder}") # 修正路径显示

# 1. 加载模型 (与您之前脚本一致，确保dtype修正)
print("🔄 Loading model...")
config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)

if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model') and \
   hasattr(model.model.diffusion_model, 'dtype') and model.model.diffusion_model.dtype == torch.float16:
    print("Overriding UNet's internal dtype from torch.float16 to torch.float32.")
    model.model.diffusion_model.dtype = torch.float32
model.float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
print("✅ Model loaded successfully!")

# 2. 检查或生成优化后的时间步
custom_timesteps = None
if os.path.exists(optimized_schedule_file):
    print(f"🔄 Loading optimized timesteps from: {optimized_schedule_file}")
    try:
        loaded_timesteps_np = np.loadtxt(optimized_schedule_file, dtype=np.float32)
        custom_timesteps = torch.from_numpy(loaded_timesteps_np).to(device) # 直接加载到目标设备
        if len(custom_timesteps) != n_steps + 1:
            print(f"⚠️ Loaded timesteps length {len(custom_timesteps)} does not match n_steps+1 ({n_steps+1}). Regenerating...")
            custom_timesteps = None # 标记为需要重新生成
        else:
            print(f"✅ Loaded {len(custom_timesteps)} timesteps.")
    except Exception as e:
        print(f"❌ Error loading timesteps from {optimized_schedule_file}: {e}. Will regenerate.")
        custom_timesteps = None

if custom_timesteps is None:
    print(f"🔄 Optimized schedule not found or invalid. Generating new schedule for {n_steps} steps...")
    if not hasattr(model, 'alphas_cumprod'):
        raise ValueError("The main diffusion model does not have 'alphas_cumprod'.")
    
    alphas_for_ns = model.alphas_cumprod.clone().detach().cpu().to(torch.float32)
    ns_instance = NoiseScheduleVP('discrete', alphas_cumprod=alphas_for_ns, dtype=torch.float32)
    

    # step_optimizer_params = {
    #     'p_is_dynamic': True, 'p_val_at_T': 3.6, 'p_val_at_eps': 0.06,
    #     'p_dynamic_transform': 'sigmoid', 'p_dynamic_sigmoid_k': 3.6,
    #     'objective_type': 'paper_power_q', 'power_q_val': 1.6, # 或者 'minimax_lte'
    #     'use_pf_inspired_error_metric': True,
    #     'use_rho_scaling': False, 'rho_q_is_dynamic': False,
    #     'rho_q_base_val': 1.6, 'rho_q_nfe_factor': 0.6,

    # }

    # step_optimizer = StepOptim(ns=ns_instance.to(torch.device("cpu")), **step_optimizer_params)
    
    step_optimizer = StepOptim(ns=ns_instance)
    eps_t_0_for_optim = (1. / ns_instance.total_N) if ns_instance.schedule == 'discrete'  else 1e-3
    # eps_t_0_for_optim = (1. / ns_instance.total_N) if ns_instance.schedule_name == 'discrete'  else 1e-3
    
    optimized_t_steps_tensor, _ = step_optimizer.get_ts_lambdas(
       n_steps,
        eps=eps_t_0_for_optim,
        # eps=0.02464701,
        # 0.02593,
        initType='edm',
        # init_rho=5.9
    )
    custom_timesteps = optimized_t_steps_tensor.to(device) # 确保在正确的设备上
    
    try:
        np.savetxt(optimized_schedule_file, custom_timesteps.cpu().numpy(), fmt='%.8f')
        print(f"✅ Optimized timesteps saved to {optimized_schedule_file}")
    except Exception as e:
        print(f"❌ Error saving timesteps to {optimized_schedule_file}: {e}")

# 3. 设置 UniPC 采样器
sampler = UniPCSampler(model)

# ... (后续的图像生成参数, 获取uc, 循环处理prompts - 与您提供的 test.py 相同) ...


uc = model.get_learned_conditioning(batch_size * [""])
print("🔄 Processing prompts...")
prompt_files = [f for f in os.listdir(prompts_folder) if f.endswith(".txt")]

if not prompt_files:
    print(f"⚠️ No .txt files found in the '{prompts_folder}' directory.")
else:
    for prompt_filename in prompt_files:
        prompt_filepath = os.path.join(prompts_folder, prompt_filename)
        try:
            with open(prompt_filepath, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"❌ Error reading {prompt_filename}: {e}")
            continue
        if not prompt: print(f"⚠️ Empty prompt in {prompt_filename}. Skipping."); continue

        print(f"\n✨ Generating image for prompt: '{prompt}'")
        c = model.get_learned_conditioning([prompt])
        # 确保 c 和 uc 在正确的设备上 (如果模型在GPU上)
        c = c.to(device)
        if uc is not None: uc = uc.to(device)

        shape_latent = [4, height // 8, width // 8]

        samples, _ = sampler.sample(
            S=n_steps, 
            conditioning=c,
            batch_size=batch_size,
            shape=shape_latent,
            verbose=True, 
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=uc,
            custom_timesteps=custom_timesteps, # <<< ✅ 传递优化后的时间步 (可能为 None)
            uni_pc_order=3,
            optimize_steps_on_demand=False
        )
        # ... (解码和保存图像 - 与您提供的 test.py 相同) ...
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
        x_samples_np = x_samples.cpu().permute(0, 2, 3, 1).numpy()
        for i, img_np_loop in enumerate(x_samples_np):
            img_pil = Image.fromarray((img_np_loop * 255).astype(np.uint8))
            base_name = os.path.splitext(prompt_filename)[0]
            output_image_filename = f"{base_name}_gs{guidance_scale}_steps{n_steps}_unipc_optim_ondemand.png"
            output_image_path = os.path.join(output_folder, output_image_filename)
            img_pil.save(output_image_path)
            print(f"🖼️ Image saved to: {output_image_path}")
    print("\n✅ All prompts processed!")

