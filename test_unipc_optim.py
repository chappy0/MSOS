
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import os
import json 


from ldm.models.diffusion.uni_pc.sampler import UniPCSampler 
from ldm.util import instantiate_from_config



from ldm.models.diffusion.stages_step_optim import StepOptim,NoiseScheduleVP
# --- Configuration ---
prompts_folder = "your prompt"

output_folder = "your output path" 
config_path = "configs/stable-diffusion/v2-inference.yaml"
ckpt_path = "./models/v2-1_512-ema-pruned.ckpt" # your should replace the path to your own model path
n_steps = 4 # the nfe value

guidance_scale = 7.5
height = 512
width = 512
batch_size =1



optimized_schedule_file = r'D:\paper\stablediffusion-main\stablediffusion-main\optimized_schedules_search\sd_v2_1_5steps_schedule_search.txt'

os.makedirs(output_folder, exist_ok=True)
print(f"ℹ️  Prompts will be read from: {prompts_folder}") 
print(f"ℹ️  Generated images will be saved to: {output_folder}")

# 1. load the model
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

# 2. check the optimal schedule
custom_timesteps = None
if os.path.exists(optimized_schedule_file):
    print(f"🔄 Loading optimized timesteps from: {optimized_schedule_file}")
    try:
        loaded_timesteps_np = np.loadtxt(optimized_schedule_file, dtype=np.float32)
        custom_timesteps = torch.from_numpy(loaded_timesteps_np).to(device) 
        if len(custom_timesteps) != n_steps + 1:
            print(f"⚠️ Loaded timesteps length {len(custom_timesteps)} does not match n_steps+1 ({n_steps+1}). Regenerating...")
            custom_timesteps = None 
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
    


    
    step_optimizer = StepOptim(ns=ns_instance)
    eps_t_0_for_optim = (1. / ns_instance.total_N) if ns_instance.schedule_name == 'discrete'  else 1e-3
    
    optimized_t_steps_tensor, _ = step_optimizer.get_ts_lambdas(
       n_steps,
        eps=eps_t_0_for_optim,
        initType='edm',

    )
    custom_timesteps = optimized_t_steps_tensor.to(device) 
    
    try:
        np.savetxt(optimized_schedule_file, custom_timesteps.cpu().numpy(), fmt='%.8f')
        print(f"✅ Optimized timesteps saved to {optimized_schedule_file}")
    except Exception as e:
        print(f"❌ Error saving timesteps to {optimized_schedule_file}: {e}")

# 3. set unipc sampler
sampler = UniPCSampler(model)




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
            custom_timesteps=custom_timesteps, 
            uni_pc_order=3,
            optimize_steps_on_demand=False
        )

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

