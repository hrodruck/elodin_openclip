import argparse, os
import cv2
import torch
import numpy as np
import open_clip
import clip
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from torch import optim
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode
from torchvision.transforms.functional import pil_to_tensor
from torch import linalg
import math

from face_loss_test import compare_tensor_face

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def custom_openclip_preprocess(img):
    return Compose([
        Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])(img)

def compare_txt_img(model, preprocess, tokenizer, img, txt):
    img = preprocess(img).unsqueeze(0) #this is a custom preprocess function based on openclip's notebook
    txt = tokenizer(txt).cuda()
    #debug
    print(f'tokenized text is {txt=}')
    img_features = model.encode_image(img)
    txt_features = model.encode_text(txt)
    return -img_features @ txt_features.T
    
def compare_img_img(model, preprocess, img1, img2):
    img1 = preprocess(img1).unsqueeze(0) #this is a custom preprocess function based on openclip's notebook
    img2 = preprocess(img2).unsqueeze(0)
    img1_features = model.encode_image(img1)
    img2_features = model.encode_image(img2)
    return -img1_features @ img2_features.T
    

def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v2-elodin.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cpu"
    )
    parser.add_argument(
        "--torchscript",
        action='store_true',
        help="Use TorchScript",
    )
    parser.add_argument(
        "--ipex",
        action='store_true',
        help="Use Intel® Extension for PyTorch*",
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    opt = parser.parse_args()
    return opt


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def main(opt):
    seed_everything(opt.seed)
    seed = opt.seed

    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device)

    if opt.plms:
        sampler = PLMSSampler(model, device=device)
    elif opt.dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(opt.repeat)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    if opt.torchscript or opt.ipex:
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder
        additional_context = torch.cpu.amp.autocast() if opt.bf16 else nullcontext()
        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        if opt.bf16 and not opt.torchscript and not opt.ipex:
            raise ValueError('Bfloat16 is supported only for torchscript+ipex')
        if opt.bf16 and unet.dtype != torch.bfloat16:
            raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                             "you'd like to use bfloat16 with CPU.")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError("Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

        if opt.ipex:
            import intel_extension_for_pytorch as ipex
            bf16_dtype = torch.bfloat16 if opt.bf16 else None
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)

            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if opt.torchscript:
            with torch.no_grad(), additional_context:
                # get UNET scripted
                if unet.use_checkpoint:
                    raise ValueError("Gradient checkpoint won't work with tracing. " +
                    "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                t_in = torch.ones(2, dtype=torch.int64)
                context = torch.ones(2, 77, 1024, dtype=torch.float32)
                scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                scripted_unet = torch.jit.optimize_for_inference(scripted_unet)
                print(type(scripted_unet))
                model.model.scripted_diffusion_model = scripted_unet

                # get Decoder for first stage model scripted
                samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                scripted_decoder = torch.jit.optimize_for_inference(scripted_decoder)
                print(type(scripted_decoder))
                model.first_stage_model.decoder = scripted_decoder

        prompts = data[0]
        print("Running a forward pass to initialize optimizations")
        uc = None
        if opt.scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        with additional_context:
            for _ in range(3):
                c = model.get_learned_conditioning(prompts)
            samples_ddim, _ = sampler.sample(S=5,
                                             conditioning=c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=opt.scale,
                                             unconditional_conditioning=uc,
                                             eta=opt.ddim_eta,
                                             x_T=start_code)
            print("Running a forward pass for decoder")
            for _ in range(3):
                x_samples_ddim = model.decode_first_stage(samples_ddim)

    precision_scope = autocast if opt.precision=="autocast" or opt.bf16 else nullcontext
    with precision_scope(opt.device), \
        model.ema_scope():

        all_samples = list()
        for n in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                
                namecon = model.cond_stage_model(prompts)
                namecon.requires_grad = False
                swap_out = (4, 5)
                swap_namecon = (4, 5)
                
                
                namecon[:,swap_out[0]:swap_out[1],:] = namecon[:,swap_namecon[0]:swap_namecon[1],:] 
                namecon.requires_grad_()
                namecon.retain_grad()
                mask = torch.ones_like(namecon) * 1e-6
                mask[:,swap_out[0]:swap_out[1],:] = 1 - 1e-6
                mask.requires_grad_()
                inv_mask = 1 - mask + 1e-6 #(just to be sure we don't flip this to a negative value)
                inv_mask.requires_grad_()
                
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                        
                initial_namecon = model.cond_stage_model(prompts).requires_grad_()
                optimizer = optim.Adam([namecon], lr = 8e-1)
                grad_acc = 1


                target_img = "tgt.jpg"
                #target_phrase = "Photo of a gem. A blue, pretty precious stone."
                target = target_img
                
                openclip_model, _, openclip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k', precision='fp16', device='cuda')
                for param in openclip_model.parameters():
                    param.requires_grad = False
                openclip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
                '''
                clip_model, _ = clip.load("ViT-B/32", device=device)
                clip_tokenizer = clip.tokenize

                for param in clip_model.parameters():
                    param.requires_grad = False
                '''
                for optimizing_iter in range (400):
                    
                    #to normalize the namecon to the sqrt of the expected normal distribution, do something like this
                    namecon_relevant_norm = linalg.vector_norm(namecon[:,swap_out[0]:swap_out[1],:], keepdim=True, dim = -1)    
                    reg_constant = 0.02
                    reg_loss = torch.mean(namecon_relevant_norm) * reg_constant
                    
                    
                    
                    '''
                    print ("printing about namecon")
                    print(namecon.is_leaf)
                    print (namecon.requires_grad)
                    '''
                    
                    c = initial_namecon * inv_mask + namecon * mask
                    
                    '''
                    print ("debug print")
                    print (mask.requires_grad)
                    print (initial_namecon.requires_grad)
                    print(c.shape)
                    print(namecon.shape)
                    print(mask.shape)
                    print(inv_mask.shape)
                    print (namecon[:,swap_namecon[0]:swap_namecon[1],:] )
                    print (namecon[:,swap_out[0]:swap_out[1],:])
                    print (c[:,swap_out[0]:swap_out[1],:])
                    print(inv_mask[:,(swap_out[0]-1):swap_out[0],:])
                    
                    
                    print ("printing conditioning itself")
                    print (c)
                    print (linalg.vector_norm(c[:,swap_out[0]:swap_out[1],:], keepdim=True, dim = -1))
                    print (c.requires_grad)
                    print (c.is_leaf)
                    '''
                    
                    
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples, _ = sampler.sample(S=opt.steps,
                                                     conditioning=c,
                                                     batch_size=opt.n_samples,
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta,
                                                     x_T=start_code)
                    print ("grad_fn before decode first stage")
                    print (samples.requires_grad)
                    print (samples.grad_fn)
                    print (samples.is_leaf)
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    print (x_samples.grad_fn)


                    for x_sample in x_samples:
                        loss = compare_txt_img (openclip_model, custom_openclip_preprocess, openclip_tokenizer, x_sample, target)
                        #loss = compare_txt_img (clip_model, custom_openclip_preprocess, clip_tokenizer, x_sample, target)
                        #loss = compare_tensor_face(x_sample, target)
                        print(f'{reg_loss=}')
                        print (f'{loss=}')
                        loss = loss + reg_loss
                        print (f'full loss is {loss}')
                        loss /= grad_acc
                        print (optimizing_iter)
                        loss.backward()
                        if base_count % grad_acc ==0:                        
                            optimizer.step()
                            optimizer.zero_grad()
                        if base_count>0 and base_count % 2 ==0:
                            x_sample = 255. * rearrange(x_sample, 'c h w -> h w c')
                            x_sample = x_sample.detach().cpu().numpy()
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                        if base_count % 25 == 0:
                            namecon_to_save = namecon.clone().detach()
                            namecon_to_save.requires_grad = False
                            namecon_to_save[:,swap_namecon[0]:swap_namecon[1],:] = namecon_to_save[:,swap_out[0]:swap_out[1],:]
                            namecon_dict = {
                                'namecon':namecon_to_save,
                                'relevant_indexes': torch.tensor(list(swap_namecon))
                            }
                            namecon_filename = os.path.join(sample_path, f'namecon_{base_count:05}.safetensors')
                            save_safetensors(namecon_dict, namecon_filename)
                        base_count += 1
                        sample_count += 1
                    del x_samples
                    del x_sample
    
    
    namecon.requires_grad = False
    namecon[:,swap_namecon[0]:swap_namecon[1],:] = namecon[:,swap_out[0]:swap_out[1],:]
    namecon_dict = {
        'namecon':namecon,
        'relevant_indexes': torch.tensor(list(swap_namecon))
    }
    namecon_filename = os.path.join(sample_path, 'namecon.safetensors')
    save_safetensors(namecon_dict, namecon_filename)


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")
          

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
