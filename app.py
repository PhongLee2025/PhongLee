import os
from pathlib import Path
import platform
import sys
sys.path.append('PhongLee1')
import cv2
import torch
import torch.nn.functional as F
import gradio as gr

from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY

# CSS tùy chỉnh
dark_theme_css = """
<style>
  body { 
    background-color: #1E1E1E; 
    color: #FFFFFF; 
  }
  h1, h2, h3, h4, h5, h6, .gr-markdown { 
    color: #FFA500 !important; /* Màu cam cho tiêu đề */
    font-weight: bold !important; /* Tô đậm */
    font-size: 2.5em !important; /* Kích thước lớn hơn */
  }
  .gr-markdown p {
    color: #FFA500 !important; /* Màu cam cho đoạn văn Markdown */
  }
  .gr-button { 
    background-color: #3B3B3B; 
    color: white; 
  }
  input[type='number'], input[type='text'], textarea, .gr-textbox, .gr-slider, .gr-checkbox, .gr-file-upload {
    background-color: #FFA500 !important; /* Màu cam */
    color: white;
  }
  label, .label, .checkbox-label {
    color: #FFA500 !important; /* Đảm bảo nhãn và các label chuyển sang màu cam */
  }
  .gradio-container {
    background-color: #1E1E1E; 
  }
  .gr-slider input[type='range'] {
    background-color: #FFA500 !important; /* Màu cam cho thanh trượt */
  }
</style>
"""

def open_folder():
    open_folder_path = os.path.abspath("outputs")
    if platform.system() == "Windows":
        os.startfile(open_folder_path)
    elif platform.system() == "Linux":
        os.system(f'xdg-open "{open_folder_path}"')

#os.system("pip freeze")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
args = parser.parse_args()

pretrain_model_url = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'realesrgan_x2': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth',
    'realesrgan_x4': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
}

# Download weights if not already present
if not os.path.exists('PhongLee1/weights/CodeFormer/codeformer.pth'):
    load_file_from_url(url=pretrain_model_url['codeformer'], model_dir='PhongLee1/weights/CodeFormer', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
    load_file_from_url(url=pretrain_model_url['detection'], model_dir='PhongLee1/weights/facelib', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/facelib/parsing_parsenet.pth'):
    load_file_from_url(url=pretrain_model_url['parsing'], model_dir='PhongLee1/weights/facelib', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth'):
    load_file_from_url(url=pretrain_model_url['realesrgan_x2'], model_dir='PhongLee1/weights/realesrgan', progress=True, file_name=None)
if not os.path.exists('CodeFormer/weights/realesrgan/RealESRGAN_x4plus.pth'):
    load_file_from_url(url=pretrain_model_url['realesrgan_x4'], model_dir='PhongLee1/weights/realesrgan', progress=True, file_name=None)

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# set enhancer with RealESRGAN based on upscale factor
def set_realesrgan(upscale_factor):
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=upscale_factor,
    )
    model_path = f"PhongLee1/weights/realesrgan/RealESRGAN_x{upscale_factor}plus.pth"
    upsampler = RealESRGANer(
        scale=upscale_factor,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to(device)
ckpt_path = "PhongLee1/weights/CodeFormer/codeformer.pth"
checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()

os.makedirs('outputs', exist_ok=True)

def inference(image, face_align, background_enhance, face_upsample, upscale, codeformer_fidelity, dont_save=False):
    """Chạy dự đoán duy nhất trên mô hình"""
    print('Bắt đầu xử lý')
    try: 
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        face_align = face_align if face_align is not None else True
        background_enhance = background_enhance if background_enhance is not None else True
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        has_aligned = not face_align
        upscale = 1 if has_aligned else upscale

        upscale = int(upscale)
        if upscale > 6:
            upscale = 6

        upsampler = set_realesrgan(min(upscale, 4))

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        print('\tKích thước hình ảnh:', img.shape)

        if upscale > 2 and max(img.shape[:2])>1000: 
            upscale = 2 
        if max(img.shape[:2]) > 1500: 
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            face_helper.align_warp_face()

        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Không thể phục hồi bằng CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        if not has_aligned:
            if bg_upsampler is not None:
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )
        else:
            restored_img = restored_face

        if not dont_save:
            save_image(restored_img)

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print('Lỗi toàn cục', error)
        return None, None

def save_image(restored_img):
    base_filename = "img_"
    extension = ".png"
    
    output_dir = "outputs"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = [f for f in os.listdir(output_dir) if f.startswith(base_filename) and f.endswith(extension)]
    
    if image_files:
        numbers = [int(f[len(base_filename):-len(extension)]) for f in image_files]
        max_number = max(numbers)
        new_number = max_number + 1
    else:
        new_number = 1
    
    new_filename = f"{base_filename}{new_number:04d}{extension}"
    
    save_path = os.path.join(output_dir, new_filename)
    
    imwrite(restored_img, save_path)
    
    print(f"Hình ảnh đã lưu tại {save_path}")

import time

def batch_inference(batch_input_folder, batch_output_folder, face_align, background_enhance, face_upsample, upscale, codeformer_fidelity, progress=gr.Progress()):
    if not os.path.exists(batch_input_folder):
        print(f"Thư mục đầu vào không tồn tại: {batch_input_folder}")
        return

    if not batch_output_folder:
        batch_output_folder = "batch_outputs"
    
    if not os.path.exists(batch_output_folder):
        os.makedirs(batch_output_folder)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    image_files = [file for file in os.listdir(batch_input_folder) if Path(file).suffix.lower() in image_extensions]
    total_images = len(image_files)

    start_time = time.time()

    for i, file in enumerate(image_files, start=1):
        image_path = os.path.join(batch_input_folder, file)
        try:
            restored_img = inference(image_path, face_align, background_enhance, face_upsample, upscale, codeformer_fidelity, True)
            
            if restored_img is not None:
                save_path = os.path.join(batch_output_folder, f"{Path(file).stem}.png")
                j = 0
                while os.path.exists(save_path):
                    j += 1
                    save_path = os.path.join(batch_output_folder, f"{Path(file).stem}_{j:04d}.png")
                restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
                imwrite(restored_img, save_path)
                print(f"Xử lý: {image_path}")
            else:
                print(f"Không thể xử lý: {image_path}")
        except Exception as e:
            print(f"Lỗi xử lý {image_path}: {e}")
        
        elapsed_time = time.time() - start_time
        processed_images = i
        remaining_images = total_images - processed_images
        processing_speed = processed_images / elapsed_time
        estimated_time_remaining = remaining_images / processing_speed if processing_speed > 0 else 0
        progress_percent = (processed_images / total_images) * 100

        progress(progress_percent / 100, desc=f"Xử lý: {processed_images}/{total_images} | Tốc độ: {processing_speed:.2f} img/s | "
                                               f"Thời gian còn lại: {estimated_time_remaining:.2f}s")

    print("Xử lý hàng loạt đã hoàn tất.")

title = "Phong Lee 0832 328262- Công Cụ Làm Nét Ảnh"
description = r"""Nhận Phục Hồi Ảnh Cũ - Chỉnh Sửa Ảnh Giá Rẻ"""

def clear():
    return None, False, False, False, 2, 0.5

# Khởi tạo giao diện Gradio
with gr.Blocks() as demo:  
    # Thêm CSS tùy chỉnh cho giao diện tối và điều chỉnh yêu cầu
    gr.HTML(dark_theme_css)

    # Markdown tiêu đề, sẽ đổi màu sang cam
    gr.Markdown(
        "## Phong Lee 0832 328262- Công Cụ Làm Nét Ảnh"
    )
    gr.Markdown(
        "## Nhận Phục Hồi Ảnh Cũ - Chỉnh Sửa Ảnh Giá Rẻ"
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Hình ảnh Đầu vào", height=512)
            face_align = gr.Checkbox(value=True, label="Căn chỉnh Khuôn mặt")
            background_enhance = gr.Checkbox(value=True, label="Tăng cường Nền")
            face_upsample = gr.Checkbox(value=True, label="Tăng cường Khuôn mặt")
            upscale = gr.Number(value=2, label="Hệ số Phóng to (tối đa 6)")
            codeformer_fidelity = gr.Slider(0, 1, value=0.5, step=0.01, label='Mức độ Phục hồi (0: chất lượng tốt hơn, 1: nhận dạng tốt hơn)')
            
            with gr.Row():
                submit_button = gr.Button("Xử lý")
                clear_button = gr.Button("Xóa")

        with gr.Column():
            image_output = gr.Image(type="numpy", label="Hình ảnh Đầu ra", format="png")
            btn_open_outputs = gr.Button("Mở Thư mục Đầu ra")
            batch_input_folder = gr.Textbox(label="Thư mục Đầu vào cho Xử lý Hàng loạt")
            batch_output_folder = gr.Textbox(label="Thư mục Đầu ra cho Xử lý Hàng loạt (Tùy chọn)")
            status_label = gr.Label()
            batch_process_button = gr.Button("Bắt đầu Xử lý Hàng loạt")

    submit_button.click(inference, inputs=[image_input, face_align, background_enhance, face_upsample, upscale, codeformer_fidelity], outputs=image_output)
    clear_button.click(clear, outputs=[image_input, face_align, background_enhance, face_upsample, upscale, codeformer_fidelity])
    btn_open_outputs.click(fn=open_folder)

    batch_process_button.click(batch_inference, inputs=[batch_input_folder, batch_output_folder, face_align, background_enhance, face_upsample, upscale, codeformer_fidelity],outputs=[status_label],  show_progress=True, queue=True)

demo.launch(inbrowser=True, share=args.share)
