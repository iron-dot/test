import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

#prompt = "Spiderman is surfing"
def video(prompt):
    
    video_frames = pipe(prompt, num_inference_steps=25).frames
    video_path = export_to_video(video_frames)
    return video_path
import gradio as gr
with gr.Blocks() as demo:
        gr.Markdown("AGI AI assistant system(3차 베타테스트)")
        with gr.Tab("video generate"):
            
            name_input = gr.Textbox()
            name_output = gr.Video()
            name_button = gr.Button("입력")
            name_button.click(video, inputs=name_input, outputs=name_output)
            
demo.launch(debug=True, share=True)
