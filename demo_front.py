import gradio as gr
from img_predict import img_predict

model_1_path = "/workspace/image_prompt/sd-model-finetuned-base/checkpoint-15000/pytorch_model.bin"

def model_1_function(image_path,predict_method,k_value):
    if predict_method == "beam-k":
        return img_predict(model_1_path, image_path, opt={'device' : 'cuda',
                                                          'predict_method' : 'beam-k',
                                                          'k_value' : int(k_value) })
    elif predict_method == "random_sample":
        return img_predict(model_1_path, image_path, opt={'device' : 'cuda',
                                                          'predict_method' : 'random_sample' })

def model_2_function(image_path):
    # return img_predict(model_1_path, image_path, opt={'device' : 'cuda' })
    return "很好，你预测的很好"

def model_3_function(image_path):
    # return img_predict(model_1_path, image_path, opt={'device' : 'cuda' })
    return "待写"


def change_textbox(choice):
    if choice == "beam-k":
        return gr.Textbox(lines=1, visible=True)
    elif choice == "random_sample":
        return gr.Textbox(visible=False)

with gr.Blocks() as demo:
    gr.Markdown(#在这里面可以写markdown，好爽
    """
    # 使用方法：选择 model, 上传图片 -> 生成描述。
    """)
    with gr.Tab("model_1"):
        gr.Markdown(
            """
            # 模型测试
            """)
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath")
                choice=gr.Radio(["beam-k", "random_sample"], label="predict_method")
                text = gr.Textbox(lines=1, visible=True,label="k_value")
                choice.change(fn=change_textbox, inputs=choice, outputs=text)
            image_output = gr.Textbox(lines=4, interactive=True, label="caption", show_copy_button=True)
        image_button = gr.Button("Generate Caption")
        gr.Markdown("**示例如下**")
        gr.Examples(
        examples=[
            ["./images/MEN-Denim-id_00000826-01_1_front.jpg", "beam-k", "5"],
            ["./images/WOMEN-Dresses-id_00006368-02_2_side.jpg", "beam-k", "3"],
            ["./images/WOMEN-Tees_Tanks-id_00007961-06_4_full.jpg", "random_sample","null"]
            ],
        inputs=[image_input,choice,text],
        outputs=image_output,
        )
    
    image_button.click(model_1_function, inputs=[image_input,choice,text], outputs=image_output)
    
    with gr.Tab("model_2"):
        gr.Interface(
            fn=model_2_function, 
            inputs=[gr.Image(type="filepath")], 
            outputs="text")
        
    with gr.Tab("model_3"):
        gr.Interface(
            fn=model_3_function, 
            inputs=[gr.Image(type="filepath")], 
            outputs="text")


demo.launch(server_port=9001,share=True, inbrowser=True)
