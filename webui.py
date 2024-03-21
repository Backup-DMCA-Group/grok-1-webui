# Copyright 2024 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model
import gradio as gr
import threading

CKPT_PATH = "./checkpoints/"


def load_model():
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    generator = inference_runner.run()
    return generator


def launch_webui(generator):
    generate_lock = threading.RLock()

    def do_generate(prompt, max_len, temperature):
        generate_lock.acquire()
        try:
            return sample_from_model(generator, prompt, int(max_len), float(temperature))
        finally:
            generate_lock.release()

    with gr.Blocks() as webui:
        with gr.Column():
            prompt = gr.TextArea(label="prompt")
            with gr.Row():
                max_len = gr.Number(100, minimum=1, step=1, label="max_len")
                temperature = gr.Slider(value=0.01, minimum=0, maximum=1, label="temperature")
            with gr.Row():
                generate = gr.Button("generate", variant="primary")
            output = gr.TextArea(label="output", interactive=False)
            generate.click(do_generate, inputs=[prompt, max_len, temperature], outputs=[output])
        webui.launch()


def main():
    generator = load_model()
    launch_webui(generator)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
