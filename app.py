import torch

import gradio as gr
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts.to(device)


def predict(prompt, language, speaker_wav, agree=False):
    """
    Main body function to run inference, with light checks to ensure valid arguments are passed to the model.

    Args:
        prompt (`str`, required):
            Text prompt to the model.
        language (`str`, required):
            Language for inference.
        speaker_wav (`str`, required):
            Path to the speaker prompt audio file.
        agree (`bool`, required, defaults to `False`):
            Whether or not the model terms have been agreed to.
    Returns:
        tuple of (waveform_visual, synthesised_audio):
            Video animation of the output speech, and audio file.
    """
    if agree:
        if len(prompt) < 2:
            raise gr.Error("Please give a longer text prompt")

        tts.tts_to_file(
            text=prompt,
            file_path="output.wav",
            speaker_wav=speaker_wav,
            language=language,
        )

        return (
            gr.make_waveform(
                audio="output.wav",
            ),
            "output.wav",
        )
    else:
        gr.Warning(
            "Please accept the Terms & Conditions of the model by checking the box!"
        )
        return (
            None,
            None,
        )


title = "Coqui🐸 XTTS"

description = """
<a href="https://huggingface.co/coqui/XTTS-v1">XTTS</a> is a Voice generation model that lets you clone voices into different languages by using just a quick 3-second audio clip. 
<br/>
Built on Tortoise, XTTS has important model changes that make cross-language voice cloning and multi-lingual speech generation super easy. 
<br/>
This is the same model that powers Coqui Studio, and Coqui API, however we apply a few tricks to make it faster and support streaming inference.
<br/>
<br/>
<p>For faster inference without waiting in the queue, you should duplicate this space and upgrade to GPU via the settings.
<br/>
<a href="https://huggingface.co/spaces/coqui/xtts?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
</p>
"""

article = """
<div style='margin:20px auto;'>
<p>By using this demo you agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml</p>
</div>
"""

examples = [
    [
        "Once when I was six years old I saw a magnificent picture",
        "en",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Lorsque j'avais six ans j'ai vu, une fois, une magnifique image",
        "fr",
        "examples/male.wav",
        None,
        False,
        True,
    ],
    [
        "Un tempo lontano, quando avevo sei anni, vidi un magnifico disegno",
        "it",
        "examples/female.wav",
        None,
        False,
        True,
    ],
    [
        "Bir zamanlar, altı yaşındayken, muhteşem bir resim gördüm",
        "tr",
        "examples/female.wav",
        None,
        False,
        True,
    ],
]

audio_upload = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            value="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cz",
                "ar",
                "zh-cn",
            ],
            max_choices=1,
            value="en",
        ),
        gr.Audio(
            label="Reference Audio",
            info="Click on the ✎ button to upload your own target speaker audio",
            type="filepath",
            value="examples/female.wav",
            source="upload",
        ),
        gr.Checkbox(
            label="Agree",
            value=False,
            info="I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml",
        ),
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio"),
    ],
    description=description,
    article=article,
    examples=examples,
)

microphone = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            value="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cz",
                "ar",
                "zh-cn",
            ],
            max_choices=1,
            value="en",
        ),
        gr.Audio(
            label="Reference Audio",
            info="Record your own target speaker audio",
            type="filepath",
            source="microphone",
        ),
        gr.Checkbox(
            label="Agree",
            value=False,
            info="I agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml",
        ),
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio"),
    ],
    description=description,
    article=article,
)

demo = gr.Blocks()

with demo:
    gr.TabbedInterface(
        [audio_upload, microphone], ["Audio file", "Microphone"], title=title
    )

demo.launch(enable_queue=True)
