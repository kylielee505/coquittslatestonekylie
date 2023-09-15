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
        return ()


title = "Coqui🐸 XTTS"

description = """
<a href="https://huggingface.co/coqui/XTTS-v1">XTTS</a> is a Voice generation model that lets you clone voices into different languages by using just a quick 3-second audio clip. 
<br/>
XTTS is built on previous research, like Tortoise, with additional architectural innovations and training to make cross-language voice cloning and multilingual speech generation possible. 
<br/>
This is the same model that powers our creator application <a href="https://coqui.ai">Coqui Studio</a> as well as the <a href="https://docs.coqui.ai">Coqui API</a>. In production we apply modifications to make low-latency streaming possible.
<br/>
Leave a star on the Github <a href="https://github.com/coqui-ai/TTS">🐸TTS</a>, where our open-source inference and training code lives.
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
        True,
    ],
    [
        "Lorsque j'avais six ans j'ai vu, une fois, une magnifique image",
        "fr",
        "examples/male.wav",
        True,
    ],
    [
        "Als ich sechs war, sah ich einmal ein wunderbares Bild",
        "de",
        "examples/female.wav",
        True,
    ],
    [
        "Cuando tenía seis años, vi una vez una imagen magnífica",
        "es",
        "examples/male.wav",
        True,
    ],
    [
        "Quando eu tinha seis anos eu vi, uma vez, uma imagem magnífica",
        "pt",
        "examples/female.wav",
        True,
    ],
    [
        "Kiedy miałem sześć lat, zobaczyłem pewnego razu wspaniały obrazek",
        "pl",
        "examples/male.wav",
        True,
    ],
    [
        "Un tempo lontano, quando avevo sei anni, vidi un magnifico disegno",
        "it",
        "examples/female.wav",
        True,
    ],
    [
        "Bir zamanlar, altı yaşındayken, muhteşem bir resim gördüm",
        "tr",
        "examples/female.wav",
        True,
    ],
    [
        "Когда мне было шесть лет, я увидел однажды удивительную картинку",
        "ru",
        "examples/female.wav",
        True,
    ],
    [
        "Toen ik een jaar of zes was, zag ik op een keer een prachtige plaat",
        "nl",
        "examples/male.wav",
        True,
    ],
    [
        "Když mi bylo šest let, viděl jsem jednou nádherný obrázek",
        "cs",
        "examples/female.wav",
        True,
    ],
    [
        "当我还只有六岁的时候， 看到了一副精彩的插画",
        "zh-cn",
        "examples/female.wav",
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
                "cs",
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
                "cs",
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
