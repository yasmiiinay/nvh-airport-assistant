"""Hello-world Gradio app for the Hugging Face Space DEPLOYMENT SMOKE TEST ONLY.

Purpose: prove the build path (requirements install, app boot, cold start)
before any model code exists. This file is replaced by the real interface in
final deployment. It deliberately imports nothing heavy.

ZeroGPU constraint (observed 12 Sep 2026): free Gradio Spaces run on ZeroGPU,
whose startup check refuses apps with no @spaces.GPU-decorated function
("No @spaces.GPU function detected during startup"). The inert probe below
satisfies that check; the UI never calls it, so no GPU quota is consumed.
Locally the `spaces` package is absent, so the decorator degrades to a no-op.
"""
import platform

import gradio as gr

try:
    import spaces  # preinstalled on ZeroGPU Spaces; not a project dependency
    _gpu_decorator = spaces.GPU
except ImportError:
    def _gpu_decorator(fn):
        return fn


@_gpu_decorator
def zerogpu_probe() -> str:
    """Exists only so ZeroGPU startup detection passes. Never called by the UI."""
    return "ok"

BANNER = (
    "Nordhaven Airport Assistant — deployment smoke test.\n"
    "No models are loaded. This placeholder verifies the Space build only."
)


def echo(message: str) -> str:
    if not message or not message.strip():
        return "Please type something so the round-trip can be verified."
    return (f"Echo: {message.strip()}\n"
            f"(python {platform.python_version()} on {platform.machine()})")


with gr.Blocks(title="NVH Assistant — smoke test") as demo:
    gr.Markdown(BANNER)
    inbox = gr.Textbox(label="Type anything", placeholder="hello")
    outbox = gr.Textbox(label="Round-trip result")
    send = gr.Button("Send")
    send.click(echo, inputs=inbox, outputs=outbox)

if __name__ == "__main__":
    demo.launch()
