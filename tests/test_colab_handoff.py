import json
from pathlib import Path


def test_colab_launcher_is_safe_bounded_and_offline_after_acquisition():
    text = Path("scripts/colab_phase2_generation.py").read_text()
    assert "torch.cuda.is_available()" in text
    assert "--allow-model-download" in text
    assert 'HF_HUB_OFFLINE="1"' in text and 'TRANSFORMERS_OFFLINE="1"' in text
    assert '"--thinking-subset-per-domain", "256"' in text
    assert "validate_artifacts" in text and "content_overlaps" in text


def test_colab_notebook_defaults_expensive_toggles_off():
    notebook = json.loads(Path("notebooks/phase2_a100_generation.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "RUN_MODEL_DOWNLOAD = False" in source
    assert "RUN_PRIMARY = False" in source and "RUN_THINKING = False" in source
    assert "drive.mount" in source and "colab_phase2_generation.py" in source
