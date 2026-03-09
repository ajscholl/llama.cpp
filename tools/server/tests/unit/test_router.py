import pytest
from pathlib import Path
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()


@pytest.mark.parametrize(
    "model,success",
    [
        ("ggml-org/tinygemma3-GGUF:Q8_0", True),
        ("non-existent/model", False),
    ]
)
def test_router_chat_completion_stream(model: str, success: bool):
    global server
    server.start()
    content = ""
    ex: ServerError | None = None
    try:
        res = server.make_stream_request("POST", "/chat/completions", data={
            "model": model,
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
            ],
            "stream": True,
        })
        for data in res:
            if data["choices"]:
                choice = data["choices"][0]
                if choice["finish_reason"] in ["stop", "length"]:
                    assert "content" not in choice["delta"]
                else:
                    assert choice["finish_reason"] is None
                    content += choice["delta"]["content"] or ''
    except ServerError as e:
        ex = e

    if success:
        assert ex is None
        assert len(content) > 0
    else:
        assert ex is not None
        assert content == ""


def _get_model_status(model_id: str) -> str:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item["status"]["value"]
    raise AssertionError(f"Model {model_id} not found in /models response")


def _get_model_info(model_id: str) -> dict:
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    for item in res.body.get("data", []):
        if item.get("id") == model_id or item.get("model") == model_id:
            return item
    raise AssertionError(f"Model {model_id} not found in /models response")


def _wait_for_model_status(model_id: str, desired: set[str], timeout: int = 60) -> str:
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        last_status = _get_model_status(model_id)
        if last_status in desired:
            return last_status
        time.sleep(1)
    raise AssertionError(
        f"Timed out waiting for {model_id} to reach {desired}, last status: {last_status}"
    )


def _load_model_and_wait(
    model_id: str, timeout: int = 60, headers: dict | None = None
) -> None:
    load_res = server.make_request(
        "POST", "/models/load", data={"model": model_id}, headers=headers
    )
    assert load_res.status_code == 200
    assert isinstance(load_res.body, dict)
    assert load_res.body.get("success") is True
    _wait_for_model_status(model_id, {"loaded"}, timeout=timeout)


def test_router_unload_model():
    global server
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    _load_model_and_wait(model_id)

    unload_res = server.make_request("POST", "/models/unload", data={"model": model_id})
    assert unload_res.status_code == 200
    assert unload_res.body.get("success") is True
    _wait_for_model_status(model_id, {"unloaded"})


def test_router_models_max_evicts_lru():
    global server
    server.models_max = 2
    server.start()

    candidate_models = [
        "ggml-org/tinygemma3-GGUF:Q8_0",
        "ggml-org/test-model-stories260K",
        "ggml-org/test-model-stories260K-infill",
    ]

    # Load only the first 2 models to fill the cache
    first, second, third = candidate_models[:3]

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)

    # Verify both models are loaded
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    # Load the third model - this should trigger LRU eviction of the first model
    _load_model_and_wait(third, timeout=120)

    # Verify eviction: third is loaded, first was evicted
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"


def _write_router_weight_preset(path: Path) -> None:
    path.write_text(
        """version = 1

[*]
models-max-weight = 4

[ggml-org/tinygemma3-GGUF:Q8_0]
model-weight = 2

[ggml-org/test-model-stories260K]
model-weight = 2

[ggml-org/test-model-stories260K-infill]
model-weight = 3
""",
        encoding="utf-8",
    )


def test_router_models_max_weight_evicts_lru(tmp_path: Path):
    global server
    preset_path = tmp_path / "router-weights.ini"
    _write_router_weight_preset(preset_path)

    server.models_preset = str(preset_path)
    server.models_max = 0
    server.start()

    first = "ggml-org/tinygemma3-GGUF:Q8_0"
    second = "ggml-org/test-model-stories260K"
    third = "ggml-org/test-model-stories260K-infill"

    _load_model_and_wait(first, timeout=120)
    _load_model_and_wait(second, timeout=120)
    assert _get_model_status(first) == "loaded"
    assert _get_model_status(second) == "loaded"

    _load_model_and_wait(third, timeout=120)
    assert _get_model_status(third) == "loaded"
    assert _get_model_status(first) == "unloaded"
    assert _get_model_status(second) == "unloaded"


def test_router_models_max_weight_rejects_oversized_model(tmp_path: Path):
    global server
    preset_path = tmp_path / "router-weights.ini"
    _write_router_weight_preset(preset_path)

    preset_path.write_text(
        preset_path.read_text(encoding="utf-8") + "\n[oversized]\nmodel = ./tmp/tinyllamas/stories260K.gguf\nmodel-weight = 10\n",
        encoding="utf-8",
    )

    server.models_preset = str(preset_path)
    server.models_max = 0
    server.start()

    res = server.make_request("POST", "/models/load", data={"model": "oversized"})
    assert res.status_code == 500
    assert "error" in res.body


def test_router_no_models_autoload():
    global server
    server.no_models_autoload = True
    server.start()
    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 400
    assert "error" in res.body

    _load_model_and_wait(model_id)

    success_res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert success_res.status_code == 200
    assert "error" not in success_res.body


def test_router_api_key_required():
    global server
    server.api_key = "sk-router-secret"
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    auth_headers = {"Authorization": f"Bearer {server.api_key}"}

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 401
    assert res.body.get("error", {}).get("type") == "authentication_error"

    _load_model_and_wait(model_id, headers=auth_headers)

    authed = server.make_request(
        "POST",
        "/v1/chat/completions",
        headers=auth_headers,
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert authed.status_code == 200
    assert "error" not in authed.body


def test_router_chunk_endpoint():
    global server
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    document = "Section A. First sentence. Second sentence.\nSection B. Third sentence."

    res = server.make_request("POST", "/v1/chunk", data={
        "model": model_id,
        "document": document,
    })

    assert res.status_code == 200
    assert "chunks" in res.body
    small_chunks = []
    for big in res.body["chunks"]:
        small_chunks.extend(big["small_chunks"])
    assert "".join(chunk["text"] for chunk in small_chunks) == document


def test_router_recovers_from_crashed_child():
    global server
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    _load_model_and_wait(model_id, timeout=120)

    killed_pid = server.kill_latest_descendant()
    assert killed_pid > 0

    _wait_for_model_status(model_id, {"unloaded"}, timeout=30)

    res = server.make_request(
        "POST",
        "/v1/chat/completions",
        data={
            "model": model_id,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4,
        },
    )
    assert res.status_code == 200
    assert "error" not in res.body
    _wait_for_model_status(model_id, {"loaded"}, timeout=120)


def test_router_reports_crashed_child_exit_code():
    global server
    server.start()

    model_id = "ggml-org/tinygemma3-GGUF:Q8_0"
    _load_model_and_wait(model_id, timeout=120)

    server.kill_latest_descendant()
    _wait_for_model_status(model_id, {"unloaded"}, timeout=30)

    info = _get_model_info(model_id)
    status = info["status"]
    assert status["value"] == "unloaded"
    assert status["failed"] is True
    assert status["exit_code"] != 0
