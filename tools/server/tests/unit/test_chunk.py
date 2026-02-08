import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _collect_small_chunks(response_body: dict) -> list[dict]:
    out: list[dict] = []
    for big in response_body["chunks"]:
        out.extend(big["small_chunks"])
    return out


def test_chunk_basic_shape_and_reconstruction():
    global server
    server.start()

    document = (
        "Section 1. Llamas are social animals. They live in herds and communicate with body language.\n\n"
        "Section 2. Chunking should preserve semantic boundaries so retrieval quality stays high."
    )

    res = server.make_request("POST", "/v1/chunk", data={"document": document})
    assert res.status_code == 200
    assert "chunks" in res.body
    assert isinstance(res.body["chunks"], list)
    assert len(res.body["chunks"]) > 0

    small_chunks = _collect_small_chunks(res.body)
    rebuilt = "".join(chunk["text"] for chunk in small_chunks)
    assert rebuilt == document

    prev_end = 0
    for chunk in small_chunks:
        assert chunk["start_token"] <= chunk["end_token"]
        assert chunk["token_count"] == chunk["end_token"] - chunk["start_token"]
        assert chunk["start_token"] == prev_end
        prev_end = chunk["end_token"]


def test_chunk_invalid_params():
    global server
    server.start()

    res = server.make_request("POST", "/v1/chunk", data={
        "document": "abc",
        "small_min_tokens": 100,
        "small_target_tokens": 50,
        "small_max_tokens": 200,
    })

    assert res.status_code == 400
    assert "error" in res.body
