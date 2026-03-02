# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the LlamaNemotronVL model family:
  - nvidia/llama-nemotron-embed-vl-1b-v2  (LlamaNemotronVLForCausalLM / embed)
  - nvidia/llama-nemotron-rerank-vl-1b-v2
      (LlamaNemotronVLForSequenceClassification / rerank)
  - nvidia/llama-nemotron-colembed-vl-3b-v2
      (LlamaNemotronColEmbedVLModel / late-interaction token_embed)

All variants share a SigLIP vision encoder with a bidirectional LLaMA backbone.
"""

import base64
from io import BytesIO
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor

from vllm.entrypoints.chat_utils import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
)
from vllm.entrypoints.pooling.score.utils import ScoreMultiModalParam, compute_maxsim_score

from ....conftest import IMAGE_ASSETS, HfRunner, PromptImageInput, VllmRunner
from ...utils import check_embeddings_close

# Prefixes used by the model API
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Text prompts for text-only embedding
HF_TEXT_PROMPTS = [
    # T -> X (text embedding queries)
    f"{QUERY_PREFIX}The label of the object is stop sign",
    f"{QUERY_PREFIX}cherry blossom",
]

# Image prompts using the model's expected format
HF_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        # I -> X (image embedding as passage/document)
        "stop_sign": f"{PASSAGE_PREFIX}<image>",
        "cherry_blossom": f"{PASSAGE_PREFIX}<image>",
    }
)

MODELS = ["nvidia/llama-nemotron-embed-vl-1b-v2"]


def _run_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    input_texts: list[str],
    input_images: PromptImageInput,
    model: str,
    *,
    dtype: str,
) -> None:
    """Run embedding comparison test between HF and vLLM.

    NOTE: Run vLLM first to avoid CUDA initialization issues with multiprocessing.
    """
    # Run vLLM inference first
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.embed(input_texts, images=input_images)

    # Run HF inference using the model's encode_queries/encode_documents API
    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs = []
        for text, image in zip(input_texts, input_images):
            with torch.inference_mode():
                if text.startswith(QUERY_PREFIX):
                    # Strip prefix and use encode_queries for query texts
                    query_text = text[len(QUERY_PREFIX) :]
                    embedding = hf_model.model.encode_queries([query_text])
                elif text.startswith(PASSAGE_PREFIX):
                    # Strip prefix and use encode_documents for passages/images
                    passage_text = text[len(PASSAGE_PREFIX) :]
                    if image is not None:
                        # Image document - pass image to encode_documents
                        embedding = hf_model.model.encode_documents(
                            images=[image],
                            texts=[passage_text],
                        )
                    else:
                        # Text-only document
                        embedding = hf_model.model.encode_documents(
                            texts=[passage_text]
                        )
                else:
                    raise ValueError(
                        f"Text must start with '{QUERY_PREFIX}' or '{PASSAGE_PREFIX}'"
                    )

                hf_outputs.append(embedding[0].tolist())

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models_text(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test text-only embedding."""
    input_texts_images = [(text, None) for text in HF_TEXT_PROMPTS]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,  # type: ignore
        model,
        dtype=dtype,
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_models_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test image embedding."""
    input_texts_images = [
        (text, asset.pil_image) for text, asset in zip(HF_IMAGE_PROMPTS, image_assets)
    ]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    _run_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,
        model,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Reranker tests — nvidia/llama-nemotron-rerank-vl-1b-v2
# ---------------------------------------------------------------------------

RERANKER_MODELS = ["nvidia/llama-nemotron-rerank-vl-1b-v2"]

# The tokenizer's built-in chat template is not suitable for the Score/Rerank
# APIs (it's inherited from the base LLM).  We must use the provided override.
_RERANKER_SCORE_TEMPLATE = (
    Path(__file__).parents[4]
    / "examples/pooling/score/template/nemotron-vl-rerank.jinja"
).read_text()

RERANKER_TEXT_QUERY = "How is AI improving the intelligence and capabilities of robots?"
RERANKER_TEXT_DOCS = [
    "AI enables robots to perceive, plan, and act autonomously.",
    (
        "A biological foundation model designed to analyze DNA, RNA, "
        "and protein sequences."
    ),
]

RERANKER_IMAGE_QUERY = "photo of a red stop sign on a street"


def _pil_to_data_uri(image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _run_hf_reranker(
    hf_runner: type[HfRunner],
    model: str,
    dtype: str,
    query: str,
    docs: list,
) -> list[float]:
    """Run HF reranker inference; docs is a list of (doc_text, doc_image|None)."""
    with hf_runner(
        model,
        dtype=dtype,
        trust_remote_code=True,
        auto_cls=AutoModelForSequenceClassification,
    ) as hf_model:
        processor = AutoProcessor.from_pretrained(
            model,
            trust_remote_code=True,
            max_input_tiles=6,
            use_thumbnail=True,
            rerank_max_length=2048,
        )
        examples = [
            {
                "question": query,
                "doc_text": doc_text if doc_text is not None else "",
                "doc_image": doc_image if doc_image is not None else "",
            }
            for doc_text, doc_image in docs
        ]
        batch_dict = processor.process_queries_documents_crossencoder(examples)
        batch_dict = {
            k: v.to(hf_model.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch_dict.items()
        }
        with torch.inference_mode():
            logits = hf_model.model(**batch_dict, return_dict=True).logits
        # vLLM applies sigmoid activation to the raw logits before returning
        # scores; apply the same here so both sides are comparable.
        scores = torch.sigmoid(logits.squeeze(-1).float())
        return scores.detach().cpu().tolist()


def _run_vllm_reranker(
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
    query: str,
    docs: list,
) -> list[float]:
    """Run vLLM reranker inference; docs is a list of (doc_text, doc_image|None)."""
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    ) as vllm_model:
        has_images = any(img is not None for _, img in docs)

        if not has_images:
            # Text-only path: use the simple string score API.
            queries = [query] * len(docs)
            doc_texts = [doc_text for doc_text, _ in docs]
            outputs = vllm_model.score(
                queries,
                doc_texts,
                chat_template=_RERANKER_SCORE_TEMPLATE,
            )
        else:
            # Multimodal path: build ScoreMultiModalParam for each pair.
            query_params = [
                ScoreMultiModalParam(
                    content=[
                        ChatCompletionContentPartTextParam(
                            type="text",
                            text=query,
                        )
                    ]
                )
            ] * len(docs)

            doc_params = []
            for doc_text, doc_image in docs:
                content: list = []
                if doc_image is not None:
                    content.append(
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url={"url": _pil_to_data_uri(doc_image)},
                        )
                    )
                if doc_text:
                    content.append(
                        ChatCompletionContentPartTextParam(
                            type="text",
                            text=doc_text,
                        )
                    )
                doc_params.append(ScoreMultiModalParam(content=content))

            raw_outputs = vllm_model.llm.score(
                query_params,
                doc_params,
                chat_template=_RERANKER_SCORE_TEMPLATE,
            )
            outputs = [o.outputs.score for o in raw_outputs]

    return outputs


def _run_reranker_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    model: str,
    dtype: str,
    query: str,
    docs: list,
) -> None:
    """Compare HF and vLLM reranker scores.

    NOTE: Run vLLM first to avoid CUDA initialization issues with multiprocessing.
    """
    vllm_scores = _run_vllm_reranker(vllm_runner, model, dtype, query, docs)
    hf_scores = _run_hf_reranker(hf_runner, model, dtype, query, docs)

    assert len(hf_scores) == len(vllm_scores), (
        f"Output length mismatch: HF={len(hf_scores)}, vLLM={len(vllm_scores)}"
    )
    for i, (hf_score, vllm_score) in enumerate(zip(hf_scores, vllm_scores)):
        assert hf_score == pytest.approx(vllm_score, rel=0.02), (
            f"Score mismatch at index {i}: HF={hf_score:.4f}, vLLM={vllm_score:.4f}"
        )


@pytest.mark.parametrize("model", RERANKER_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_reranker_text(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Test reranking with text-only query and text documents."""
    docs = [(text, None) for text in RERANKER_TEXT_DOCS]
    _run_reranker_test(hf_runner, vllm_runner, model, dtype, RERANKER_TEXT_QUERY, docs)


@pytest.mark.parametrize("model", RERANKER_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_reranker_image_doc(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test reranking with text query against image documents."""
    docs = [(None, asset.pil_image) for asset in image_assets]
    _run_reranker_test(hf_runner, vllm_runner, model, dtype, RERANKER_IMAGE_QUERY, docs)


# ---------------------------------------------------------------------------
# ColEmbed tests — nvidia/llama-nemotron-colembed-vl-3b-v2
# ---------------------------------------------------------------------------
# This model uses ColBERT-style late interaction: per-token L2-normalised
# embeddings are returned rather than a single sequence vector.
# HF inference: model(**inputs).last_hidden_state * mask → F.normalize per token
# vLLM inference: llm.encode(..., pooling_task="token_embed")
# ---------------------------------------------------------------------------

COLEMBED_MODELS = ["nvidia/llama-nemotron-colembed-vl-3b-v2"]

# Query prefix expected by the model (processor adds "query: " internally,
# but vLLM receives the full prompt verbatim).
COLEMBED_QUERY_PREFIX = "query: "

# Passage prefix for image documents. The HF processor builds the document
# prompt as:  "passage: <img>...<IMG_CONTEXT>...</img> passage: "
# (double prefix is an artefact of the HF processor implementation).
# We match this exact format on the vLLM side.
COLEMBED_PASSAGE_PREFIX = "passage: "

COLEMBED_TEXT_QUERIES = [
    f"{COLEMBED_QUERY_PREFIX}The label of the object is stop sign",
    f"{COLEMBED_QUERY_PREFIX}cherry blossom",
]

# Image document prompts replicating the HF processor's double-prefix format:
#   process_documents({"images": [img], "texts": [""]})
# produces: "passage: <img>...<IMG_CONTEXT>N...</img> passage:  "
# We pass "passage: <image> passage: " so the vLLM processor expands
# <image> to the same image token block and appends the matching suffix.
COLEMBED_IMAGE_PROMPTS = IMAGE_ASSETS.prompts(
    {
        "stop_sign": f"{COLEMBED_PASSAGE_PREFIX}<image> {COLEMBED_PASSAGE_PREFIX}",
        "cherry_blossom": f"{COLEMBED_PASSAGE_PREFIX}<image> {COLEMBED_PASSAGE_PREFIX}",
    }
)


def _run_colembed_test(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    input_texts: list[str],
    input_images: PromptImageInput,
    model: str,
    *,
    dtype: str,
) -> None:
    """Compare HF and vLLM per-token embeddings for the ColEmbed model.

    NOTE: Run vLLM first to avoid CUDA initialisation issues.
    """
    # --- vLLM ---
    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    ) as vllm_model:
        vllm_inputs = vllm_model.get_inputs(input_texts, images=input_images)
        req_outputs = vllm_model.llm.encode(vllm_inputs, pooling_task="token_embed")
        # outputs.data is a 2-D structure [n_tokens, hidden_size]; flatten for
        # comparison.
        vllm_outputs = [
            torch.tensor(req.outputs.data).reshape(-1).tolist()
            for req in req_outputs
        ]

    # --- HF ---
    hf_outputs = []
    with hf_runner(
        model, dtype=dtype, auto_cls=AutoModel, trust_remote_code=True
    ) as hf_model:
        hf_processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=True
        )
        device = next(hf_model.model.parameters()).device

        for text, image in zip(input_texts, input_images):
            if image is not None:
                # Image document: use process_documents with an empty text
                # string so the processor adds the passage prefix and image
                # tokens in the same format as the vLLM side.
                inputs = hf_processor.process_documents(
                    {"images": [image], "texts": [""]}
                )
            else:
                # Text query: strip the "query: " prefix we added for vLLM
                # because process_queries adds it internally.
                raw_query = text[len(COLEMBED_QUERY_PREFIX):]
                inputs = hf_processor.process_queries([raw_query])

            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
                if v is not None
            }

            with torch.inference_mode():
                last_hidden = hf_model.model(**inputs).last_hidden_state
                # Zero-out padding positions and L2-normalise each token.
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                emb = F.normalize(last_hidden * mask, p=2, dim=-1)

            # Flatten to 1-D for comparison (shape: [n_tokens * hidden_size]).
            hf_outputs.append(emb[0].reshape(-1).tolist())

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", COLEMBED_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_colembed_text(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Test ColEmbed per-token embeddings for text-only queries."""
    input_images: list[None] = [None] * len(COLEMBED_TEXT_QUERIES)
    _run_colembed_test(
        hf_runner,
        vllm_runner,
        COLEMBED_TEXT_QUERIES,
        input_images,  # type: ignore[arg-type]
        model,
        dtype=dtype,
    )


@pytest.mark.parametrize("model", COLEMBED_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_colembed_image(
    hf_runner,
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test ColEmbed per-token embeddings for image documents."""
    input_texts_images = [
        (text, asset.pil_image)
        for text, asset in zip(COLEMBED_IMAGE_PROMPTS, image_assets)
    ]
    input_texts = [text for text, _ in input_texts_images]
    input_images = [image for _, image in input_texts_images]

    _run_colembed_test(
        hf_runner,
        vllm_runner,
        input_texts,
        input_images,
        model,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# ColEmbed score/rerank (late-interaction MaxSim) tests
# ---------------------------------------------------------------------------
# These tests exercise the vLLM score API for the ColEmbed model, which uses
# ColBERT-style MaxSim scoring.
#
# The chat_template / score_template is only applied on the cross-encoder
# path; the late-interaction path tokenises the string directly.  Prefixes
# must therefore be added manually: "query: " for queries and "passage: "
# for documents.  For image documents supplied as ScoreMultiModalParam the
# prefix is prepended as an explicit text content part.
# ---------------------------------------------------------------------------


def _make_image_doc_param(image) -> ScoreMultiModalParam:
    """Build a ScoreMultiModalParam for an image document with the required
    'passage: ' prefix as a leading text part."""
    return ScoreMultiModalParam(
        content=[
            ChatCompletionContentPartTextParam(
                type="text",
                text=COLEMBED_PASSAGE_PREFIX,
            ),
            ChatCompletionContentPartImageParam(
                type="image_url",
                image_url={"url": _pil_to_data_uri(image)},
            ),
        ]
    )


@pytest.mark.parametrize("model", COLEMBED_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_colembed_late_interaction_text(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Verify score() MaxSim matches manual MaxSim from token_embed outputs."""
    query = "The label of the object is stop sign"
    document = "A red octagonal stop sign mounted on a post."

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    ) as vllm_model:
        q_outputs = vllm_model.token_embed(
            [f"{COLEMBED_QUERY_PREFIX}{query}"]
        )
        d_outputs = vllm_model.token_embed(
            [f"{COLEMBED_PASSAGE_PREFIX}{document}"]
        )

        q_emb = torch.tensor(q_outputs[0])
        d_emb = torch.tensor(d_outputs[0])
        manual_score = compute_maxsim_score(q_emb, d_emb).item()

        # The late-interaction path does not apply any score template, so
        # prefixes must be embedded directly in the strings passed to score().
        vllm_scores = vllm_model.score(
            f"{COLEMBED_QUERY_PREFIX}{query}",
            f"{COLEMBED_PASSAGE_PREFIX}{document}",
        )

    assert len(vllm_scores) == 1
    assert vllm_scores[0] == pytest.approx(manual_score, rel=0.01)


@pytest.mark.parametrize("model", COLEMBED_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_colembed_relevance_ordering(
    vllm_runner,
    model: str,
    dtype: str,
) -> None:
    """Verify that a relevant document scores higher than an irrelevant one."""
    query = "photo of a red stop sign"
    documents = [
        "A red octagonal stop sign mounted on a street post.",
        "Cherry blossom trees blooming in spring with pink flowers.",
    ]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    ) as vllm_model:
        scores = vllm_model.score(
            f"{COLEMBED_QUERY_PREFIX}{query}",
            [f"{COLEMBED_PASSAGE_PREFIX}{doc}" for doc in documents],
        )

    assert len(scores) == 2
    assert scores[0] > scores[1], (
        f"Stop sign doc should score higher than cherry blossom doc: "
        f"{scores[0]:.4f} vs {scores[1]:.4f}"
    )


@pytest.mark.parametrize("model", COLEMBED_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_colembed_score_text_query_image_docs(
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test late-interaction scoring: text query against image documents."""
    query = "photo of a red stop sign on a street"

    image_doc_params = [_make_image_doc_param(asset.pil_image) for asset in image_assets]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    ) as vllm_model:
        raw_outputs = vllm_model.llm.score(
            f"{COLEMBED_QUERY_PREFIX}{query}",
            image_doc_params,
        )

    assert len(raw_outputs) == len(image_assets)
    for output in raw_outputs:
        assert isinstance(output.outputs.score, float)


@pytest.mark.parametrize("model", COLEMBED_MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_colembed_score_mixed_docs(
    vllm_runner,
    image_assets,
    model: str,
    dtype: str,
) -> None:
    """Test late-interaction scoring: text query against mixed text+image docs.

    The text document about the stop sign should score higher than an image
    of cherry blossoms for the stop-sign query.
    """
    query = "photo of a red stop sign"
    text_doc = f"{COLEMBED_PASSAGE_PREFIX}A red octagonal stop sign mounted on a street post."
    image_doc = _make_image_doc_param(image_assets[1].pil_image)
    documents: list = [text_doc, image_doc]

    with vllm_runner(
        model,
        runner="pooling",
        dtype=dtype,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    ) as vllm_model:
        raw_outputs = vllm_model.llm.score(
            f"{COLEMBED_QUERY_PREFIX}{query}",
            documents,
        )

    assert len(raw_outputs) == 2
    scores = [o.outputs.score for o in raw_outputs]
    assert all(isinstance(s, float) for s in scores)
    assert scores[0] > scores[1], (
        f"Text doc about stop sign should score higher than cherry blossom image: "
        f"{scores[0]:.4f} vs {scores[1]:.4f}"
    )
