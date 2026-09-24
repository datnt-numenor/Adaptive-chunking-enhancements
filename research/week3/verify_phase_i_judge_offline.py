"""Exercise the paper's judge code without credentials or provider calls.

Run in an isolated environment with DeepEval installed. The package stubs
avoid importing unrelated chunking dependencies; rag_eval.py itself is loaded
unchanged from the repository and its metric logic is exercised with a fake
model.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import types

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def _load_original_judge():
    source_root = Path(__file__).resolve().parents[2] / "src" / "adaptive_chunking"
    package = types.ModuleType("adaptive_chunking")
    package.__path__ = [str(source_root)]
    paper = types.ModuleType("adaptive_chunking.paper")
    paper.__path__ = [str(source_root / "paper")]
    sys.modules[package.__name__] = package
    sys.modules[paper.__name__] = paper

    module_name = "adaptive_chunking.paper.rag_eval"
    spec = importlib.util.spec_from_file_location(module_name, source_root / "paper" / "rag_eval.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeModel(DeepEvalBaseLLM):
    def __init__(self):
        self.prompts = []

    def get_model_name(self):
        return "offline-fake"

    def load_model(self):
        return self

    def generate(self, prompt, schema=None):
        assert schema is not None
        self.prompts.append(prompt)
        if schema.__name__ == "RetrievalCompletenessVerdict":
            return schema(completeness_level=2, reason="Both claims are present."), 0.0
        if schema.__name__ == "ReasonScore":
            return schema(score=9, reason="No contradiction.")
        raise AssertionError(f"Unexpected schema: {schema.__name__}")

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema=schema)


def main():
    assert importlib.metadata.version("deepeval") == "3.5.9"
    judge = _load_original_judge()
    model = FakeModel()
    case = LLMTestCase(
        input="Which claims are supported?",
        actual_output="Both claims.",
        expected_output="Claim A and claim B.",
        retrieval_context=["Claim A is supported.", "Claim B is supported."],
    )
    metric = judge.RetrievalCompletenessMetric(model)
    assert metric.measure(case) == 1.0
    assert metric.is_successful() is True
    assert metric.evaluation_cost == 0.0
    assert len(model.prompts) == 1
    assert "Claim A is supported.\n\nClaim B is supported." in model.prompts[0]

    # The exact original three-step configuration must be accepted by this
    # DeepEval version. Constructing GEval does not make a model request.
    g_eval = GEval(
        name="Correctness",
        model=model,
        evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
            "Lightly penalize omissions of detail, focusing on the main idea",
            "Vague language or contradicting opinions are permissible",
        ],
        async_mode=False,
    )
    assert g_eval.name == "Correctness"
    assert g_eval.measure(case, _show_indicator=False) == 0.9
    assert g_eval.is_successful() is True
    assert len(model.prompts) == 2
    print("JUDGE_OFFLINE_OK deepeval=3.5.9 completeness_score=1.0 geval_score=0.9 provider_calls=0")


if __name__ == "__main__":
    main()
