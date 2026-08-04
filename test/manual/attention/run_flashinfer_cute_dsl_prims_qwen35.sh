#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_PATH="${MODEL_PATH:-${SGLANG_ROOT}/../artifact/Qwen3.5-35B-A3B-FP8}"
TP_SIZE="${TP_SIZE:-1}"
PORT="${PORT:-31000}"
NUM_THREADS="${NUM_THREADS:-16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-8192}"
# Same model/configuration FA2 baseline on 2026-07-31: 0.970. Allow an
# absolute 0.05 regression for FP8 PRIMS prefill.
MIN_GSM8K_SCORE="${MIN_GSM8K_SCORE:-0.92}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/artifacts/flashinfer_cute_dsl_prims}"
SERVER_LOG="${OUTPUT_DIR}/server.log"
EVAL_LOG="${OUTPUT_DIR}/gsm8k_100.log"
BASE_URL="http://127.0.0.1:${PORT}"

mkdir -p "${OUTPUT_DIR}"
test -f "${MODEL_PATH}/config.json"
test -f "${MODEL_PATH}/model.safetensors.index.json"

python - <<'PY'
import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)
import cutlass.experimental
import flashinfer
workspace = torch.empty(1024, dtype=torch.uint8, device="cuda")
w = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    workspace, "NHD", backend="cute-dsl-prims"
)
assert w._backend == "cute-dsl-prims"
PY

PYTHONPATH="${SGLANG_ROOT}/python" pytest -q \
  "${SGLANG_ROOT}/test/registered/unit/server_args/test_flashinfer_cute_dsl_prims_args.py" \
  "${SGLANG_ROOT}/test/manual/attention/test_flashinfer_cute_dsl_prims_metadata.py"

# The integration branch identifies itself as FlashInfer 0.6.15 while the
# installed SGLang package gates released wheels at 0.6.15.post1. This bypass
# is scoped to this local-source test process.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export PYTHONPATH="${SGLANG_ROOT}/python"

cat >"${OUTPUT_DIR}/launch_command.txt" <<EOF
python -m sglang.launch_server --model-path ${MODEL_PATH} --attention-backend flashinfer --flashinfer-prefill-backend cute-dsl-prims --kv-cache-dtype fp8_e4m3 --page-size 64 --cuda-graph-backend-prefill disabled --default-chat-template-kwargs '{"enable_thinking":false}' --tp-size ${TP_SIZE} --mem-fraction-static ${MEM_FRACTION_STATIC} --max-total-tokens ${MAX_TOTAL_TOKENS} --host 127.0.0.1 --port ${PORT}
EOF

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --attention-backend flashinfer \
  --flashinfer-prefill-backend cute-dsl-prims \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --cuda-graph-backend-prefill disabled \
  --tp-size "${TP_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 300); do
  if curl -fsS "${BASE_URL}/health_generate" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -n 200 "${SERVER_LOG}"
    exit 1
  fi
  sleep 1
done
curl -fsS "${BASE_URL}/health_generate" >/dev/null
grep -q "prefill sub-backend=cute-dsl-prims, decode backend=fa2" "${SERVER_LOG}"
curl -fsS "${BASE_URL}/server_info" >"${OUTPUT_DIR}/server_info.json"
python - "${OUTPUT_DIR}/server_info.json" <<'PY'
import json
import pathlib
import sys

info = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert info["attention_backend"] == "flashinfer"
assert info["flashinfer_prefill_backend"] == "cute-dsl-prims"
assert info["kv_cache_dtype"] == "fp8_e4m3"
assert info["page_size"] == 64
assert info["cuda_graph_config"]["prefill"]["backend"] == "disabled"
PY

python - "${BASE_URL}" "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import sys
import urllib.request

base_url, output_dir = sys.argv[1], pathlib.Path(sys.argv[2])
url = base_url + "/v1/chat/completions"
shared = "FlashInfer SGLang shared prefix. " * 80
responses = []
for prompt in (
    "Reply with exactly: integration works",
    shared + "Reply with ONE.",
    shared + "Reply with TWO.",
):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        item = json.load(response)
    message = item["choices"][0]["message"]
    assert message["content"]
    assert not message.get("reasoning_content")
    responses.append(item)
(output_dir / "smoke_responses.json").write_text(
    json.dumps(responses, indent=2, ensure_ascii=False)
)
PY

grep -q "#cached-token: [1-9]" "${SERVER_LOG}"
WARMUP_LOG_BYTES="$(wc -c <"${SERVER_LOG}")"

python -m sglang.test.run_eval \
  --base-url "${BASE_URL}" \
  --model "${MODEL_PATH}" \
  --eval-name gsm8k \
  --api chat \
  --num-examples 100 \
  --num-threads "${NUM_THREADS}" \
  --max-tokens 512 \
  --temperature 0 \
  --chat-template-kwargs '{"enable_thinking":false}' \
  2>&1 | tee "${EVAL_LOG}"

python - "${EVAL_LOG}" "${MIN_GSM8K_SCORE}" "${OUTPUT_DIR}" <<'PY'
import json
import pathlib
import re
import sys

log_path, threshold, output_dir = pathlib.Path(sys.argv[1]), float(sys.argv[2]), pathlib.Path(sys.argv[3])
text = log_path.read_text()
score_match = re.search(r"^Score: ([0-9.]+)$", text, re.MULTILINE)
assert score_match, "GSM8K score missing"
score = float(score_match.group(1))
assert "100/100" in text, "evaluator did not complete exactly 100 examples"
assert score >= threshold, f"GSM8K score {score} is below {threshold}"
(output_dir / "metrics.json").write_text(
    json.dumps({"num_examples": 100, "score": score, "minimum": threshold}, indent=2)
)
PY

curl -fsS "${BASE_URL}/health_generate" >/dev/null
if tail -c "+$((WARMUP_LOG_BYTES + 1))" "${SERVER_LOG}" |
  grep -q "Compiling CuTe-DSL kernel"; then
  echo "Unexpected steady-state CuTe-DSL compilation in ${SERVER_LOG}" >&2
  exit 1
fi
if grep -Eqi "fallback|kernel launch error|illegal memory access" "${SERVER_LOG}"; then
  echo "Unexpected fallback or kernel failure in ${SERVER_LOG}" >&2
  exit 1
fi

git -C "${SGLANG_ROOT}" rev-parse HEAD >"${OUTPUT_DIR}/sglang_revision.txt"
git -C "${SGLANG_ROOT}/../flashinfer" rev-parse HEAD >"${OUTPUT_DIR}/flashinfer_revision.txt"
echo "FlashInfer cute-dsl-prims Qwen3.5 E2E passed; artifacts: ${OUTPUT_DIR}"
