
set -euo pipefail

# ====== 1. Basic configuration ======
MODEL_NAME="path to your model"
MODEL_BASENAME="xx"
PORT=9000
TASK_SCRIPT="/evalscope/run_rule_based_task.py"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# ====== 2.  vLLM OpenAI API server ======
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting vLLM server..." \
  | tee -a "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

/path/to/evalscope/bin/python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --served-model-name "${MODEL_BASENAME}" \
  --trust-remote-code \
  --port "${PORT}" \
  > "${LOG_DIR}/vllm-${MODEL_BASENAME}-${PORT}.log" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}" >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

# ====== 3. Poll the logs and wait for vLLM to finish starting up ======
VLLM_LOG="${LOG_DIR}/vllm-${MODEL_BASENAME}-${PORT}.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') Waiting for vLLM to finish startup (checking ${VLLM_LOG})..." \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

while true; do
  if [ -f "$VLLM_LOG" ] && grep -q "Application startup complete." "$VLLM_LOG"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') vLLM startup confirmed." \
      >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"
    break
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') Still waiting for vLLM startup (checking ${VLLM_LOG})..." \
    >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"
  sleep 10
done

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting eval..." \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

/path/to/evalscope/bin/python "${TASK_SCRIPT}" \
  --model-name "${MODEL_BASENAME}" \
  --port "${PORT}" \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out" 2>&1

EVAL_STATUS=$?

# ====== 5. Cleanup: shut down the vLLM process ======
echo "$(date '+%Y-%m-%d %H:%M:%S') Killing vLLM (PID=${VLLM_PID})..." \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"
kill "${VLLM_PID}" || true

wait "${VLLM_PID}" || true

echo "$(date '+%Y-%m-%d %H:%M:%S') Done. Eval exit code: ${EVAL_STATUS}" \
  >> "${LOG_DIR}/judge-${MODEL_BASENAME}.out"

exit "${EVAL_STATUS}"
