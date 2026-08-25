#!/bin/sh
set -eu

# ============================================================================
# DIRETÓRIOS COM DADOS GERADOS
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)" #ns3_experiments
RUNS_DIR="${SCRIPT_DIR}/runs"
LAST_RUN_FILE="${RUNS_DIR}/.last_prepared_run"

# Prioridade de selecao:
# 1) variavel de ambiente TARGET_DIR
# 2) argumento posicional 1
# 3) arquivo marker do step_1_prepare_run.py
TARGET_DIR="${TARGET_DIR:-${1:-}}"
if [ -z "${TARGET_DIR}" ]; then
	if [ -f "${LAST_RUN_FILE}" ]; then
		TARGET_DIR="$(head -n 1 "${LAST_RUN_FILE}" | tr -d '[:space:]')"
	fi
fi

if [ -z "${TARGET_DIR}" ]; then
	echo "Erro: TARGET_DIR nao definido."
	echo "Use: TARGET_DIR=<run_name> ./step_2_run_ns3.sh"
	echo "ou:  ./step_2_run_ns3.sh <run_name>"
	echo "ou rode primeiro o step_1_prepare_run.py para gerar ${LAST_RUN_FILE}."
	exit 1
fi

RUN_DIR="${SCRIPT_DIR}/runs/${TARGET_DIR}"
NS3_DIR="${SCRIPT_DIR}/../../ns3-sat-sim/simulator"

if [ ! -d "${RUN_DIR}" ]; then
	echo "Erro: run dir nao existe: ${RUN_DIR}"
	exit 1
fi

mkdir -p "${RUN_DIR}/logs_ns3"
cd "${NS3_DIR}"

./waf --run="main_satnet --run_dir='${RUN_DIR}'" 2>&1 | tee "${RUN_DIR}/logs_ns3/console.txt"
