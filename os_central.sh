#!/bin/bash

# Configuration parameters
DATA_DIR="Experiment"
TEMPERATURE=0
START_ID=-1
NUM_RUNS=2
RUN_NAME="cabinet_Llama-3-8B-Instruct_central_self_organization"
TSTEPS=10
TASK="cabinet"
OUTPUT_MODE="action_and_path"
COMM_MODE="centralized"
CONTROL_FREQ=10
SKIP_DISPLAY=false
DIRECT_WAYPOINTS=3
NUM_REPLANS=5
CONT=false
LOAD_RUN_NAME="cabinet"
LOAD_RUN_ID=0
MAX_FAILED_WAYPOINTS=1
DEBUG_MODE=false
USE_WELD=1
REL_POSE=false
SPLIT_PARSED_PLANS=false
NO_HISTORY=false
NO_FEEDBACK=false
LLM_SOURCE="Your-LLM-Model-Path"

# Run the experiment with the specified parameters
python os_centralized.py \
    --data_dir "$DATA_DIR" \
    --temperature "$TEMPERATURE" \
    --start_id "$START_ID" \
    --num_runs "$NUM_RUNS" \
    --run_name "$RUN_NAME" \
    --tsteps "$TSTEPS" \
    --task "$TASK" \
    --output_mode "$OUTPUT_MODE" \
    --comm_mode "$COMM_MODE" \
    --control_freq "$CONTROL_FREQ" \
    $( [ "$SKIP_DISPLAY" = false ] && echo "--skip_display" ) \
    --direct_waypoints "$DIRECT_WAYPOINTS" \
    --num_replans "$NUM_REPLANS" \
    $( [ "$CONT" = true ] && echo "--cont" ) \
    --load_run_name "$LOAD_RUN_NAME" \
    --load_run_id "$LOAD_RUN_ID" \
    --max_failed_waypoints "$MAX_FAILED_WAYPOINTS" \
    $( [ "$DEBUG_MODE" = true ] && echo "--debug_mode" ) \
    --use_weld "$USE_WELD" \
    $( [ "$REL_POSE" = true ] && echo "--rel_pose" ) \
    $( [ "$SPLIT_PARSED_PLANS" = true ] && echo "--split_parsed_plans" ) \
    $( [ "$NO_HISTORY" = true ] && echo "--no_history" ) \
    $( [ "$NO_FEEDBACK" = true ] && echo "--no_feedback" ) \
    --llm_source "$LLM_SOURCE"
