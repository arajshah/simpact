# Isolated AnyGoal paths for the Section 8 contact-action audit.
# Source this file; do not export these paths into the SIMPACT training env.
export ANYGOAL_ROOT=/cs/local/nbu/people/ashah264/anygoal-audit
export ANYGOAL_HOME=/cs/local/nbu/people/ashah264/anygoal-audit-home
export ANYGOAL_DATA_ROOT="${ANYGOAL_HOME}/data"
export ANYGOAL_CHECKPOINT_ROOT="${ANYGOAL_HOME}/checkpoints"
export ANYGOAL_SOURCE_ROOT="${ANYGOAL_HOME}/sources"
export ANYGOAL_CACHE_ROOT="${ANYGOAL_HOME}/cache"
export ANYGOAL_OUTPUT_ROOT="${ANYGOAL_HOME}/outputs"
export STABLEWM_HOME="${ANYGOAL_HOME}/stablewm"
export HF_HOME="${ANYGOAL_HOME}/hf"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export ANYGOAL_AUDIT_RESULTS=/cs/local/nbu/people/ashah264/simpact/results/anygoal_contact_audit_v1
export PYTHONNOUSERSITE=1
