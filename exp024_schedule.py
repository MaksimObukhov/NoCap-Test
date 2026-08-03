"""Frozen absolute-token batch schedule for exp024."""


TOKEN_CLOCK_BATCH = 262_144
TARGET_TOKENS = 2_700_083_200
WARMUP_TOKENS = 144_965_632
WARMDOWN_TOKENS = 579_862_528
VALIDATION_INTERVAL_TOKENS = 67_108_864
SAVE_INTERVAL_TOKENS = 134_217_728
MILESTONE_START_TOKENS = 2_120_220_672
MILESTONE_INTERVAL_TOKENS = 67_108_864

# (exclusive token boundary, gradient-accumulation steps)
PHASES = (
    (201_326_592, 4),
    (469_762_048, 8),
    (939_524_096, 16),
    (TARGET_TOKENS, 32),
)


def phase_index_at(tokens_seen):
    if not 0 <= tokens_seen < TARGET_TOKENS:
        raise ValueError(f"tokens_seen outside training interval: {tokens_seen}")
    for index, (boundary, _accumulation) in enumerate(PHASES):
        if tokens_seen < boundary:
            return index
    raise AssertionError("unreachable")


def accumulation_steps_at(tokens_seen):
    return PHASES[phase_index_at(tokens_seen)][1]


def validate_schedule(micro_batch_tokens=16_384):
    previous_boundary = 0
    cumulative_updates = 0
    records = []
    for boundary, accumulation in PHASES:
        effective_batch = micro_batch_tokens * accumulation
        phase_tokens = boundary - previous_boundary
        if phase_tokens <= 0 or phase_tokens % effective_batch != 0:
            raise ValueError("phase boundary is not aligned to its effective batch")
        updates = phase_tokens // effective_batch
        cumulative_updates += updates
        records.append(
            {
                "start_tokens": previous_boundary,
                "end_tokens": boundary,
                "accumulation": accumulation,
                "effective_batch_tokens": effective_batch,
                "updates": updates,
                "cumulative_updates": cumulative_updates,
            }
        )
        previous_boundary = boundary
    if previous_boundary != TARGET_TOKENS:
        raise ValueError("schedule does not end at the target token budget")
    return records

