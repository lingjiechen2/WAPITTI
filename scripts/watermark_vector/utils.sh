function wait_for_jobs_below_threshold() {
    # wait_for_jobs_below_threshold "alice" 30
    local user="${1:-zhouzhanhui}"    # Default: "zhouzhanhui" if not provided
    local max_jobs="${2:-30}"             # Default: 30 if not provided
    local current_jobs
    local sleep_interval=60       # Check every 60 seconds (adjustable)

    while true; do
        current_jobs=$(squeue -u "$user" -h | wc -l)
        if [[ "$current_jobs" -lt "$max_jobs" ]]; then
            echo "Job count for '$user' ($current_jobs) is below $max_jobs. Proceeding..."
            break
        fi
        echo "Current job count for '$user': $current_jobs (waiting for <$max_jobs)..."
        sleep "$sleep_interval"
    done
}

function get_watermark_args() {
    local watermark="$1"
    local watermark_args=""

    # --- Watermark arg switch ---
    if [ "$watermark" = "aar-k2" ]; then
        watermark_args="--watermark_type aar --aar_watermark_k 2"
    elif [ "$watermark" = "aar-k3" ]; then
        watermark_args="--watermark_type aar --aar_watermark_k 3"
    elif [ "$watermark" = "aar-k4" ]; then
        watermark_args="--watermark_type aar --aar_watermark_k 4"
    elif [ "$watermark" = "kgw-k0-gamma0.25-delta1" ]; then
        watermark_args="--watermark_type kgw --kgw_watermark_gamma 0.25 --kgw_watermark_delta 1.0 --kgw_watermark_seeding_scheme simple_0"
    elif [ "$watermark" = "kgw-k0-gamma0.25-delta2" ]; then
        watermark_args="--watermark_type kgw --kgw_watermark_gamma 0.25 --kgw_watermark_delta 2.0 --kgw_watermark_seeding_scheme simple_0"
    elif [ "$watermark" = "kgw-k1-gamma0.25-delta1" ]; then
        watermark_args="--watermark_type kgw --kgw_watermark_gamma 0.25 --kgw_watermark_delta 1.0 --kgw_watermark_seeding_scheme simple_1"
    elif [ "$watermark" = "kgw-k1-gamma0.25-delta2" ]; then
        watermark_args="--watermark_type kgw --kgw_watermark_gamma 0.25 --kgw_watermark_delta 2.0 --kgw_watermark_seeding_scheme simple_1"
    elif [ "$watermark" = "kgw-k2-gamma0.25-delta1" ]; then
        watermark_args="--watermark_type kgw --kgw_watermark_gamma 0.25 --kgw_watermark_delta 1.0 --kgw_watermark_seeding_scheme simple_2"
    elif [ "$watermark" = "kgw-k2-gamma0.25-delta2" ]; then
        watermark_args="--watermark_type kgw --kgw_watermark_gamma 0.25 --kgw_watermark_delta 2.0 --kgw_watermark_seeding_scheme simple_2"
    elif [ "$watermark" = "kth-shift1" ]; then
        watermark_args="--watermark_type kth --kth_watermark_key_len 256 --kth_watermark_num_shifts 1"
    elif [ "$watermark" = "kth-shift2" ]; then
        watermark_args="--watermark_type kth --kth_watermark_key_len 256 --kth_watermark_num_shifts 2"
    elif [ "$watermark" = "kth-shift4" ]; then
        watermark_args="--watermark_type kth --kth_watermark_key_len 256 --kth_watermark_num_shifts 4"
    elif [ "$watermark" = "kth-shift256" ]; then
        watermark_args="--watermark_type kth --kth_watermark_key_len 256 --kth_watermark_num_shifts 256"
    else
        echo "Unsupported watermark type ${watermark}." >&2
        return 1
    fi

    echo "$watermark_args"
}
