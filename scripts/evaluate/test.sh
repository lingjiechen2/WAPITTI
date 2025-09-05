# After shift 4
models_array=("$@")
prefix="your_prefix_here/"

# Add prefix to each element
models_array=( "${models_array[@]/#/$prefix}" )

# Now models_array holds the prefixed values
for m in "${models_array[@]}"; do
  echo "$m"
done
