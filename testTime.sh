#!/bin/bash
# Compare disk access performance between ~/Desktop and current directory ($PWD)

desktop_dir="/home/user/Desktop"
media_dir="$PWD"

echo "📊 Comparing disk access times:"
echo "Desktop: $desktop_dir"
echo "Current (Media): $media_dir"
echo "--------------------------------------------"

test_file="test_io_speed.tmp"

# === Function to test write + read time ===
test_disk() {
    local path=$1
    local label=$2
    cd "$path" || { echo "❌ Cannot access $path"; return; }

    echo "▶ Testing $label..."
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1  # clear cache for fairness

    # Write test (measure sequential write speed)
    write_time=$( (time dd if=/dev/zero of=$test_file bs=1M count=1024 conv=fdatasync 2>&1) 2>&1 | grep 'copied' | awk '{print $(NF-1), $NF}')
    
    # Read test (measure sequential read speed, bypassing cache)
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1
    read_time=$( (time dd if=$test_file of=/dev/null bs=1M 2>&1) 2>&1 | grep 'copied' | awk '{print $(NF-1), $NF}')

    echo "  Write speed ($label): $write_time"
    echo "  Read speed  ($label): $read_time"
    echo "--------------------------------------------"

    rm -f "$test_file"
}

test_disk "$desktop_dir" "Desktop"
test_disk "$media_dir"   "Media (PWD)"

echo "✅ Test complete."

