#!/bin/bash

# 初始數值
iteration=50000

# 最大值
max_iteration=60000

# 初始化最高分數
max_psnr=0
max_ssim=0
max_combined_score=0
best_psnr_iter=0
best_ssim_iter=0
best_combined_iter=0

# 迴圈執行直到達到最大值
while [ "$iteration" -le "$max_iteration" ]; do
  echo "Running with iteration: $iteration"

  # 執行命令並將 iteration 傳入參數
  CUDA_VISIBLE_DEVICES=0 bash hw4.sh /project/g/r13922043/hw4_dataset/dataset/public_test \
    ../gaussian-splatting/output/d79f13d7-4/point_cloud/iteration_${iteration}/final_test \
    $iteration \
    > ../gaussian-splatting/output/d79f13d7-4/point_cloud/iteration_${iteration}/render${iteration}_log

  CUDA_VISIBLE_DEVICES=0 python grade.py ../gaussian-splatting/output/d79f13d7-4/point_cloud/iteration_${iteration}/final_test \
    /project/g/r13922043/hw4_dataset/dataset/public_test/images/ \
    > ../gaussian-splatting/output/d79f13d7-4/point_cloud/iteration_${iteration}/score${iteration}_log

  # 提取 psnr 和 ssim 分數
  log_file="../gaussian-splatting/output/d79f13d7-4/point_cloud/iteration_${iteration}/score${iteration}_log"
  if [ -f "$log_file" ]; then
    psnr=$(grep "Testing psnr" "$log_file" | awk '{print $3}')
    ssim=$(grep "Testing ssim" "$log_file" | awk '{print $3}')
    combined_score=$(echo "$psnr + $ssim" | bc)

    echo "Iteration: $iteration | PSNR: $psnr | SSIM: $ssim | Combined Score: $combined_score"

    # 檢查並更新最高 PSNR
    if (( $(echo "$psnr > $max_psnr" | bc -l) )); then
      max_psnr=$psnr
      best_psnr_iter=$iteration
    fi

    # 檢查並更新最高 SSIM
    if (( $(echo "$ssim > $max_ssim" | bc -l) )); then
      max_ssim=$ssim
      best_ssim_iter=$iteration
    fi

    # 檢查並更新最高相加分數
    if (( $(echo "$combined_score > $max_combined_score" | bc -l) )); then
      max_combined_score=$combined_score
      best_combined_iter=$iteration
    fi
  else
    echo "Log file not found for iteration $iteration"
  fi

  # 增加 1000
  iteration=$((iteration + 1000))
done

echo "All iterations completed up to $max_iteration."
echo "Highest PSNR: $max_psnr at iteration $best_psnr_iter"
echo "Highest SSIM: $max_ssim at iteration $best_ssim_iter"
echo "Highest Combined Score: $max_combined_score at iteration $best_combined_iter"

echo "All iterations completed up to $max_iteration."