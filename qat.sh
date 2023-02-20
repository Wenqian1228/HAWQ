export CUDA_VISIBLE_DEVICES=0
python quant_train.py -a resnet50 \
                      --epochs 1 --lr 0.0001 \
                      --batch-size 128 \
                      --data /data/wqzhao/Datasets/imagenet \
                      --pretrained \
                      --save-path ./checkpoints/ \
                      --act-range-momentum=0.99 \
                      --wd 1e-4 \
                      --data-percentage 0.001 \
                      --fix-BN \
                      --checkpoint-iter -1 \
                      --quant-scheme latency_0.5
                    #   --evaluate
