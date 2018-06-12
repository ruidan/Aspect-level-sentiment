
CUDA_VISIBLE_DEVICES="0" python train.py \
--domain res \
--alpha 0.1 \
--is-pretrain 1 \
--seed 1234 \

CUDA_VISIBLE_DEVICES="0" python train.py \
--domain lt \
--alpha 0.1 \
--is-pretrain 1 \
--seed 1234 \

CUDA_VISIBLE_DEVICES="0" python train.py \
--domain res_15 \
--alpha 0.1 \
--is-pretrain 1 \
--seed 1234 \

CUDA_VISIBLE_DEVICES="0" python train.py \
--domain res_16 \
--alpha 0.1 \
--is-pretrain 1 \
--seed 1234 \







