for i in {0..63}
do
    python clean-train/main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0 --manifold-mixup 1 --rotations --cosine --gamma 0.9 --milestones 100 --batch-size 128 --preprocessing ME --rmclass $i --save-model /users/local/r21lafar/64/models/m$i --save-features /users/local/r21lafar/64/features/f$i --wandb brain-imt
done