python main.py --dataset miniimagenet --dataset-path /users/local/datasets/ --model resnet12 --lr 0.0005 --milestones [5,10] --gamma 0.5 --load-model /users/local/r21lafar/models/f_baseline2.pt1 --batch-size 80 --temperature 64 --epochs 10 --episodic --features-epi /users/local/r21lafar/features/mini/minifeatures1.pt11 --runs 1000 --custom-epi   --episodes-per-epoch 100 --n-runs 200 --n-shots 1 --wandb brain-imt --test-features /users/local/r21lafar/features/m_baseline2.pt1 ;
python main.py --dataset miniimagenet --dataset-path /users/local/datasets/ --model resnet12 --lr 0.0005 --milestones [5,10] --gamma 0.5 --load-model /users/local/r21lafar/models/f_baseline2.pt1 --batch-size 80 --temperature 64 --epochs 10 --episodic --features-epi /users/local/r21lafar/features/mini/minifeatures1.pt11 --runs 1000 --custom-epi   --episodes-per-epoch 100 --n-runs 200 --n-shots 1 --wandb brain-imt 