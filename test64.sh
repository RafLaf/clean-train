
  
python main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0   --milestones 100 --batch-size 128 --preprocessing ME  --test-features /users/local/vincent/full1  --wandb brain-imt


python main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0   --milestones 100 --batch-size 128 --preprocessing ME  --test-features /users/local/vincent/f11  --wandb brain-imt  --nb-of-rm 1 



python main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0   --milestones 100 --batch-size 128 --preprocessing ME  --test-features /users/local/vincent/full1  --wandb brain-imt --forced-class



python main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0   --milestones 100 --batch-size 128 --preprocessing ME  --test-features /users/local/vincent/f11  --wandb brain-imt  --nb-of-rm 1 --forced-class
