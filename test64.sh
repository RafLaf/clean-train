
  
python main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0   --milestones 100 --batch-size 128 --preprocessing ME  --test-features /users/local/vincent/full1  --wandb brain-imt


for i in {0..63}
do
    python main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0   --milestones 100 --batch-size 128 --preprocessing ME  --test-features /users/local/vincent/f$i\1  --wandb brain-imt  --nb-of-rm 1 
done


python main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0   --milestones 100 --batch-size 128 --preprocessing ME  --test-features /users/local/vincent/full1  --wandb brain-imt --forced-class


for i in {0..63}
do
    python main.py --dataset-path /users/local/datasets --dataset miniimagenet --model resnet12 --epochs 0   --milestones 100 --batch-size 128 --preprocessing ME  --test-features /users/local/vincent/f$i\1  --wandb brain-imt  --nb-of-rm 1 --forced-class
done