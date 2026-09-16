"""Save repeatable model runs without overwriting the original fitted model."""
import argparse
from src.model_runs import archive_original, train_run, MODEL_NAMES

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='+',choices=list(MODEL_NAMES),default=[])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--features',type=int,default=40)
    parser.add_argument('--sampler',choices=['none','smote','smoteenn'],default='smoteenn')
    args=parser.parse_args()
    print('Original snapshot:',archive_original(),flush=True)
    for model in args.models:
        print('Saved run:',train_run(model,args.seed,args.features,args.sampler),flush=True)
