from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv import Config

CFG_PATH = "./udaseg/configs/segformer_mitb0_pretrained.py" # NOTE: Eventually I think I can't load in everything from the config because I'm making too many custom changes

# def modify_config(cfg:Config, dataset_root:str, single_gpu=True):
    # """Modify config with changes specific to this workflow. Specifically:
        # - Change SyncBN to regular BN for single-gpu training"""

def main(single_gpu=True):
    cfg = Config.fromfile(CFG_PATH)


    # Model pretraining is defined in config
    model = build_segmentor(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    
    if single_gpu:
        model = revert_sync_batchnorm(model)
    cfg.gpu_ids = range(1)
    # cfg.device = get_device()

    cfg.seed = 0
    set_random_seed(0, deterministic=False)

    
    datasets = [build_dataset(cfg.data.train)]

    train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                    meta=dict())

if __name__ == '__main__':
    main()