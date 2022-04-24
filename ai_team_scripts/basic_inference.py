from mmcv import Config
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

def main():
    cfg_path = './fast_scnn_lr0.12_8x4_160k_cityscapes.py'
    checkpoint_path = 'fast_scnn_lr0.12_8x4_160k_cityscapes_20210630_164853-0cec9937.pth'

    model = init_segmentor(cfg_path, checkpoint_path, device='cuda:0')

    img_path = '/home/moritz/Documents/datasets/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'

    result = inference_segmentor(model, img_path)

    show_result_pyplot(model, img_path, result, get_palette('cityscapes'))

if __name__ == '__main__':
    main()