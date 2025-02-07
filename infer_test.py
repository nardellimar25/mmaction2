import argparse
import os.path as osp
from operator import itemgetter
from typing import Optional, Tuple

from mmengine import Config, DictAction
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 Video Inference')
    parser.add_argument('config', help='Path to model config file')
    parser.add_argument('checkpoint', help='Path to model checkpoint file')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('label', help='Path to label file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='Device to run inference on (cpu or cuda:0)')
    parser.add_argument(
        '--out-filename', default=None, help='Path to output video with predictions')
    parser.add_argument(
        '--fps', default=30, type=int, help='Frames per second for output video')
    parser.add_argument(
        '--font-scale', default=None, type=float, help='Font scale for text in output video')
    parser.add_argument(
        '--font-color', default='white', help='Font color for text in output video')
    parser.add_argument(
        '--target-resolution', nargs=2, default=None, type=int,
        help='Target resolution (width height) for resizing frames')
    return parser.parse_args()

def visualize_output(video_path: str, out_filename: str, data_sample: str, labels: list,
                      fps: int = 30, font_scale: Optional[float] = None,
                      font_color: str = 'white', target_resolution: Optional[Tuple[int]] = None):
    """Generates an output video with recognized actions."""
    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError("Online video processing is not supported")
    
    out_type = 'gif' if osp.splitext(out_filename)[1] == '.gif' else 'video'
    visualizer = ActionVisualizer()
    visualizer.dataset_meta = dict(classes=labels)
    
    text_cfg = {'colors': font_color}
    if font_scale is not None:
        text_cfg.update({'font_sizes': font_scale})
    
    visualizer.add_datasample(
        out_filename, video_path, data_sample,
        draw_pred=True, draw_gt=False, text_cfg=text_cfg,
        fps=fps, out_type=out_type,
        out_path=osp.join('demo', out_filename),
        target_resolution=target_resolution)

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    model = init_recognizer(cfg, args.checkpoint, device=args.device)
    
    print("Running inference...")
    pred_result = inference_recognizer(model, args.video)
    pred_scores = pred_result.pred_score.tolist()
    score_tuples = sorted(enumerate(pred_scores), key=itemgetter(1), reverse=True)
    top5_label = score_tuples[:5]
    
    labels = [line.strip() for line in open(args.label).readlines()]
    results = [(labels[k[0]], k[1]) for k in top5_label]
    
    print('\nTop-5 Recognized Actions:')
    for action, score in results:
        print(f'{action}: {score:.4f}')
    
    if args.out_filename:
        visualize_output(args.video, args.out_filename, pred_result,
                         labels, fps=args.fps,
                         font_scale=args.font_scale,
                         font_color=args.font_color,
                         target_resolution=args.target_resolution)

if __name__ == '__main__':
    main()
