# -*- coding: utf-8 -*-
# @Time     :2024/10/21 18:50
# @Author   :qihangfu

import os
import argparse

import logging
from eot_data_init import (
    prepare_eval,
)
from eot_utils import (
    get_accuracy,
    save_results,
    eval_all_samples
)
from api_utils import IChatAPI

logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'



def main():
    logging.basicConfig(
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    threshold = args.threshold
    model_name = args.model_name
    api = IChatAPI(model_name)

    args.threshold = threshold
    (subjects, prepare_few_shot_samples, prepare_eval_samples, prepare_eval_fn) = prepare_eval(args)
    metrics = None
    for subject in subjects:
        if os.path.exists(f'{args.save_path}/{subject}.jsonl'):
            logger.info(f"Results already exist: {args.save_path}/{subject}.jsonl")
            continue

        logger.info(f"{model_name}-Preparing: {subject}")
        few_shot_samples = prepare_few_shot_samples(subject)
        eval_samples = prepare_eval_samples(subject)
        eval_fn = prepare_eval_fn(args=args, api=api, few_shot_samples=few_shot_samples)

        logger.info(f"Run started: {subject}")
        results = eval_all_samples(
            eval_fn, eval_samples,
            name=f'{args.task},{args.num_few_shot},{args.setting},{subject}, {threshold}',
            threads=10
        )

        if len(results) > 0:
            metrics = {'type': 'metric', 'data': {}}
            metrics['data']['accuracy'] = get_accuracy(results)
            logger.info("Final report:")
            for key, value in metrics['data'].items():
                logger.info(f"{key}: {value}")
        logger.info(f"Run completed: {subject}")

        save_results(f'{args.save_path}/{subject}.jsonl', results, metrics)
        logger.info(f"Results saved: {subject}")


if __name__ == '__main__':
    main()
