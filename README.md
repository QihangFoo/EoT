# [ACL 2025] Exclusion of Thought: Mitigating Cognitive Load in Large Language Models for Enhanced Reasoning in Multiple-Choice Tasks



Multiple-choice questions (MCQs) are a widely used and vital assessment format for evaluating large language models (LLMs). This study reveals that LLMs are susceptible to "cognitive load" caused by distractor options in MCQs, leading to excessive attention to distractors and consequent vacillation between correct and incorrect options. To mitigate this cognitive burden, we introduce a novel reasoning prompt strategy, called **EoT**, which effectively reduces cognitive load by steering the model’s attention away from erroneous options. This enables the model to focus more effectively on reasonable answers. Additionally, by documenting the elimination process, EoT enhances the transparency and interpretability of the model's reasoning. Experimental results demonstrate that EoT, as a plug-and-play approach, significantly reduces cognitive load and improves performance, showcasing its potential to enhance both the accuracy and interpretability of LLMs. 





## Setup

- python 3.11, pytorch >= 2.1

- pip install -r requirement.txt

  (You may need to change the version of transformers according to the model config)

## Running



```bash
python eot_eval.py --task datast --threshold 0.3 --num_few_shot 5 --prompt_type [standard/cot/zero_shot_cot/complex_cot] --setting [eot/bbase] --p 2
```




