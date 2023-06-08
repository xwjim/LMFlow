#!/usr/bin/env python
# coding=utf-8
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import json
import os
import sys
sys.path.remove(os.path.abspath(os.path.dirname(sys.argv[0])))
from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments

class Report_LLM()
    def __init__():

        parser = HfArgumentParser((ModelArguments, DatasetArguments, PipelineArguments))
        model_args, data_args, pipeline_args = parser.parse_args_into_dataclasses()

        with open (pipeline_args.deepspeed, "r") as f:
            ds_config = json.load(f)
        
        self.model = AutoModel.get_model(
            model_args, 
            tune_strategy='none', 
            ds_config=ds_config, 
            use_accelerator=pipeline_args.use_accelerator_for_evaluator
        )

    def fetch_answer(self,prompts,max_length=300,batch_size=1):

        outputs = []
        for item in prompts:
            batch_input = self.model.encode(item, return_tensors="pt",padding=True).to(device=self.local_rank)
            inputs = batch_input['input_ids']
            mask = batch_input['attention_mask']
            outputs = model.inference(inputs, max_new_tokens=300,attention_mask=mask,temperature=0.0)
            text_out = model.decode(outputs, skip_special_tokens=True)
            outputs.append(text_out)
        return outputs

if __name__ == "__main__":
    data = Report_LLM()

