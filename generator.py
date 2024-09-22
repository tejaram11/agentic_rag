# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:13:57 2024

@author: TEJA
"""

from langchain.llms.base import LLM
from typing import Optional, List
from pydantic import PrivateAttr, Field
import torch
import json

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



class LocalLLM(LLM): 
    model_path: str = Field(..., description="Path to the model") 
    _llm_model: Optional[AutoModelForCausalLM] = PrivateAttr(default=None)
    _llm_tokenizer: Optional[AutoTokenizer] = PrivateAttr(default=None)
    def __init__(self,model_path):
        super().__init__(model_path=model_path)
        
        #self.model_path=model_path
        self.load_model()
        
    def load_model(self):
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        access_token = "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"
        self._llm_model = AutoModelForCausalLM.from_pretrained(              
                                                          self.model_path, 
                                                          torch_dtype=torch.float16, 
                                                          quantization_config=quantization_config,
                                                          use_auth_token=access_token,
                                                          device_map='auto'
                                                          )
        self._llm_tokenizer = AutoTokenizer.from_pretrained(
                                                        self.model_path, 
                                                        torch_dtype=torch.float16, 
                                                        quantization_config = quantization_config,
                                                        use_auth_token=access_token,
                                                        device_map='auto'
                                                        )
    def tool_call(self, query:str,tools: List,stop:Optional[List[str]]=None)->str:
        chat = [
            {"role": "user", "content": query}
            ]
        
        inputs = self._llm_tokenizer.apply_chat_template(
                                                     chat,
                                                       tools=tools,
                                                        add_generation_prompt=True,
                                                        return_tensors="pt"
                                                               ).to(self._llm_model.device)
        
        

        outputs = self._llm_model.generate(inputs, max_new_tokens=500)
        response = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        function=json.loads(response.split("assistant")[1])
        function["arguments"] = function.pop("parameters")
        
        if function['name']=="get_special_today":
            temp=tools[0]()
            return temp
        elif function['name'] == "extreme_fun_activity":
            tools[1]()
            
        
        
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Tokenize input
        inputs = self._llm_tokenizer.encode(prompt, return_tensors="pt").to(self._llm_model.device)

        # Generate output
        outputs = self._llm_model.generate(inputs, max_new_tokens=100)
        response = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        
        return response

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": "local-llm"}

    @property
    def _llm_type(self) -> str:
        return "custom"