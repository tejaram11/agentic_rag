# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 23:19:23 2024

@author: TEJA
"""

import pdfplumber
import re

class pdf_processor():
    def __init__(self,margin=50):
        self.margin = margin
        self.final_text=[]
        self.summary=[]
        
    def process_pdf(self,pdf_path,skip_pages:list,continous_pages:list):
        with pdfplumber.open(pdf_path) as pdf:
            for i,first_page in enumerate(pdf.pages):
                if i in skip_pages:
                    continue
                if i in continous_pages:
                    text=first_page.extract_text()
                    self.final_text.append(text)
                    self.summary.append(text)
                else:
                    left_bbox = (0, 0, first_page.width / 2, first_page.height-self.margin)  # Left half of the page
                    left_text = first_page.within_bbox(left_bbox).extract_text()
            
                    right_bbox = (first_page.width / 2, 0, first_page.width, first_page.height-self.margin)  # Right half of the page
                    right_text = first_page.within_bbox(right_bbox).extract_text()
                    self.final_text.append(left_text)
                    self.final_text.append(right_text)

        return self.final_text
            
    
            
        
        

