"""
Complete ML Pipeline for E-commerce Purchase Prediction and Recommendation System

Pipeline Sequence:
1. Data Management -> 2. Validation -> 3. Data Preprocessing -> 
4. Hyperparameter Tuning -> 5. Model Training -> 6. Next Purchase Data Prep -> 
7. Recommendation System -> 8. Market Basket Analysis -> 9. Dashboard

Author: ML Pipeline System
Created: 2025
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import json

class CompletePipelineManager:
    def __init__(self):
        self.pipeline_status = {}
        self.start_time = None
        self.steps = [
            {
                'name': 'Data Management',
                'script': 'datamanagement.py',
                'description': 'Load and merge all e-commerce datasets',
                'required_files': ['data/olist_orders_dataset.csv', 'data