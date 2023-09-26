# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
import logging
import importlib
import json
import subprocess
import time

from typing import Any, Dict, Tuple
import virtualenv
from prompt_toolkit.shortcuts import radiolist_dialog, input_dialog, yes_no_dialog
from prompt_toolkit.styles import Style

BYTE_MLPERF_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(BYTE_MLPERF_ROOT)
sys.path.insert(0, BYTE_MLPERF_ROOT)

import argparse
from byte_mlperf.core.configs.workload_store import load_workload
from byte_mlperf.core.configs.dataset_store import load_dataset, get_accuracy_checker
from byte_mlperf.core.configs.model_store import load_model_config
from byte_mlperf.core.configs.backend_store import init_compile_backend, init_runtime_backend
from byte_mlperf.tools.build_pdf import build_pdf

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PerfEngine")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


class InstanceEngine:
    def __init__(self, workload, backend) -> None:
        super().__init__()
        #three format: Dict, File Path, Workload Name
        self.workload = load_workload(workload)
        self.backend_type = backend
        self.old_os_path = os.environ['PATH']
        self.prev_sys_path = list(sys.path)
        self.real_prefix = sys.prefix
        self.compile_only_mode = False

        log.info("******************* Backend Env Initization *******************")
        status = self.activate_venv(self.backend_type)
        if not status:
            log.warning("Activate virtualenv Failed, Please Check...")
        self.compile_backend = init_compile_backend(self.backend_type)
        self.runtime_backend = init_runtime_backend(self.backend_type)

        self.base_report = {
            "Model": self.workload['model'].upper(),
            "Backend": self.backend_type,
            "Host Info": self.get_cpu_name()
        }

    def get_best_batch_size(self):
        return self.compile_backend.get_best_batch_size()

    def compile(self, batch_sizes):
        # Initalize Model Config Info
        model_info = load_model_config(workload['model'])
        pre_compile_config = {"workload": workload, 'model_info': model_info}
        interact_info = self.check_interact_info(pre_compile_config)
        pre_compile_config['interact_info'] = interact_info
        if not model_info['dataset_name']:
            model_info['dataset_name'] = 'fake_dataset'
        '''
        Compile Backend could do some optimization like convert model format here
        '''
        log.info("******************************************* Running Backend Compilation... *******************************************")
        log.info("Running Backend Preoptimization...")
        pre_compile_config = self.compile_backend.pre_optimize(pre_compile_config)

        # Initalize dataset
        dataset = load_dataset(model_info)
        dataset.preprocess()
        base_report['Dataset'] = model_info['dataset_name'].upper(
        ) if model_info['dataset_name'] else None

        #Placeholder Only
        segment_info = self.compile_backend.segment(pre_compile_config)

        best_batch_sizes = self.compile_backend.get_best_batch_size()
        if isinstance(best_batch_sizes, list) and not batch_sizes:
            pre_compile_config['workload'][
                'batch_sizes'] = best_batch_sizes

        log.info("Start to compile the model...")
        start = time.time()
        compile_info = self.compile_backend.compile(pre_compile_config,
                                                    dataset)
        end = time.time()

        graph_compile_report = {}
        graph_compile_report["Compile Duration"] = round(end - start, 5)
        graph_compile_report["Compile Precision"] = compile_info[
            'compile_precision']
        graph_compile_report["Subgraph Coverage"] = compile_info['sg_percent']
        if 'optimizations' in compile_info:
            graph_compile_report['Optimizations'] = compile_info['optimizations']
        if 'instance_count' in compile_info:
            base_report['Instance Count'] = compile_info['instance_count']
        if 'device_count' in compile_info:
            base_report['Device Count'] = compile_info['device_count']
        self.base_report['Graph Compile'] = graph_compile_report
        return graph_compile_report

    def benchmark(self):
        batch_sizes = pre_compile_config['workload']['batch_sizes']
        self.runtime_backend.configs = compile_info
        self.runtime_backend.workload = workload
        self.runtime_backend.model_info = model_info

        self.runtime_backend.load(workload['batch_sizes'][0])
        # test accuracy
        accuracy_report = {}
        AccuracyChecker = get_accuracy_checker(
            model_info['dataset_name']
            if model_info['dataset_name'] else 'fake_dataset')
        AccuracyChecker.runtime_backend = self.runtime_backend
        AccuracyChecker.dataloader = dataset
        AccuracyChecker.output_dir = output_dir
        AccuracyChecker.configs = compile_info
    
    def save_compiled_model(self) -> bool:
        return True
    
    def save_report(self) -> bool:
        return True

    def get_cpu_name(self):
        command = "lscpu | grep 'Model name' | awk -F: '{print $2}'"
        cpu_name = subprocess.check_output(command, shell=True)
        return cpu_name.decode().strip()

    def activate_venv(self, hardware_type: str) -> bool:
        if os.path.exists('byte_mlperf/backends/' + hardware_type +
                          '/requirements.txt'):
            log.info("Activating Virtual Env for " + hardware_type)

            venv_dir = os.path.join("byte_mlperf/backends",
                                    hardware_type + "/venv")
            activate_file = os.path.join(venv_dir, 'bin', 'activate_this.py')
            if not os.path.exists(venv_dir):
                log.info("venv not exist, Creating Virtual Env for " +
                         hardware_type)

                virtualenv.create_environment(venv_dir)
                exec(open(activate_file).read(), {'__file__': activate_file})
                python_path = os.path.join(venv_dir, 'bin', 'python3')
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'
                ])
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '-r', 'byte_mlperf/backends/' +
                    hardware_type + '/requirements.txt', '-q'
                ])
            else:
                exec(open(activate_file).read(), {'__file__': activate_file})
                '''
                just in case install failed in pre-run.
                '''
                python_path = os.path.join(venv_dir, 'bin', 'python3')
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'
                ])
                subprocess.call([
                    python_path, '-m', 'pip', 'install', '-r', 'byte_mlperf/backends/' +
                    hardware_type + '/requirements.txt', '-q'
                ])

                if not hasattr(sys, 'real_prefix'):
                    return False
                return True
        return True

    def deactivate_venv(self):
        sys.path[:
                 0] = self.prev_sys_path  #will also revert the added site-packages
        sys.prefix = self.real_prefix
        os.environ['PATH'] = self.old_os_path

def instance(workload, backend='CPU'):
    return InstanceEngine(workload, backend)
