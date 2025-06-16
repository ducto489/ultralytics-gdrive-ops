#!/usr/bin/env python3
"""
YOLOv11 Training and Testing Experiment
Dataset: data_test_1 with 4 classes (rach_long_vai, rach_tai_bien, ban_soi_lan_soi_day_soi, tuot_hang)
"""

import os
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import de_parallel
from copy import deepcopy

class YOLOv11Experiment:
    def __init__(self, data_path, experiment_name=None):
        self.data_path = Path(data_path)
        self.experiment_name = experiment_name or f"yolov11_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = Path("experiments") / self.experiment_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup experiment logging"""
        print(f"🚀 Setting up YOLOv11 experiment: {self.experiment_name}")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
    def prepare_data_config(self):
        """Prepare and update data.yaml for the experiment"""
        data_yaml_path = self.data_path / "data.yaml"
        
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")
            
        # Load existing data.yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update path to absolute path
        data_config['path'] = str(self.data_path.absolute())
        
        # Save updated config to experiment directory
        exp_data_yaml = self.results_dir / "data.yaml"
        with open(exp_data_yaml, 'w') as f:
            yaml.dump(data_config, f)
            
        print(f"✅ Data config prepared: {len(data_config['names'])} classes detected")
        for idx, name in data_config['names'].items():
            print(f"   Class {idx}: {name}")
            
        return str(exp_data_yaml)
    
    def load_training_config(self):
        """Load training configuration"""
        config_path = self.data_path / "training_config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Default config for YOLOv11
            config = {
                'pretrained_model': 'yolo11n.pt',
                'epochs': 10,
                'imgsz': 960,
                'batch': 8,
                'exist_ok': True,
                'save_period': -1,
                'fliplr': 0.8,
                'flipud': 0.6,
                'scale': 0.1,
                'patience': 200
            }
        
        # Save config to experiment directory
        with open(self.results_dir / "training_config.yaml", 'w') as f:
            yaml.dump(config, f)
            
        return config
    
    def train_model(self, data_config_path, training_config):
        """Train YOLOv11 model"""
        print(f"🏋️  Starting YOLOv11 training...")
        
        # Initialize model
        model_name = training_config.pop('pretrained_model', 'yolo11n.pt')
        print(f"📦 Loading model: {model_name}")
        model = YOLO(model_name)
        
        # Setup training arguments
        train_args = training_config.copy()
        train_args.update({
            'data': data_config_path,
            'project': str(self.results_dir),
            'name': 'training',
            'exist_ok': True,
            'save': True,
            'plots': True,
            'verbose': True
        })
        
        print(f"⚙️  Training configuration:")
        for key, value in train_args.items():
            print(f"   {key}: {value}")
        
        # Start training
        print(f"🚀 Starting training for {train_args['epochs']} epochs...")
        results = model.train(**train_args)
        
        # Save best model to experiment directory
        best_model_path = self.results_dir / "training" / "weights" / "best.pt"
        if best_model_path.exists():
            shutil.copy(best_model_path, self.results_dir / "best_model.pt")
            print(f"✅ Best model saved to: {self.results_dir / 'best_model.pt'}")
        
        return model, results
    
    def test_model(self, data_config_path):
        """Test the trained model"""
        print(f"🔍 Starting model evaluation...")
        
        best_model_path = self.results_dir / "best_model.pt"
        if not best_model_path.exists():
            print(f"❌ Best model not found at {best_model_path}")
            return None
            
        # Load best model
        model = YOLO(str(best_model_path))
        
        # Run validation
        print(f"📊 Running validation on test/val set...")
        val_results = model.val(
            data=data_config_path,
            save_json=True,
            save_hybrid=True,
            conf=0.25,
            iou=0.6,
            max_det=300,
            half=False,
            device=None,
            dnn=False,
            plots=True,
            rect=False,
            split='val'
        )
        
        # Save validation results
        results_summary = {
            'mAP50': float(val_results.results_dict.get('metrics/mAP50(B)', 0)),
            'mAP50-95': float(val_results.results_dict.get('metrics/mAP50-95(B)', 0)),
            'precision': float(val_results.results_dict.get('metrics/precision(B)', 0)),
            'recall': float(val_results.results_dict.get('metrics/recall(B)', 0)),
            'fitness': float(val_results.fitness or 0)
        }
        
        with open(self.results_dir / "test_results.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"📈 Test Results:")
        for metric, value in results_summary.items():
            print(f"   {metric}: {value:.4f}")
        
        return val_results
    
    def export_model(self, formats=['onnx']):
        """Export trained model to different formats"""
        print(f"📦 Exporting model to formats: {formats}")
        
        best_model_path = self.results_dir / "best_model.pt"
        if not best_model_path.exists():
            print(f"❌ Best model not found for export")
            return
            
        model = YOLO(str(best_model_path))
        
        for fmt in formats:
            try:
                print(f"   Exporting to {fmt}...")
                if fmt == 'onnx':
                    model.export(format=fmt, opset=12, dynamic=True)
                else:
                    model.export(format=fmt)
                print(f"   ✅ {fmt} export successful")
            except Exception as e:
                print(f"   ❌ {fmt} export failed: {e}")
    
    def run_experiment(self, export_formats=['onnx']):
        """Run complete training and testing experiment"""
        print(f"🎯 Starting YOLOv11 Experiment: {self.experiment_name}")
        print(f"📂 Data path: {self.data_path}")
        print("="*60)
        
        try:
            # 1. Prepare data configuration
            data_config_path = self.prepare_data_config()
            
            # 2. Load training configuration
            training_config = self.load_training_config()
            
            # 3. Train model
            model, train_results = self.train_model(data_config_path, training_config)
            
            # 4. Test model
            test_results = self.test_model(data_config_path)
            
            # 5. Export model
            self.export_model(export_formats)
            
            print("="*60)
            print(f"🎉 Experiment completed successfully!")
            print(f"📁 Results saved in: {self.results_dir}")
            print(f"🏆 Best model: {self.results_dir / 'best_model.pt'}")
            
            return {
                'experiment_name': self.experiment_name,
                'results_dir': str(self.results_dir),
                'train_results': train_results,
                'test_results': test_results
            }
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            raise


def main():
    """Main function to run the experiment"""
    # Dataset path
    data_path = "/hdd1t/mduc/ultralytics-gdrive-ops/logs/data_test_1"
    
    # Create and run experiment
    experiment = YOLOv11Experiment(
        data_path=data_path,
        experiment_name="yolov11_data_test_1_experiment"
    )
    
    # Run the complete experiment
    results = experiment.run_experiment(export_formats=['onnx', 'torchscript'])
    
    return results


if __name__ == "__main__":
    # Print system info
    print("🔧 System Information:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA devices: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
    print("="*60)
    
    # Run experiment
    main() 