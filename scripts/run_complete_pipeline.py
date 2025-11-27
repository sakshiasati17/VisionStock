"""
Complete End-to-End Pipeline for VisionStock Project
====================================================
This script runs the complete workflow:
1. Baseline evaluation (pre-trained YOLOv8)
2. Fine-tuning on custom annotated dataset
3. Evaluation of fine-tuned model
4. Comparison: Baseline vs Fine-tuned
5. Generate comprehensive reports
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime
import logging
import json
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from notebooks.fine_tuning import train_model
from notebooks.baseline_evaluation import evaluate_baseline_model
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompletePipeline:
    """Complete end-to-end pipeline orchestrator."""
    
    def __init__(self, config: dict):
        self.config = config
        self.project_root = project_root
        self.results_dir = project_root / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.baseline_model = config.get("baseline_model", "yolov8n.pt")
        self.custom_data_yaml = config.get("custom_data_yaml", "data/custom.yaml")
        self.sku_data_yaml = config.get("sku_data_yaml", None)  # Optional SKU-110K dataset
        
        # Training config
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 16)
        self.device = config.get("device", "cpu")
        
        # Results storage
        self.baseline_metrics = None
        self.finetuned_model_path = None
        self.finetuned_metrics = None
        self.comparison_results = None
    
    def step1_baseline_evaluation(self):
        """Step 1: Evaluate baseline pre-trained model."""
        logger.info("="*80)
        logger.info("STEP 1: BASELINE MODEL EVALUATION")
        logger.info("="*80)
        
        # For baseline: Evaluate pre-trained YOLOv8 on custom dataset
        # This establishes baseline performance before fine-tuning
        eval_data = self.custom_data_yaml if Path(self.custom_data_yaml).exists() else None
        
        logger.info(f"Loading baseline model: {self.baseline_model}")
        logger.info("📌 Baseline = Pre-trained YOLOv8 (trained on COCO) evaluated on custom dataset")
        model = YOLO(self.baseline_model)
        
        if eval_data:
            logger.info(f"Evaluating baseline on custom dataset: {eval_data}")
            logger.info("This shows how pre-trained model performs on retail shelf images")
            results = model.val(data=eval_data, imgsz=640, plots=True, save_json=True)
        else:
            logger.warning("⚠️  No custom dataset found. Using COCO validation set for baseline.")
            logger.info("💡 For proper baseline, ensure data/custom.yaml exists with your dataset")
            results = model.val(data="coco.yaml", split="val", imgsz=640)
        
        # Extract metrics
        self.baseline_metrics = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
        }
        
        # Calculate F1
        if self.baseline_metrics["precision"] + self.baseline_metrics["recall"] > 0:
            self.baseline_metrics["f1_score"] = (
                2 * (self.baseline_metrics["precision"] * self.baseline_metrics["recall"]) /
                (self.baseline_metrics["precision"] + self.baseline_metrics["recall"])
            )
        else:
            self.baseline_metrics["f1_score"] = 0.0
        
        logger.info("\n" + "="*80)
        logger.info("BASELINE EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"mAP50:     {self.baseline_metrics['mAP50']:.4f}")
        logger.info(f"mAP50-95:  {self.baseline_metrics['mAP50_95']:.4f}")
        logger.info(f"Precision: {self.baseline_metrics['precision']:.4f}")
        logger.info(f"Recall:    {self.baseline_metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {self.baseline_metrics['f1_score']:.4f}")
        logger.info("="*80)
        
        # Save baseline results
        baseline_file = self.results_dir / "baseline_metrics.json"
        with open(baseline_file, 'w') as f:
            json.dump(self.baseline_metrics, f, indent=2)
        logger.info(f"✅ Baseline metrics saved to: {baseline_file}")
        
        return self.baseline_metrics
    
    def step2_finetune_model(self):
        """Step 2: Fine-tune model on custom annotated dataset."""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FINE-TUNING MODEL ON CUSTOM DATASET")
        logger.info("="*80)
        
        if not Path(self.custom_data_yaml).exists():
            raise FileNotFoundError(f"Custom dataset YAML not found: {self.custom_data_yaml}")
        
        logger.info(f"Dataset: {self.custom_data_yaml}")
        logger.info(f"Epochs: {self.epochs}, Batch: {self.batch_size}, Device: {self.device}")
        
        # Train model
        experiment_name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        project_dir = self.results_dir / "training"
        
        results, summary = train_model(
            model_path=self.baseline_model,
            data_yaml=self.custom_data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=640,
            device=self.device,
            project=str(project_dir),
            name=experiment_name
        )
        
        # Find the best model
        best_model_path = project_dir / experiment_name / "weights" / "best.pt"
        if not best_model_path.exists():
            # Try last.pt if best.pt doesn't exist
            best_model_path = project_dir / experiment_name / "weights" / "last.pt"
        
        if not best_model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {best_model_path}")
        
        self.finetuned_model_path = str(best_model_path)
        logger.info(f"✅ Fine-tuning complete!")
        logger.info(f"✅ Best model saved to: {self.finetuned_model_path}")
        
        # Save training summary
        training_summary_file = self.results_dir / "training_summary.json"
        with open(training_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Training summary saved to: {training_summary_file}")
        
        return self.finetuned_model_path
    
    def step3_evaluate_finetuned(self):
        """Step 3: Evaluate fine-tuned model."""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: EVALUATING FINE-TUNED MODEL")
        logger.info("="*80)
        
        if not self.finetuned_model_path:
            raise ValueError("Fine-tuned model path not set. Run step2_finetune_model() first.")
        
        logger.info(f"Evaluating model: {self.finetuned_model_path}")
        logger.info(f"On dataset: {self.custom_data_yaml}")
        
        self.finetuned_metrics = self._evaluate_model(
            model_path=self.finetuned_model_path,
            data_yaml=self.custom_data_yaml
        )
        
        logger.info("\n" + "="*80)
        logger.info("FINE-TUNED MODEL EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"mAP50:     {self.finetuned_metrics['mAP50']:.4f}")
        logger.info(f"mAP50-95:  {self.finetuned_metrics['mAP50_95']:.4f}")
        logger.info(f"Precision: {self.finetuned_metrics['precision']:.4f}")
        logger.info(f"Recall:    {self.finetuned_metrics['recall']:.4f}")
        logger.info(f"F1-Score:  {self.finetuned_metrics['f1_score']:.4f}")
        logger.info("="*80)
        
        # Save fine-tuned results
        finetuned_file = self.results_dir / "finetuned_metrics.json"
        with open(finetuned_file, 'w') as f:
            json.dump(self.finetuned_metrics, f, indent=2)
        logger.info(f"✅ Fine-tuned metrics saved to: {finetuned_file}")
        
        return self.finetuned_metrics
    
    def _evaluate_model(self, model_path: str, data_yaml: str) -> dict:
        """Evaluate a model and return metrics."""
        logger.info(f"Evaluating model: {model_path}")
        model = YOLO(model_path)
        results = model.val(data=data_yaml, imgsz=640, plots=False)
        
        # Extract metrics from results
        metrics = {
            'mAP50': float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            'mAP50_95': float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
            'precision': float(results.results_dict.get("metrics/precision(B)", 0.0)),
            'recall': float(results.results_dict.get("metrics/recall(B)", 0.0)),
        }
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        return metrics
    
    def _compare_models(self, baseline_metrics: dict, finetuned_metrics: dict) -> dict:
        """Compare baseline vs fine-tuned model metrics."""
        comparison = {
            'baseline': baseline_metrics,
            'finetuned': finetuned_metrics,
            'improvements': {
                'mAP50': finetuned_metrics['mAP50'] - baseline_metrics['mAP50'],
                'mAP50_95': finetuned_metrics['mAP50_95'] - baseline_metrics['mAP50_95'],
                'precision': finetuned_metrics['precision'] - baseline_metrics['precision'],
                'recall': finetuned_metrics['recall'] - baseline_metrics['recall'],
                'f1_score': finetuned_metrics['f1_score'] - baseline_metrics['f1_score']
            }
        }
        return comparison
    
    def step4_compare_models(self):
        """Step 4: Compare baseline vs fine-tuned models."""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: COMPARING BASELINE VS FINE-TUNED MODELS")
        logger.info("="*80)
        
        if not self.baseline_metrics or not self.finetuned_metrics:
            raise ValueError("Both baseline and fine-tuned metrics required for comparison.")
        
        comparison_dir = self.results_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        # Use baseline model path (pre-trained)
        baseline_model_path = self.baseline_model
        
        self.comparison_results = self._compare_models(
            baseline_metrics=self.baseline_metrics,
            finetuned_metrics=self.finetuned_metrics
        )
        
        # Save comparison results
        comparison_file = comparison_dir / "comparison_results.json"
        with open(comparison_file, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        logger.info(f"✅ Comparison results saved to: {comparison_dir}")
        
        return self.comparison_results
    
    def step5_generate_report(self):
        """Step 5: Generate comprehensive final report."""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: GENERATING COMPREHENSIVE REPORT")
        logger.info("="*80)
        
        report = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "baseline_model": self.baseline_model,
                "custom_dataset": self.custom_data_yaml,
                "training_epochs": self.epochs,
                "batch_size": self.batch_size,
                "device": self.device
            },
            "baseline_metrics": self.baseline_metrics,
            "finetuned_metrics": self.finetuned_metrics,
            "comparison": self.comparison_results,
            "success_metrics": {
                "map_improvement_pct": self.comparison_results["improvements"]["mAP50_95_improvement_pct"] if self.comparison_results else 0,
                "map_improvement_target_met": self.comparison_results["improvements"]["mAP50_95_improvement_pct"] >= 10 if self.comparison_results else False,
                "precision_target_met": self.finetuned_metrics["precision"] >= 0.85 if self.finetuned_metrics else False,
                "recall_target_met": self.finetuned_metrics["recall"] >= 0.85 if self.finetuned_metrics else False
            }
        }
        
        report_file = self.results_dir / "complete_pipeline_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        logger.info(f"✅ Complete report saved to: {report_file}")
        logger.info(f"✅ Markdown report saved to: {self.results_dir / 'REPORT.md'}")
        
        return report
    
    def _generate_markdown_report(self, report: dict):
        """Generate human-readable markdown report."""
        md_content = f"""# VisionStock Complete Pipeline Report

**Generated:** {report['pipeline_info']['timestamp']}

## Pipeline Configuration

- **Baseline Model:** {report['pipeline_info']['baseline_model']}
- **Custom Dataset:** {report['pipeline_info']['custom_dataset']}
- **Training Epochs:** {report['pipeline_info']['training_epochs']}
- **Batch Size:** {report['pipeline_info']['batch_size']}
- **Device:** {report['pipeline_info']['device']}

## Results Summary

### Baseline Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | {report['baseline_metrics']['mAP50']:.4f} |
| mAP50-95 | {report['baseline_metrics']['mAP50_95']:.4f} |
| Precision | {report['baseline_metrics']['precision']:.4f} |
| Recall | {report['baseline_metrics']['recall']:.4f} |
| F1-Score | {report['baseline_metrics']['f1_score']:.4f} |

### Fine-tuned Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | {report['finetuned_metrics']['mAP50']:.4f} |
| mAP50-95 | {report['finetuned_metrics']['mAP50_95']:.4f} |
| Precision | {report['finetuned_metrics']['precision']:.4f} |
| Recall | {report['finetuned_metrics']['recall']:.4f} |
| F1-Score | {report['finetuned_metrics']['f1_score']:.4f} |

### Improvements

| Metric | Improvement | Improvement % |
|--------|-------------|---------------|
| mAP50 | {report['comparison']['improvements']['mAP50_improvement']:+.4f} | {report['comparison']['improvements']['mAP50_improvement_pct']:+.2f}% |
| mAP50-95 | {report['comparison']['improvements']['mAP50_95_improvement']:+.4f} | {report['comparison']['improvements']['mAP50_95_improvement_pct']:+.2f}% |
| Precision | {report['comparison']['improvements']['precision_improvement']:+.4f} | - |
| Recall | {report['comparison']['improvements']['recall_improvement']:+.4f} | - |
| F1-Score | {report['comparison']['improvements']['f1_improvement']:+.4f} | - |

## Success Metrics Check

- ✅ **mAP Improvement ≥ 10%:** {report['success_metrics']['map_improvement_target_met']} ({report['success_metrics']['map_improvement_pct']:.2f}%)
- ✅ **Precision ≥ 85%:** {report['success_metrics']['precision_target_met']} ({report['finetuned_metrics']['precision']*100:.2f}%)
- ✅ **Recall ≥ 85%:** {report['success_metrics']['recall_target_met']} ({report['finetuned_metrics']['recall']*100:.2f}%)

## Files Generated

- `baseline_metrics.json` - Baseline evaluation results
- `finetuned_metrics.json` - Fine-tuned model evaluation results
- `training_summary.json` - Training configuration and results
- `comparison/model_comparison.json` - Detailed comparison
- `comparison/model_comparison.csv` - CSV format comparison
- `complete_pipeline_report.json` - This report in JSON format

## Next Steps

1. Review the comparison results in `comparison/` directory
2. Check training plots in `training/` directory
3. Use the fine-tuned model for inference: `{self.finetuned_model_path}`
4. Integrate with FastAPI backend for production use
"""
        
        report_md_file = self.results_dir / "REPORT.md"
        with open(report_md_file, 'w') as f:
            f.write(md_content)
    
    def run_complete_pipeline(self, skip_baseline=False, skip_training=False, skip_eval=False):
        """Run the complete end-to-end pipeline."""
        logger.info("\n" + "="*80)
        logger.info("VISIONSTOCK COMPLETE END-TO-END PIPELINE")
        logger.info("="*80)
        logger.info(f"Results will be saved to: {self.results_dir}")
        logger.info("="*80)
        
        try:
            # Step 1: Baseline Evaluation
            if not skip_baseline:
                self.step1_baseline_evaluation()
            else:
                logger.info("⏭️  Skipping baseline evaluation")
                self.baseline_metrics = {
                    "mAP50": 0.0,
                    "mAP50_95": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0
                }
            
            # Step 2: Fine-tuning
            if not skip_training:
                self.step2_finetune_model()
            else:
                logger.info("⏭️  Skipping fine-tuning")
                # Try to find existing fine-tuned model
                existing_models = list(self.project_root.glob("runs/detect/*/weights/best.pt"))
                if existing_models:
                    self.finetuned_model_path = str(existing_models[-1])
                    logger.info(f"Using existing model: {self.finetuned_model_path}")
                else:
                    raise ValueError("No fine-tuned model found and training was skipped.")
            
            # Step 3: Evaluate Fine-tuned
            if not skip_eval:
                self.step3_evaluate_finetuned()
            else:
                logger.info("⏭️  Skipping fine-tuned evaluation")
                self.finetuned_metrics = self.baseline_metrics.copy()
            
            # Step 4: Compare Models
            if not skip_baseline and not skip_eval:
                self.step4_compare_models()
            else:
                logger.info("⏭️  Skipping comparison")
            
            # Step 5: Generate Report
            self.step5_generate_report()
            
            logger.info("\n" + "="*80)
            logger.info("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"📁 All results saved to: {self.results_dir}")
            logger.info(f"📊 View report: {self.results_dir / 'REPORT.md'}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Complete end-to-end pipeline for VisionStock project"
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="yolov8n.pt",
        help="Path to baseline pre-trained model (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--custom-data",
        type=str,
        default="data/custom.yaml",
        help="Path to custom dataset YAML (default: data/custom.yaml)"
    )
    parser.add_argument(
        "--sku-data",
        type=str,
        default=None,
        help="Path to SKU-110K dataset YAML (optional)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: 'cpu' or '0' for GPU (default: cpu)"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation step"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip fine-tuning step (use existing model)"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip fine-tuned evaluation step"
    )
    
    args = parser.parse_args()
    
    config = {
        "baseline_model": args.baseline_model,
        "custom_data_yaml": args.custom_data,
        "sku_data_yaml": args.sku_data,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "device": args.device
    }
    
    pipeline = CompletePipeline(config)
    pipeline.run_complete_pipeline(
        skip_baseline=args.skip_baseline,
        skip_training=args.skip_training,
        skip_eval=args.skip_eval
    )


if __name__ == "__main__":
    main()

