"""Convert SKU-110K dataset to YOLO format and set up for evaluation."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_annotations_to_yolo(annotations_csv, images_dir, labels_dir, split_name):
    """Convert CSV annotations to YOLO format."""
    logger.info(f"Converting {split_name} annotations...")
    
    # Read CSV
    names = ['image', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height']
    df = pd.read_csv(annotations_csv, names=names)
    
    # Get unique images
    unique_images = df['image'].unique()
    logger.info(f"  Found {len(unique_images)} images")
    
    # Create labels directory
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert each image's annotations
    for img_name in tqdm(unique_images, desc=f"  Converting {split_name}"):
        img_annotations = df[df['image'] == img_name]
        
        # Get image dimensions (should be same for all annotations of same image)
        w = img_annotations.iloc[0]['image_width']
        h = img_annotations.iloc[0]['image_height']
        
        # Create label file
        label_file = labels_dir / f"{Path(img_name).stem}.txt"
        
        with open(label_file, 'w') as f:
            for _, ann in img_annotations.iterrows():
                # Convert to normalized YOLO format
                x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
                
                # Normalize coordinates
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Class is always 0 (object)
                cls = 0
                
                # Write YOLO format: class x_center y_center width height
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    logger.info(f"  ✅ Converted {len(unique_images)} images")
    return len(unique_images)


def setup_sku110k_dataset():
    """Set up SKU-110K dataset in YOLO format."""
    logger.info("="*80)
    logger.info("SETTING UP SKU-110K DATASET FOR YOLO")
    logger.info("="*80)
    
    # Source dataset
    source_dir = Path.home() / "Downloads" / "SKU110K_fixed"
    
    if not source_dir.exists():
        logger.error(f"❌ Dataset not found at: {source_dir}")
        return False
    
    # Target directory in project
    target_dir = project_root / "data" / "sku110k" / "SKU-110K"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Source: {source_dir}")
    logger.info(f"Target: {target_dir}")
    
    # Check for annotation CSV files
    annotations_dir = source_dir / "annotations"
    images_dir = source_dir / "images"
    
    if not annotations_dir.exists():
        logger.error(f"❌ Annotations directory not found: {annotations_dir}")
        return False
    
    if not images_dir.exists():
        logger.error(f"❌ Images directory not found: {images_dir}")
        return False
    
    # Find annotation CSV files
    train_csv = annotations_dir / "annotations_train.csv"
    val_csv = annotations_dir / "annotations_val.csv"
    test_csv = annotations_dir / "annotations_test.csv"
    
    logger.info("\n📊 Found annotation files:")
    logger.info(f"   Train: {train_csv.exists()}")
    logger.info(f"   Val: {val_csv.exists()}")
    logger.info(f"   Test: {test_csv.exists()}")
    
    # Create YOLO structure
    for split in ['train', 'val', 'test']:
        split_dir = target_dir / split
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Convert annotations
    if train_csv.exists():
        convert_annotations_to_yolo(
            train_csv,
            images_dir,
            target_dir / "train" / "labels",
            "train"
        )
    
    if val_csv.exists():
        convert_annotations_to_yolo(
            val_csv,
            images_dir,
            target_dir / "val" / "labels",
            "val"
        )
    
    if test_csv.exists():
        convert_annotations_to_yolo(
            test_csv,
            images_dir,
            target_dir / "test" / "labels",
            "test"
        )
    
    # Create symlinks for images (to save space - 12GB!)
    logger.info("\n📁 Creating image symlinks (to save space)...")
    for split in ['train', 'val', 'test']:
        split_csv = annotations_dir / f"annotations_{split}.csv"
        if split_csv.exists():
            df = pd.read_csv(split_csv, names=['image', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'])
            unique_images = df['image'].unique()
            
            for img_name in tqdm(unique_images, desc=f"  Linking {split} images"):
                src_img = images_dir / img_name
                dst_img = target_dir / split / "images" / img_name
                
                if src_img.exists() and not dst_img.exists():
                    # Create symlink
                    dst_img.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        dst_img.symlink_to(src_img)
                    except:
                        # If symlink fails, copy (will take more space)
                        shutil.copy2(src_img, dst_img)
    
    # Create YAML file
    yaml_path = target_dir / "data.yaml"
    yaml_content = {
        'path': str(target_dir.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['object']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"\n✅ Created YAML: {yaml_path}")
    logger.info("\n✅ Dataset setup complete!")
    logger.info(f"📁 Dataset ready at: {target_dir}")
    
    return True


def main():
    """Main function."""
    success = setup_sku110k_dataset()
    
    if success:
        logger.info("\n✅ Dataset is ready for evaluation!")
        logger.info("📊 Next: Run baseline evaluation")
        logger.info("   python scripts/evaluate_sku110k_baseline.py")
    else:
        logger.error("\n❌ Dataset setup failed")


if __name__ == "__main__":
    main()

