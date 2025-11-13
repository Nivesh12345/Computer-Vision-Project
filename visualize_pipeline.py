"""
Pipeline Visualization Tool
Generates images showing each step of the parking detection preprocessing pipeline
"""
import cv2
import pickle
import numpy as np
import os
import argparse
from pathlib import Path


class PipelineVisualizer:
    """Visualizes each step of the parking detection preprocessing pipeline"""
    
    def __init__(self, video_path=None, image_path=None, spaces_file='polygons', output_dir='pipeline_steps'):
        """
        Initialize the visualizer
        
        Args:
            video_path: Path to video file (optional)
            image_path: Path to image file (optional)
            spaces_file: Path to polygons pickle file
            output_dir: Directory to save output images
        """
        # Load frame
        if image_path:
            self.frame = cv2.imread(image_path)
            if self.frame is None:
                raise ValueError(f"Could not load image: {image_path}")
            print(f"✓ Loaded image: {image_path}")
        elif video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            success, self.frame = cap.read()
            cap.release()
            if not success:
                raise ValueError(f"Could not read frame from video: {video_path}")
            print(f"✓ Loaded frame from video: {video_path}")
        else:
            raise ValueError("Either video_path or image_path must be provided")
        
        # Load parking spaces
        try:
            with open(spaces_file, 'rb') as f:
                data = pickle.load(f)
            
            self.polygons = []
            for item in data:
                if isinstance(item, (list, tuple)):
                    self.polygons.append(np.array(item, dtype=np.int32))
                else:
                    self.polygons.append(item)
            
            print(f"✓ Loaded {len(self.polygons)} parking spaces")
        except FileNotFoundError:
            print(f"⚠ Warning: Parking spaces file not found: {spaces_file}")
            print("   Polygon masking steps will be skipped")
            self.polygons = []
        
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"✓ Output directory: {self.output_dir}")
        
        # Preprocessing parameters (matching improved_detector.py)
        self.gaussian_kernel = (3, 3)
        self.adaptive_block = 25
        self.adaptive_c = 16
        self.median_kernel = 5
        self.threshold_multiplier = 0.18
        self.min_threshold = 300
        
        # Store intermediate results
        self.steps = {}
    
    def add_text_overlay(self, img, text, position='top', font_scale=1.0, thickness=2):
        """Add text overlay to image"""
        overlay = img.copy()
        h, w = overlay.shape[:2]
        
        # Calculate text size
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        if position == 'top':
            y = text_h + 20
            x = 20
        elif position == 'bottom':
            y = h - 20
            x = 20
        else:
            y, x = position
        
        # Draw background rectangle
        cv2.rectangle(overlay, (x - 10, y - text_h - 10), 
                     (x + text_w + 10, y + baseline + 10), 
                     (0, 0, 0), -1)
        cv2.rectangle(overlay, (x - 10, y - text_h - 10), 
                     (x + text_w + 10, y + baseline + 10), 
                     (255, 255, 255), 2)
        
        # Draw text
        cv2.putText(overlay, text, (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return overlay
    
    def step1_original_frame(self):
        """Step 1: Original Frame (BGR)"""
        annotated = self.add_text_overlay(
            self.frame.copy(),
            "Step 1: Original Frame (BGR) - Shape: {}x{}x3, Range: [0-255]".format(
                self.frame.shape[1], self.frame.shape[0]
            )
        )
        filename = self.output_dir / "01_original_frame.png"
        cv2.imwrite(str(filename), annotated)
        self.steps['original'] = self.frame.copy()
        print(f"  ✓ Saved: {filename}")
        return self.frame.copy()
    
    def step2_grayscale(self):
        """Step 2: Color Conversion (BGR → Grayscale)"""
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        annotated = self.add_text_overlay(
            gray.copy(),
            "Step 2: Grayscale Conversion - Formula: 0.299*R + 0.587*G + 0.114*B, Shape: {}x{}".format(
                gray.shape[1], gray.shape[0]
            )
        )
        filename = self.output_dir / "02_grayscale.png"
        cv2.imwrite(str(filename), annotated)
        self.steps['grayscale'] = gray.copy()
        print(f"  ✓ Saved: {filename}")
        return gray.copy()
    
    def step3_gaussian_blur(self, gray):
        """Step 3: Gaussian Blur (Noise Reduction)"""
        blur = cv2.GaussianBlur(gray, self.gaussian_kernel, 1)
        annotated = self.add_text_overlay(
            blur.copy(),
            "Step 3: Gaussian Blur - Kernel: {}x{}, Sigma: 1 (Noise Reduction)".format(
                self.gaussian_kernel[0], self.gaussian_kernel[1]
            )
        )
        filename = self.output_dir / "03_gaussian_blur.png"
        cv2.imwrite(str(filename), annotated)
        self.steps['blur'] = blur.copy()
        print(f"  ✓ Saved: {filename}")
        return blur.copy()
    
    def step4_adaptive_threshold(self, blur):
        """Step 4: Adaptive Thresholding (Binarization)"""
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block,
            self.adaptive_c
        )
        annotated = self.add_text_overlay(
            thresh.copy(),
            "Step 4: Adaptive Threshold - Block: {}, C: {}, Type: BINARY_INV (Cars=White)".format(
                self.adaptive_block, self.adaptive_c
            )
        )
        filename = self.output_dir / "04_adaptive_threshold.png"
        cv2.imwrite(str(filename), annotated)
        self.steps['threshold'] = thresh.copy()
        print(f"  ✓ Saved: {filename}")
        return thresh.copy()
    
    def step5_median_blur(self, thresh):
        """Step 5: Median Blur (Speckle Removal)"""
        median = cv2.medianBlur(thresh, self.median_kernel)
        annotated = self.add_text_overlay(
            median.copy(),
            "Step 5: Median Blur - Kernel: {}x{} (Speckle Removal)".format(
                self.median_kernel, self.median_kernel
            )
        )
        filename = self.output_dir / "05_median_blur.png"
        cv2.imwrite(str(filename), annotated)
        self.steps['median'] = median.copy()
        print(f"  ✓ Saved: {filename}")
        return median.copy()
    
    def step6_dilation(self, median):
        """Step 6: Morphological Dilation (Gap Filling)"""
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(median, kernel, iterations=1)
        annotated = self.add_text_overlay(
            dilated.copy(),
            "Step 6: Morphological Dilation - Kernel: 3x3, Iterations: 1 (Gap Filling)"
        )
        filename = self.output_dir / "06_dilation.png"
        cv2.imwrite(str(filename), annotated)
        self.steps['dilated'] = dilated.copy()
        print(f"  ✓ Saved: {filename}")
        return dilated.copy()
    
    def step7_polygon_masks(self):
        """Step 7: Polygon Masks for Each Parking Space"""
        if not self.polygons:
            print("  ⚠ Skipped: No polygons loaded")
            return
        
        h, w = self.frame.shape[:2]
        
        # Create a visualization showing all masks
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i, polygon in enumerate(self.polygons):
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
            # Color each mask differently
            color = tuple(np.random.randint(0, 255, 3).tolist())
            mask_vis[mask > 0] = color
            
            # Draw polygon outline
            cv2.polylines(mask_vis, [pts], isClosed=True, color=(255, 255, 255), thickness=2)
            
            # Add space number
            cx = int(np.mean(polygon[:, 0]))
            cy = int(np.mean(polygon[:, 1]))
            cv2.putText(mask_vis, str(i + 1), (cx - 10, cy + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        annotated = self.add_text_overlay(
            mask_vis,
            "Step 7: Polygon Masks - {} Parking Spaces Defined".format(len(self.polygons))
        )
        filename = self.output_dir / "07_polygon_masks.png"
        cv2.imwrite(str(filename), annotated)
        print(f"  ✓ Saved: {filename}")
        
        # Also save individual mask overlays on original frame
        for i, polygon in enumerate(self.polygons[:min(5, len(self.polygons))]):  # Limit to first 5
            overlay = self.frame.copy()
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 255), thickness=3)
            
            cx = int(np.mean(polygon[:, 0]))
            cy = int(np.mean(polygon[:, 1]))
            cv2.putText(overlay, f"Space {i+1}", (cx - 40, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            annotated = self.add_text_overlay(
                overlay,
                "Polygon Mask - Space {}".format(i + 1)
            )
            filename = self.output_dir / "07_polygon_mask_space_{:02d}.png".format(i + 1)
            cv2.imwrite(str(filename), annotated)
    
    def step8_masked_results(self, dilated):
        """Step 8: Masked Results for Each Parking Space"""
        if not self.polygons:
            print("  ⚠ Skipped: No polygons loaded")
            return
        
        h, w = dilated.shape[:2]
        
        # Process first few spaces as examples
        num_examples = min(5, len(self.polygons))
        
        for i in range(num_examples):
            polygon = self.polygons[i]
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            
            # Apply mask
            cropped = cv2.bitwise_and(dilated, dilated, mask=mask)
            
            # Count pixels
            count = cv2.countNonZero(cropped)
            area = cv2.contourArea(pts)
            threshold = max(self.min_threshold, int(area * self.threshold_multiplier))
            is_occupied = count >= threshold
            confidence = min(1.0, count / threshold) if threshold > 0 else 0.0
            
            # Create visualization: show masked region on original frame
            vis = self.frame.copy()
            cv2.fillPoly(vis, [pts], (0, 255, 0) if not is_occupied else (0, 0, 255))
            cv2.polylines(vis, [pts], isClosed=True, 
                         color=(0, 255, 0) if not is_occupied else (0, 0, 255), 
                         thickness=3)
            
            # Add text info
            cx = int(np.mean(polygon[:, 0]))
            cy = int(np.mean(polygon[:, 1]))
            
            status_text = "FREE" if not is_occupied else "OCCUPIED"
            info_text = [
                f"Space {i+1}: {status_text}",
                f"Pixel Count: {count}",
                f"Threshold: {threshold}",
                f"Confidence: {confidence:.2f}"
            ]
            
            y_offset = cy - 40
            for j, text in enumerate(info_text):
                cv2.putText(vis, text, (cx - 80, y_offset + j * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            annotated = self.add_text_overlay(
                vis,
                "Masked Result - Space {}".format(i + 1)
            )
            filename = self.output_dir / "08_masked_result_space_{:02d}.png".format(i + 1)
            cv2.imwrite(str(filename), annotated)
            
            # Also save just the masked binary region
            cropped_vis = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
            annotated_crop = self.add_text_overlay(
                cropped_vis,
                "Masked Binary Region - Space {} (White pixels = Car)".format(i + 1)
            )
            filename_crop = self.output_dir / "08_masked_binary_space_{:02d}.png".format(i + 1)
            cv2.imwrite(str(filename_crop), annotated_crop)
        
        print(f"  ✓ Saved: {num_examples} masked result images")
    
    def step9_comparison_grid(self):
        """Step 9: Create a comparison grid showing all preprocessing steps"""
        if 'dilated' not in self.steps:
            print("  ⚠ Skipped: Preprocessing steps not complete")
            return
        
        # Resize all images to same size for grid
        h, w = self.frame.shape[:2]
        target_size = (w // 2, h // 2)  # Half size for grid
        
        images = []
        labels = []
        
        # Original
        img = cv2.resize(self.frame, target_size)
        images.append(img)
        labels.append("1. Original")
        
        # Grayscale
        if 'grayscale' in self.steps:
            img = cv2.cvtColor(cv2.resize(self.steps['grayscale'], target_size), cv2.COLOR_GRAY2BGR)
            images.append(img)
            labels.append("2. Grayscale")
        
        # Blur
        if 'blur' in self.steps:
            img = cv2.cvtColor(cv2.resize(self.steps['blur'], target_size), cv2.COLOR_GRAY2BGR)
            images.append(img)
            labels.append("3. Gaussian Blur")
        
        # Threshold
        if 'threshold' in self.steps:
            img = cv2.cvtColor(cv2.resize(self.steps['threshold'], target_size), cv2.COLOR_GRAY2BGR)
            images.append(img)
            labels.append("4. Adaptive Threshold")
        
        # Median
        if 'median' in self.steps:
            img = cv2.cvtColor(cv2.resize(self.steps['median'], target_size), cv2.COLOR_GRAY2BGR)
            images.append(img)
            labels.append("5. Median Blur")
        
        # Dilated
        if 'dilated' in self.steps:
            img = cv2.cvtColor(cv2.resize(self.steps['dilated'], target_size), cv2.COLOR_GRAY2BGR)
            images.append(img)
            labels.append("6. Dilation")
        
        # Create grid
        cols = 3
        rows = (len(images) + cols - 1) // cols
        
        grid_h = rows * target_size[1] + (rows + 1) * 10 + rows * 30  # 10px padding, 30px for labels
        grid_w = cols * target_size[0] + (cols + 1) * 10
        
        grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 50
        
        for idx, (img, label) in enumerate(zip(images, labels)):
            row = idx // cols
            col = idx % cols
            
            y_start = row * (target_size[1] + 40) + 10
            x_start = col * (target_size[0] + 10) + 10
            
            grid[y_start:y_start + target_size[1], x_start:x_start + target_size[0]] = img
            
            # Add label
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = x_start + (target_size[0] - text_w) // 2
            text_y = y_start + target_size[1] + 25
            cv2.putText(grid, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        filename = self.output_dir / "09_comparison_grid.png"
        cv2.imwrite(str(filename), grid)
        print(f"  ✓ Saved: {filename}")
    
    def generate_all(self):
        """Generate all visualization steps"""
        print("\n" + "="*60)
        print("  Generating Pipeline Visualization Images")
        print("="*60 + "\n")
        
        print("Processing preprocessing steps...")
        gray = self.step1_original_frame()
        gray = self.step2_grayscale()
        blur = self.step3_gaussian_blur(gray)
        thresh = self.step4_adaptive_threshold(blur)
        median = self.step5_median_blur(thresh)
        dilated = self.step6_dilation(median)
        
        print("\nProcessing polygon masking...")
        self.step7_polygon_masks()
        self.step8_masked_results(dilated)
        
        print("\nCreating comparison grid...")
        self.step9_comparison_grid()
        
        print("\n" + "="*60)
        print("  ✓ All visualization images generated!")
        print(f"  Output directory: {self.output_dir.absolute()}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization images for parking detection pipeline"
    )
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--spaces", type=str, default="polygons", 
                       help="Path to polygons pickle file (default: polygons)")
    parser.add_argument("--output", type=str, default="pipeline_steps",
                       help="Output directory for images (default: pipeline_steps)")
    parser.add_argument("--frame", type=int, default=0,
                       help="Frame number to extract from video (default: 0)")
    
    args = parser.parse_args()
    
    # If video provided, extract specific frame
    if args.video and not args.image:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video: {args.video}")
            return
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        success, frame = cap.read()
        cap.release()
        
        if not success:
            print(f"Error: Could not read frame {args.frame} from video")
            return
        
        # Save frame temporarily
        temp_image = "temp_frame.png"
        cv2.imwrite(temp_image, frame)
        args.image = temp_image
    
    if not args.video and not args.image:
        # Try default image
        if os.path.exists("carParkImg.png"):
            args.image = "carParkImg.png"
            print("Using default image: carParkImg.png")
        else:
            print("Error: Either --video or --image must be provided")
            parser.print_help()
            return
    
    try:
        visualizer = PipelineVisualizer(
            video_path=args.video,
            image_path=args.image,
            spaces_file=args.spaces,
            output_dir=args.output
        )
        visualizer.generate_all()
        
        # Clean up temp file
        if args.video and os.path.exists("temp_frame.png"):
            os.remove("temp_frame.png")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

