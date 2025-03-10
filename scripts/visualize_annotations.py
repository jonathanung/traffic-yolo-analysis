import os
import cv2
import tkinter as tk
from tkinter import ttk
import pandas as pd
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np

class DatasetViewer:
    def __init__(self, root, dataset_dir: str):
        self.root = root
        self.root.title("LISA Traffic Light Dataset Viewer")
        
        # Dataset paths
        self.dataset_dir = Path(dataset_dir)
        self.current_sequence = None
        self.current_image_idx = 0
        
        # Load annotations
        self.annotations = self.load_annotations()
        
        # Setup UI
        self.setup_ui()
        
        # Load first image
        self.load_sequences()
        
    def load_annotations(self) -> pd.DataFrame:
        """Load and combine all annotation files"""
        annotations = []
        
        # Recursively find all annotation CSV files
        for anno_file in self.dataset_dir.rglob("*BOX.csv"):
            try:
                df = pd.read_csv(anno_file, sep=';')  # LISA dataset uses semicolon separator
                print(f"Columns in {anno_file}: {df.columns.tolist()}")  # Debug print
                
                # Add sequence name to dataframe
                df['sequence'] = anno_file.parent.name
                annotations.append(df)
            except Exception as e:
                print(f"Error loading {anno_file}: {e}")
                
        if not annotations:
            raise Exception("No annotation files found!")
            
        combined_df = pd.concat(annotations, ignore_index=True)
        print("Combined DataFrame columns:", combined_df.columns.tolist())  # Debug print
        return combined_df
    
    def setup_ui(self):
        """Setup the user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Sequence selector
        ttk.Label(main_frame, text="Sequence:").grid(row=0, column=0, sticky=tk.W)
        self.sequence_cb = ttk.Combobox(main_frame, state="readonly")
        self.sequence_cb.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.sequence_cb.bind('<<ComboboxSelected>>', self.on_sequence_changed)
        
        # Image display
        self.canvas = tk.Canvas(main_frame, width=800, height=600)
        self.canvas.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Navigation buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, columnspan=2)
        
        ttk.Button(btn_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Fast cycling toggle
        self.is_cycling = False
        self.cycle_direction = 1  # 1 for forward, -1 for backward
        self.cycle_btn = ttk.Button(btn_frame, text="Start Cycling", command=self.toggle_cycling)
        self.cycle_btn.pack(side=tk.LEFT, padx=20)
        
        # Image counter label
        self.counter_label = ttk.Label(btn_frame, text="")
        self.counter_label.pack(side=tk.LEFT, padx=20)
        
        # Key bindings
        self.root.bind('<Left>', lambda e: self.start_cycling(-1))
        self.root.bind('<Right>', lambda e: self.start_cycling(1))
        self.root.bind('<KeyRelease-Left>', lambda e: self.stop_cycling())
        self.root.bind('<KeyRelease-Right>', lambda e: self.stop_cycling())
        self.root.bind('<space>', lambda e: self.toggle_cycling())
        
    def load_sequences(self):
        """Load available sequences"""
        sequences = self.annotations['sequence'].unique()
        self.sequence_cb['values'] = sorted(sequences)
        if sequences.size > 0:
            self.sequence_cb.set(sequences[0])
            self.on_sequence_changed()
    
    def on_sequence_changed(self, event=None):
        """Handle sequence change"""
        self.current_sequence = self.sequence_cb.get()
        self.current_image_idx = 0
        self.load_current_image()
    
    def load_current_image(self):
        """Load and display current image with annotations"""
        if not self.current_sequence:
            return
            
        # Get annotations for current sequence
        sequence_annos = self.annotations[self.annotations['sequence'] == self.current_sequence]
        
        # Use 'Filename' or 'filename' based on what's in the CSV
        filename_col = 'Filename' if 'Filename' in sequence_annos.columns else 'filename'
        unique_frames = sorted(sequence_annos[filename_col].unique())
        
        if not unique_frames:
            print(f"No frames found for sequence {self.current_sequence}")
            return
            
        # Update current image if needed
        if self.current_image_idx >= len(unique_frames):
            self.current_image_idx = 0
        elif self.current_image_idx < 0:
            self.current_image_idx = len(unique_frames) - 1
            
        current_frame = unique_frames[self.current_image_idx]
        
        # Update counter label
        self.counter_label.config(text=f"Image {self.current_image_idx + 1} of {len(unique_frames)}")
        
        # Extract sequence name from the current sequence
        # Handle different naming conventions
        if 'dayClip' in self.current_sequence or 'nightClip' in self.current_sequence:
            # For dayClip/nightClip format
            sequence_name = self.current_sequence
        else:
            # For other formats
            sequence_name = self.current_sequence.replace('frameAnnotationsBOX', '')
        
        # Remove 'dayTest/' or 'dayTraining/' from the frame filename
        current_frame_clean = current_frame.split('/')[-1] if '/' in current_frame else current_frame
        
        # Try different possible paths based on the observed directory structure
        possible_paths = [
            # Standard paths
            Path("data/raw/lisa") / sequence_name / sequence_name / "frames" / current_frame_clean,
            Path("data/raw/lisa") / sequence_name / "frames" / current_frame_clean,
            
            # dayTrain/nightTrain specific paths
            Path("data/raw/lisa/dayTrain/dayTrain") / sequence_name / "frames" / current_frame_clean,
            Path("data/raw/lisa/nightTrain/nightTrain") / sequence_name / "frames" / current_frame_clean,
            
            # Other possible paths
            Path("data/raw/lisa/images") / sequence_name / current_frame_clean,
            Path("data/raw/lisa") / sequence_name / current_frame_clean,
            
            # Try with original filename (without cleaning)
            Path("data/raw/lisa/dayTrain/dayTrain") / sequence_name / "frames" / current_frame,
            Path("data/raw/lisa/nightTrain/nightTrain") / sequence_name / "frames" / current_frame
        ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if not img_path:
            print(f"Image not found: Tried paths for {current_frame_clean} in sequence {sequence_name}")
            for path in possible_paths:
                print(f"  - {path}")
            
            # Display a blank image instead of failing
            img = np.zeros((300, 400, 3), dtype=np.uint8)
            img = cv2.putText(img, "Image not found", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Read image and annotations
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to load image: {img_path}")
                # Display a blank image instead of failing
                img = np.zeros((300, 400, 3), dtype=np.uint8)
                img = cv2.putText(img, "Failed to load image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Continue with the image we have (either loaded or blank)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Only try to draw annotations if we have a valid image
        if img is not None and img.size > 0:
            # Get annotations for this specific frame
            frame_annos = sequence_annos[sequence_annos[filename_col] == current_frame]
            
            # Print debug info
            print(f"Found {len(frame_annos)} annotations for frame {current_frame}")
            if len(frame_annos) > 0:
                print(f"Sample annotation: {frame_annos.iloc[0].to_dict()}")
            
            # Draw bounding boxes
            for _, anno in frame_annos.iterrows():
                try:
                    # Adjust column names based on actual CSV structure
                    x1 = int(float(anno['Upper left corner X']))
                    y1 = int(float(anno['Upper left corner Y']))
                    x2 = int(float(anno['Lower right corner X']))
                    y2 = int(float(anno['Lower right corner Y']))
                    
                    # Draw rectangle with thicker line
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Add label with better visibility
                    label = anno['Annotation tag']
                    cv2.putText(img, str(label), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 0, 255), 2)
                    
                    print(f"Drew box at ({x1},{y1})-({x2},{y2}) with label {label}")
                except Exception as e:
                    print(f"Error drawing annotation: {e}")
                    print(f"Problematic annotation: {anno}")
        
        # Resize image to fit canvas if needed
        canvas_width = self.canvas.winfo_width() or 800  # Default if not yet rendered
        canvas_height = self.canvas.winfo_height() or 600  # Default if not yet rendered
        
        # Ensure we have valid dimensions for resizing
        if img.shape[0] > 0 and img.shape[1] > 0 and canvas_width > 0 and canvas_height > 0:
            canvas_ratio = canvas_width / canvas_height
            img_ratio = img.shape[1] / img.shape[0]
            
            if img_ratio > canvas_ratio:
                # Fit to width
                new_width = canvas_width
                new_height = int(new_width / img_ratio)
            else:
                # Fit to height
                new_height = canvas_height
                new_width = int(new_height * img_ratio)
            
            # Ensure dimensions are at least 1 pixel
            new_width = max(1, new_width)
            new_height = max(1, new_height)
            
            img = cv2.resize(img, (new_width, new_height))
        
        # Convert to PhotoImage and display
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
    def next_image(self):
        """Show next image"""
        self.current_image_idx += 1
        self.load_current_image()
        
    def prev_image(self):
        """Show previous image"""
        self.current_image_idx -= 1
        self.load_current_image()

    def toggle_cycling(self):
        """Toggle fast cycling mode"""
        if self.is_cycling:
            self.stop_cycling()
        else:
            self.start_cycling(1)  # Start cycling forward

    def start_cycling(self, direction):
        """Start cycling through images"""
        self.is_cycling = True
        self.cycle_direction = direction
        self.cycle_btn.config(text="Stop Cycling")
        self.cycle_images()

    def stop_cycling(self):
        """Stop cycling through images"""
        self.is_cycling = False
        self.cycle_btn.config(text="Start Cycling")

    def cycle_images(self):
        """Cycle through images at high speed"""
        if not self.is_cycling:
            return
        
        # Update index without full processing
        if self.cycle_direction > 0:
            self.current_image_idx += 1
        else:
            self.current_image_idx -= 1
        
        # Get sequence annotations
        sequence_annos = self.annotations[self.annotations['sequence'] == self.current_sequence]
        filename_col = 'Filename' if 'Filename' in sequence_annos.columns else 'filename'
        unique_frames = sorted(sequence_annos[filename_col].unique())
        
        # Handle index wrapping
        if self.current_image_idx >= len(unique_frames):
            self.current_image_idx = 0
        elif self.current_image_idx < 0:
            self.current_image_idx = len(unique_frames) - 1
        
        # Update counter only
        self.counter_label.config(text=f"Image {self.current_image_idx + 1} of {len(unique_frames)}")
        
        # Only load every 3rd image during fast cycling to reduce processing load
        if self.current_image_idx % 3 == 0:
            self.load_current_image_fast()
        
        # Schedule the next update
        self.root.after(50, self.cycle_images)

    def load_current_image_fast(self):
        """Lightweight version of load_current_image for fast cycling"""
        if not self.current_sequence:
            return
        
        # Get annotations for current sequence
        sequence_annos = self.annotations[self.annotations['sequence'] == self.current_sequence]
        filename_col = 'Filename' if 'Filename' in sequence_annos.columns else 'filename'
        unique_frames = sorted(sequence_annos[filename_col].unique())
        
        if not unique_frames or self.current_image_idx >= len(unique_frames):
            return
        
        current_frame = unique_frames[self.current_image_idx]
        
        # Extract sequence name from the current sequence
        # Handle different naming conventions
        if 'dayClip' in self.current_sequence or 'nightClip' in self.current_sequence:
            # For dayClip/nightClip format
            sequence_name = self.current_sequence
        else:
            # For other formats
            sequence_name = self.current_sequence.replace('frameAnnotationsBOX', '')
        
        # Remove 'dayTest/' or 'dayTraining/' from the frame filename
        current_frame_clean = current_frame.split('/')[-1] if '/' in current_frame else current_frame
        
        # Try different possible paths based on the observed directory structure
        possible_paths = [
            # Standard paths
            Path("data/raw/lisa") / sequence_name / sequence_name / "frames" / current_frame_clean,
            Path("data/raw/lisa") / sequence_name / "frames" / current_frame_clean,
            
            # dayTrain/nightTrain specific paths
            Path("data/raw/lisa/dayTrain/dayTrain") / sequence_name / "frames" / current_frame_clean,
            Path("data/raw/lisa/nightTrain/nightTrain") / sequence_name / "frames" / current_frame_clean,
            
            # Other possible paths
            Path("data/raw/lisa/images") / sequence_name / current_frame_clean,
            Path("data/raw/lisa") / sequence_name / current_frame_clean,
            
            # Try with original filename (without cleaning)
            Path("data/raw/lisa/dayTrain/dayTrain") / sequence_name / "frames" / current_frame,
            Path("data/raw/lisa/nightTrain/nightTrain") / sequence_name / "frames" / current_frame
        ]
        
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if not img_path:
            return
        
        # Read image with reduced size for faster processing
        img = cv2.imread(str(img_path))
        if img is None:
            return
        
        # Store original dimensions before resizing
        original_height, original_width = img.shape[:2]
        
        # Reduce processing - use smaller image size during fast cycling
        target_width, target_height = 640, 480
        img = cv2.resize(img, (target_width, target_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Only draw a few annotations during fast cycling
        frame_annos = sequence_annos[sequence_annos[filename_col] == current_frame]
        for i, (_, anno) in enumerate(frame_annos.iterrows()):
            # Only process first 3 annotations for speed
            if i >= 3:
                break
            
            try:
                # Get original coordinates
                x1 = int(float(anno['Upper left corner X']))
                y1 = int(float(anno['Upper left corner Y']))
                x2 = int(float(anno['Lower right corner X']))
                y2 = int(float(anno['Lower right corner Y']))
                
                # Scale coordinates for the resized image
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label (optional during fast cycling)
                label = anno['Annotation tag']
                cv2.putText(img, str(label), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 1)
            except Exception as e:
                # Silently ignore errors during fast cycling
                pass
        
        # Display the image
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Process UI events to keep the interface responsive
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    viewer = DatasetViewer(root, "data/raw/lisa/Annotations/Annotations")
    root.mainloop()

if __name__ == "__main__":
    main() 