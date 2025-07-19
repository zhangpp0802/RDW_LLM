import cv2
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from PIL import Image

class BoundaryExtractor:
    def __init__(self, image_path):
        """
        Initialize the boundary extractor with an image path.
        
        Args:
            image_path (str): Path to the black and white PNG image
        """
        self.image_path = image_path
        self.image = None
        self.binary_image = None
        self.contours = []
        self.interpolated_boundaries = []
    
    def load_image(self):
        """Load and preprocess the image."""
        # Load image using PIL first to handle PNG properly
        pil_image = Image.open(self.image_path)
        
        # Convert to grayscale if needed
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        # Convert to numpy array
        self.image = np.array(pil_image)
        
        # Create binary image (black areas = 255, white areas = 0)
        # Assuming black pixels have low values (close to 0)
        self.binary_image = np.where(self.image < 128, 255, 0).astype(np.uint8)
    
    def find_contours(self):
        """Find contours of black areas."""
        # Find contours using OpenCV
        contours, hierarchy = cv2.findContours(
            self.binary_image, 
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_NONE  # Store all boundary points
        )
        
        # Convert contours to more convenient format
        self.contours = []
        for contour in contours:
            # Remove single dimension and convert to list of (x, y) points
            points = contour.squeeze()
            if len(points.shape) == 2 and points.shape[0] > 2:  # Valid contour
                self.contours.append(points)
        
        print(f"Found {len(self.contours)} black area boundaries")
    
    def interpolate_boundary_points(self, num_points=100):
        """
        Interpolate points along each boundary.
        
        Args:
            num_points (int): Number of points to interpolate for each boundary
        """
        self.interpolated_boundaries = []
        
        for i, contour in enumerate(self.contours):
            # Calculate cumulative distance along the contour
            distances = np.zeros(len(contour))
            for j in range(1, len(contour)):
                dist = np.sqrt((contour[j][0] - contour[j-1][0])**2 + 
                              (contour[j][1] - contour[j-1][1])**2)
                distances[j] = distances[j-1] + dist
            
            # Close the contour by adding the distance back to start
            total_distance = distances[-1] + np.sqrt(
                (contour[0][0] - contour[-1][0])**2 + 
                (contour[0][1] - contour[-1][1])**2
            )
            
            # Create parameter array for interpolation
            t = distances / total_distance
            
            # Add the closing point
            contour_closed = np.vstack([contour, contour[0]])
            t_closed = np.append(t, 1.0)
            
            # Interpolate x and y coordinates separately
            try:
                # Use periodic boundary conditions for closed contours
                fx = interpolate.interp1d(t_closed, contour_closed[:, 0], 
                                        kind='cubic', assume_sorted=True)
                fy = interpolate.interp1d(t_closed, contour_closed[:, 1], 
                                        kind='cubic', assume_sorted=True)
                
                # Generate evenly spaced parameter values
                t_new = np.linspace(0, 1, num_points, endpoint=False)
                
                # Interpolate points
                x_new = fx(t_new)
                y_new = fy(t_new)
                
                interpolated_points = np.column_stack([x_new, y_new])
                # print(interpolated_points)
                self.interpolated_boundaries.append(interpolated_points)
                
                print(f"Boundary {i+1}: {len(contour)} original points -> {num_points} interpolated points")
                
            except Exception as e:
                print(f"Warning: Could not interpolate boundary {i+1}: {e}")
                # Fallback: use original points
                self.interpolated_boundaries.append(contour)
    
    def get_boundary_points(self):
        """
        Return the interpolated boundary points.
        
        Returns:
            list: List of numpy arrays, each containing interpolated points for one boundary
        """
        return self.interpolated_boundaries
    
    def visualize_results(self, show_original=True, show_interpolated=True):
        """
        Visualize the original image, detected boundaries, and interpolated points.
        
        Args:
            show_original (bool): Whether to show original contour points
            show_interpolated (bool): Whether to show interpolated points
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image with detected contours
        axes[0].imshow(self.image, cmap='gray')
        axes[0].set_title('Original Image with Detected Boundaries')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.contours)))
        
        for i, (contour, color) in enumerate(zip(self.contours, colors)):
            if show_original:
                axes[0].plot(contour[:, 0], contour[:, 1], 'o-', 
                           color=color, markersize=2, linewidth=1, 
                           label=f'Boundary {i+1} (original)')
        
        axes[0].legend()
        axes[0].set_aspect('equal')
        
        # Interpolated boundaries
        axes[1].imshow(self.image, cmap='gray')
        axes[1].set_title('Interpolated Boundary Points')
        
        for i, (interp_points, color) in enumerate(zip(self.interpolated_boundaries, colors)):
            if show_interpolated:
                axes[1].plot(interp_points[:, 0], interp_points[:, 1], 'o-', 
                           color=color, markersize=3, linewidth=2, 
                           label=f'Boundary {i+1} (interpolated)')
        
        axes[1].legend()
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    def save_boundary_points(self, output_file='boundary_points.txt'):
        """
        Save boundary points to a text file.
        
        Args:
            output_file (str): Output file path
        """
        with open(output_file, 'w') as f:
            f.write(f"# Boundary points extracted from {self.image_path}\n")
            f.write(f"# Found {len(self.interpolated_boundaries)} boundaries\n\n")
            
            for i, boundary in enumerate(self.interpolated_boundaries):
                f.write(f"# Boundary {i+1} ({len(boundary)} points)\n")
                for j, point in enumerate(boundary):
                    f.write(f"{point[0]:.3f},{point[1]:.3f}\n")
                f.write("\n")
        
        print(f"Boundary points saved to {output_file}")
    
    def save_image_with_red_points(self, output_path='output_with_red_points.png', point_size=2):
        """
        Save the original image with red boundary points overlaid.
        
        Args:
            output_path (str): Path for the output PNG file
            point_size (int): Size of the red points (radius in pixels)
        """
        # Create a color version of the original image
        if len(self.image.shape) == 2:  # Grayscale
            # Convert grayscale to RGB
            color_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        else:
            color_image = self.image.copy()
        
        # Draw red points for each interpolated boundary
        for boundary in self.interpolated_boundaries:
            for point in boundary:
                x, y = int(round(point[0])), int(round(point[1]))
                # Draw red circle (BGR format: red is (0, 0, 255))
                cv2.circle(color_image, (x, y), point_size, (255, 0, 0), -1)
        
        # Save the image
        # Convert RGB to BGR for cv2.imwrite
        if len(color_image.shape) == 3:
            color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, color_image_bgr)
        else:
            cv2.imwrite(output_path, color_image)
        
        print(f"Image with red boundary points saved to {output_path}")
    
    def create_overlay_image_pil(self, output_path='output_with_red_points.png', point_size=3):
        """
        Alternative method using PIL to save image with red points.
        
        Args:
            output_path (str): Path for the output PNG file
            point_size (int): Size of the red points (radius in pixels)
        """
        from PIL import Image, ImageDraw
        
        # Convert to PIL Image
        if len(self.image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(self.image, mode='L')
            # Convert to RGB to add colored points
            pil_image = pil_image.convert('RGB')
        else:
            pil_image = Image.fromarray(self.image)
        
        # Create drawing context
        draw = ImageDraw.Draw(pil_image)
        
        # Draw red points for each interpolated boundary
        for boundary in self.interpolated_boundaries:
            for point in boundary:
                x, y = point[0], point[1]
                # Draw red circle
                draw.ellipse([x-point_size, y-point_size, x+point_size, y+point_size], 
                           fill='red', outline='red')
        
        # Save the image
        pil_image.save(output_path, 'PNG')
        print(f"Image with red boundary points saved to {output_path}")
        
        return pil_image

# Example usage
def process_image(image_path, num_interpolated_points=100, output_image_path=None, point_size=3):
    """
    Complete pipeline to extract and interpolate boundary points, and save image with red points.
    
    Args:
        image_path (str): Path to the input PNG image
        num_interpolated_points (int): Number of points to interpolate per boundary
        output_image_path (str): Path for output PNG with red points (optional)
        point_size (int): Size of red points
    
    Returns:
        tuple: (list of interpolated boundary point arrays, PIL Image with red points)
    """
    # Create extractor
    extractor = BoundaryExtractor(image_path)
    
    # Process image
    extractor.load_image()
    extractor.find_contours()
    extractor.interpolate_boundary_points(num_interpolated_points)
    
    # Visualize results
    extractor.visualize_results()
    
    # Save text results
    extractor.save_boundary_points()
    
    # Save image with red points
    if output_image_path is None:
        output_image_path = image_path.replace('.png', '_with_red_points.png')
    
    # Use PIL method for better PNG handling
    result_image = extractor.create_overlay_image_pil(output_image_path, point_size)
    
    # Return interpolated points and the result image
    return extractor.get_boundary_points(), result_image

def process_image_simple(image_path, num_points=50, point_size=3):
    """
    Simplified function to just create the image with red boundary points.
    
    Args:
        image_path (str): Path to input PNG
        num_points (int): Number of interpolated points per boundary
        point_size (int): Size of red points
    
    Returns:
        str: Path to the output image
    """
    extractor = BoundaryExtractor(image_path)
    extractor.load_image()
    extractor.find_contours()
    extractor.interpolate_boundary_points(num_points)
    
    # Generate output filename
    output_path = image_path.replace('.png', '_with_red_points.png')
    extractor.create_overlay_image_pil(output_path, point_size)
    
    return output_path

# Example usage:
if __name__ == "__main__":
    # Replace with your image path
    image_path = "/depot/bera89/data/zhan5058/TUTR/generate_agent_points/006.png"
    
    try:
        # Method 1: Full processing with visualization and red points overlay
        boundaries, result_image = process_image(
            image_path, 
            num_interpolated_points=20, 
            output_image_path="/depot/bera89/data/zhan5058/TUTR/generate_agent_points/006_r.png",
            point_size=3
        )
        
        # Method 2: Simple processing - just create the red points image
        # output_path = process_image_simple("your_image.png", num_points=50, point_size=2)
        # print(f"Processed image saved to: {output_path}")

        #boundary points
        # boundaries
        print(np.array(boundaries).reshape(-1, 2).shape) #(7, num_interpolated_points, 2)
        
        # Access individual boundaries
        for i, boundary in enumerate(boundaries):
            print(f"Boundary {i+1} shape: {boundary.shape}")
            print(f"First 5 points: {boundary[:5]}")
            print()
            
    except FileNotFoundError:
        print(f"Please replace 'your_image.png' with the actual path to your PNG image")
    except Exception as e:
        print(f"Error processing image: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install opencv-python numpy scipy matplotlib pillow")