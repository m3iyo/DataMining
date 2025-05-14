import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

DATASET_PATH = "FracAtlas/images" 
IMAGE_SIZE = (224, 224)
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
PCA_N_COMPONENTS = 0.95 
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
MODEL_FILENAME = "bone_fracture_svm_model.joblib"
SCALER_FILENAME = "scaler.joblib"
PCA_FILENAME = "pca_transformer.joblib"
METRICS_FILENAME = "training_metrics.txt" 
CONFUSION_MATRIX_IMG_FILENAME = "confusion_matrix.png" 

# Helper funvtions
def load_images_from_folder(folder_path: str) -> tuple[list[np.ndarray], list[int]]:
    """
    Loads images and their corresponding labels from subfolders.
    Assumes subfolders are named after class labels (e.g., 'fractured', 'non_fractured').

    Args:
        folder_path: Path to the root dataset directory.

    Returns:
        A tuple containing a list of images (as numpy arrays) and a list of labels (integers).
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path)) # ['fractured', 'non_fractured']
    label_map = {name: i for i, name in enumerate(class_names)}

    print(f"Loading images from: {folder_path}")
    print(f"Found classes: {class_names} mapped to {label_map}")

    for class_name in class_names:
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            try:
                # Load image in grayscale directly
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label_map[class_name])
                else:
                    print(f"Warning: Could not read image {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    print(f"Loaded {len(images)} images.")
    return images, labels, class_names

def preprocess_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray | None:
    """
    Resizes and normalizes a grayscale image.

    Args:
        image: The input grayscale image (numpy array).
        target_size: The target size (width, height) for resizing.

    Returns:
        The preprocessed image (numpy array) or None if image is invalid.
    """
    if image is None:
        return None
    # Resize
    img_resized = cv2.resize(image, target_size)
    # Normalize to [0, 1] range
    img_normalized = img_resized / 255.0
    return img_normalized

def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extracts HOG features from a single image.

    Args:
        image: The input preprocessed image.

    Returns:
        The HOG feature vector.
    """
    features = hog(image,
                   orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   block_norm='L2-Hys', # Common normalization
                   visualize=False, #
                   transform_sqrt=True, 
                   feature_vector=True) 
    return features

if __name__ == "__main__":
    # Load Data
    if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
        print(f"Error: Dataset directory '{DATASET_PATH}' is empty or does not exist.")
        print("Please ensure your FracAtlas dataset is organized with subdirectories for each class (e.g., 'images/fractured', 'images/non_fractured').")
        exit()

    images, labels, class_names_list = load_images_from_folder(DATASET_PATH)

    if not images:
        print("No images were loaded. Exiting.")
        exit()

    # Preprocess Images
    print("Preprocessing images...")
    processed_images = [preprocess_image(img, IMAGE_SIZE) for img in images]
    # Filter out any images that failed preprocessing
    valid_indices = [i for i, img in enumerate(processed_images) if img is not None]
    processed_images = [processed_images[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]

    if not processed_images:
        print("No images remained after preprocessing. Exiting.")
        exit()

    # Extract HOG Features
    print("Extracting HOG features...")
    hog_features = np.array([extract_hog_features(img) for img in processed_images])
    print(f"HOG features extracted. Shape: {hog_features.shape}")

    # Feature Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(hog_features)
    joblib.dump(scaler, SCALER_FILENAME)
    print(f"Scaler saved to {SCALER_FILENAME}")

    # Dimensionality Reduction with PCA
    print(f"Applying PCA to retain {PCA_N_COMPONENTS*100}% of variance...")
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
    pca_features = pca.fit_transform(scaled_features)
    print(f"PCA applied. Shape after PCA: {pca_features.shape}")
    print(f"Number of components selected by PCA: {pca.n_components_}")
    joblib.dump(pca, PCA_FILENAME)
    print(f"PCA transformer saved to {PCA_FILENAME}")

    # Split Data into Training and Testing sets
    print(f"Splitting data into training and testing sets (test_size={TEST_SPLIT_SIZE})...")
    X_train, X_test, y_train, y_test = train_test_split(
        pca_features,
        np.array(labels), 
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE,
        stratify=np.array(labels) 
    )
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

    # Handle Class Imbalance with SMOTE (on training data only)
    print("Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"Shape after SMOTE: X_train: {X_train_smote.shape}, y_train: {y_train_smote.shape}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_smote)}")

    # Train SVM Classifier (Optimized for speed - using fixed hyperparameters)
    print("Training SVM classifier with fixed hyperparameters (optimized for speed)...")
    best_svm_model = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced', random_state=RANDOM_STATE, probability=True)
    
    best_svm_model.fit(X_train_smote, y_train_smote)
    print(f"Using fixed SVM parameters: C={best_svm_model.C}, gamma='{best_svm_model.gamma}', kernel='{best_svm_model.kernel}'")

    # Save the Trained Model
    joblib.dump(best_svm_model, MODEL_FILENAME)
    print(f"Trained SVM model saved to {MODEL_FILENAME}")

    # Evaluate the Model on the Test Set
    print("\n--- Model Evaluation on Test Set ---")
    y_pred = best_svm_model.predict(X_test)

    print("\nClassification Report:")
    if 'class_names_list' in locals() and len(class_names_list) == len(np.unique(np.concatenate((y_test, y_pred)))):
        target_names_for_report = class_names_list
    else:
        unique_labels = np.unique(np.concatenate((y_test, y_pred)))
        target_names_for_report = [f"Class {i}" for i in unique_labels]
        print(f"Warning: Using generic class names for report: {target_names_for_report}")

    classification_report_str = classification_report(y_test, y_pred, target_names=target_names_for_report)
    print(classification_report_str)

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save Metrics to File
    print(f"\nSaving evaluation metrics to {METRICS_FILENAME}...")
    try:
        with open(METRICS_FILENAME, 'w') as f:
            f.write("--- Model Configuration ---\n")
            f.write(f"PCA Components Retained: {pca.n_components_} (explaining {PCA_N_COMPONENTS*100}% variance)\n")
            f.write(f"SVM Parameters: C={best_svm_model.C}, gamma='{best_svm_model.gamma}', kernel='{best_svm_model.kernel}', class_weight='{best_svm_model.class_weight}'\n\n")
            
            f.write("--- Evaluation Metrics ---\n")
            f.write("Classification Report:\n")
            f.write(classification_report_str)
            f.write("\n\nConfusion Matrix:\n")
            f.write(np.array2string(cm, separator=', '))
        print(f"Metrics successfully saved to {METRICS_FILENAME}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    # Display Confusion Matrix Plot and Save to File
    print(f"\nGenerating and saving confusion matrix plot to {CONFUSION_MATRIX_IMG_FILENAME}...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names_for_report, yticklabels=target_names_for_report)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Bone Fracture Detection (SVM+HOG)')
    plt.tight_layout()
    try:
        plt.savefig(CONFUSION_MATRIX_IMG_FILENAME)
        print(f"Confusion matrix plot saved to {CONFUSION_MATRIX_IMG_FILENAME}")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot. If running in a headless environment, this is expected. Error: {e}")

    print("\n--- Script Finished ---")
    print(f"To use the trained model for prediction on new images:")
    print(f"1. Load the model: model = joblib.load('{MODEL_FILENAME}')")
    print(f"2. Load the scaler: scaler = joblib.load('{SCALER_FILENAME}')")
    print(f"3. Load the PCA transformer: pca = joblib.load('{PCA_FILENAME}')")
    print(f"4. Preprocess the new image (resize to {IMAGE_SIZE}, grayscale, normalize).")
    print(f"5. Extract HOG features.")
    print(f"6. Scale features using the loaded scaler: scaled_features = scaler.transform(hog_features.reshape(1, -1))")
    print(f"7. Transform features using PCA: pca_features = pca.transform(scaled_features)")
    print(f"8. Predict: prediction = model.predict(pca_features)")
