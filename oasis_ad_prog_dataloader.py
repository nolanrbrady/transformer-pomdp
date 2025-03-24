import nibabel as nib
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from transformers import T5Tokenizer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torchvision import transforms
import torchio as tio
from sklearn.model_selection import train_test_split

class LongitudinalOASISLoader:
    def __init__(self, batch_size=16, test_size=0.15, val_size=0.15, random_state=42):
        self.csv_path = "../OASIS-2/OASIS-2_Varied_Verbose_Descriptions.csv"
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = None
        self._init_transforms()
        
    def _init_transforms(self):
        self.train_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),  # Ensure consistent spacing
            tio.RescaleIntensity((-1, 1)),  # Normalize to [-1,1]
            tio.RandomAffine(scales=(0.9, 1.1), degrees=10),  # Data augmentation
            tio.CropOrPad((256, 256, 256)),  # Standard size
        ])
        
        self.val_transform = tio.Compose([
            tio.ToCanonical(),
            tio.Resample(1),
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad((256, 256, 256)),
        ])

    def _load_mri(self, path, transform):
        """
        Loads an MRI file using TorchIO and applies the given transform.
        Added try/except to catch and log errors.
        Adjusts the path to use ../OASIS-2 if needed.
        """
        try:
            # If the path does not already start with '../', prepend it.
            if not path.startswith("../"):
                path = "../" + path
            subject = tio.Subject(
                mri=tio.ScalarImage(path)
            )
            transformed = transform(subject)
            return transformed.mri.data.float()
        except Exception as e:
            print(f"Error loading MRI at {path}: {e}")
            raise

    def _preprocess_features(self, df, is_train=False):
        print(df.columns)
        # Handle missing values
        df['SES'] = df['SES'].fillna(df['SES'].median())
        
        # Encode categorical variables
        df['M/F'] = df['M/F'].map({'M': 0, 'F': 1})
        df['Hand'] = df['Hand'].map({'R': 0, 'L': 1, 'A': 2})
        
        # Normalize continuous features
        cont_features = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
        
        if is_train:
            self.scaler = StandardScaler()
            df[cont_features] = self.scaler.fit_transform(df[cont_features])
        else:
            df[cont_features] = self.scaler.transform(df[cont_features])
            
        return df

    def _create_sequence_pairs(self, df):
        """
        Create input/output pairs from a longitudinal DataFrame.
        Both input and output features now contain the same set of 8 features.
        """
        df = df.sort_values(['Subject ID', 'MR Delay'])
        pairs = []
        
        for subj_id, group in df.groupby('Subject ID'):
            if len(group) < 2:
                continue
                
            for i in range(len(group)-1):
                current = group.iloc[i]
                future = group.iloc[i+1]
                
                pairs.append({
                    'input_mri': current['MRI-URL'],
                    'input_features': current[['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']].values,
                    'demographics': current[['M/F', 'Hand']].values,
                    'output_mri': future['MRI-URL'],
                    # Use the same set of features for output as for input:
                    'output_features': future[['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']].values,
                    'time_delta': future['MR Delay'] - current['MR Delay'],
                    'subject_id': subj_id
                })
            
        return pd.DataFrame(pairs).copy()

    def _train_val_test_split(self, pairs_df):
        """
        Splits the pairs DataFrame into train, validation, and test sets based on unique subjects.
        """
        unique_subjects = pairs_df['subject_id'].unique()
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size + self.val_size, 
                                random_state=self.random_state)
        train_idx, temp_idx = next(gss.split(unique_subjects, groups=unique_subjects))
        
        test_val_subjects = unique_subjects[temp_idx]
        test_val_size = self.test_size / (self.test_size + self.val_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=test_val_size, 
                                 random_state=self.random_state)
        val_idx, test_idx = next(gss2.split(test_val_subjects, groups=test_val_subjects))
        
        train_subjects = unique_subjects[train_idx]
        val_subjects = test_val_subjects[val_idx]
        test_subjects = test_val_subjects[test_idx]
        
        return (
            pairs_df[pairs_df['subject_id'].isin(train_subjects)],
            pairs_df[pairs_df['subject_id'].isin(val_subjects)],
            pairs_df[pairs_df['subject_id'].isin(test_subjects)]
        )

    def load_data(self):
        """
        Loads the CSV, preprocesses missing values and categorical features,
        creates longitudinal pairs, splits the data into train/val/test,
        and scales the continuous features based only on the training set.
        Ensures that no NaN values remain using mean imputation for continuous
        features and mode imputation for categorical features.
        """
        # Load and preprocess base data
        df = pd.read_csv(self.csv_path)
        df = df[df['Group'] != 'Converted'].copy()  # Remove converted cases
        
        # Define the continuous features we want to use
        cont_features = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']
        # Ensure continuous columns are numeric and impute missing values with the column mean
        for col in cont_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].mean(), inplace=True)
        
        # For categorical variables, fill missing values using the mode
        if df['M/F'].isnull().any():
            df['M/F'].fillna(df['M/F'].mode()[0], inplace=True)
        if df['Hand'].isnull().any():
            df['Hand'].fillna(df['Hand'].mode()[0], inplace=True)
            
        # Encode categorical variables
        df['M/F'] = df['M/F'].map({'M': 0, 'F': 1})
        df['Hand'] = df['Hand'].map({'R': 0, 'L': 1, 'A': 2})
        
        # Create sequence pairs with original values
        pairs_df = self._create_sequence_pairs(df)
        
        # Split data by subject
        train_df, val_df, test_df = self._train_val_test_split(pairs_df)
        
        # Normalize continuous features after split
        # Both input and output features now include the same 8 columns.
        self.scaler = StandardScaler()
        
        # Fit only on the training set
        train_inputs = np.vstack(train_df['input_features'])
        train_df['input_features'] = self.scaler.fit_transform(train_inputs).tolist()
        train_outputs = np.vstack(train_df['output_features'])
        train_df['output_features'] = self.scaler.transform(train_outputs).tolist()
        
        # Apply the same scaling to validation and test sets
        for df_ in [val_df, test_df]:
            df_['input_features'] = self.scaler.transform(np.vstack(df_['input_features'])).tolist()
            df_['output_features'] = self.scaler.transform(np.vstack(df_['output_features'])).tolist()
        
        return train_df, val_df, test_df

    def create_dataset(self, df, is_train=False):
        """
        Creates a PyTorch Dataset from the preprocessed DataFrame.
        Captures the loader instance so that the _load_mri method can be invoked.
        """
        transform = self.train_transform if is_train else self.val_transform
        loader_instance = self  # Capture the loader instance
        
        class LongitudinalDataset(torch.utils.data.Dataset):
            def __init__(self, df, transform):
                # Reset index to avoid index issues during batching
                self.df = df.reset_index(drop=True)
                self.transform = transform
                
            def __len__(self):
                return len(self.df)
                
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                
                # Load MRI scans using the loader's _load_mri method
                input_mri = loader_instance._load_mri(row['input_mri'], self.transform)
                output_mri = loader_instance._load_mri(row['output_mri'], self.transform)
                
                # Explicitly convert features to numeric numpy arrays before converting to torch tensors.
                input_features = torch.tensor(np.array(row['input_features'], dtype=np.float32), dtype=torch.float)
                output_features = torch.tensor(np.array(row['output_features'], dtype=np.float32), dtype=torch.float)
                demographics = torch.tensor(np.array(row['demographics'], dtype=np.int64), dtype=torch.long)
                time_delta = torch.tensor([row['time_delta']], dtype=torch.float)
                
                return {
                    'input': {
                        'mri': input_mri,
                        'features': input_features,
                        'demographics': demographics,
                    },
                    'output': {
                        'mri': output_mri,
                        'features': output_features,
                        'time_delta': time_delta,
                    }
                }
        
        return LongitudinalDataset(df, transform)
    

batch_size = 16
# Initialize loader
loader = LongitudinalOASISLoader(batch_size=batch_size)

# Load and preprocess data
train_df, val_df, test_df = loader.load_data()

# Create PyTorch datasets
train_ds = loader.create_dataset(train_df, is_train=True)
val_ds = loader.create_dataset(val_df)
test_ds = loader.create_dataset(test_df)

# Create dataloaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

for batch in train_loader:
    print(batch['input']['features'])
    break