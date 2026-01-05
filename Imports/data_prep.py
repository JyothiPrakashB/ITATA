from datasets import Dataset
from datasets.dataset_dict import DatasetDict
import ast
import json
import pandas as pd


class DatasetLoader:
    def __init__(self, train_df_id=None, test_df_id=None, val_df_id=None, 
                 train_df_ood=None, test_df_ood=None, val_df_ood=None, sample_size = 1):
        
        self.train_df_id = train_df_id.sample(frac = sample_size, random_state = 1999) if train_df_id is not None else train_df_id
        self.test_df_id = test_df_id
        self.train_df_ood = train_df_ood
        self.test_df_ood = test_df_ood
        self.val_df_id = val_df_id
        self.val_df_ood = val_df_ood

    def set_data_for_training_semeval_with_validation(self, tokenize_function):
        """Create datasets with validation split for training and evaluation"""
    
        # Create datasets dictionary
        id_ds = DatasetDict()
        if self.train_df_id is not None:
            id_ds['train'] = Dataset.from_pandas(self.train_df_id)
        if hasattr(self, 'val_df_id') and self.val_df_id is not None:
            id_ds['validation'] = Dataset.from_pandas(self.val_df_id)
        if self.test_df_id is not None:
            id_ds['test'] = Dataset.from_pandas(self.test_df_id)
    
        # Tokenize all datasets
        id_tokenized_ds = id_ds.map(tokenize_function, batched=True)
    
        # Include empty OOD datasets for compatibility with existing code
        ood_ds = DatasetDict()
        ood_tokenized_ds = None
    
        return id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds

    def reconstruct_strings(self, df, col):
        """
        Reconstruct strings to dictionaries when loading csv/xlsx files.
        """
        reconstructed_col = []
        for text in df[col]:
            if text != '[]' and isinstance(text, str):
                text = text.replace('[', '').replace(']', '').replace('{', '').replace('}', '').split(", '")
                req_list = []
                for idx, pair in enumerate(text):
                    splitter = ': ' if ': ' in pair else ':'
                    if idx%2==0:
                        reconstructed_dict = {} 
                        reconstructed_dict[pair.split(splitter)[0].replace("'", '')] = pair.split(splitter)[1].replace("'", '')
                    else:
                        reconstructed_dict[pair.split(splitter)[0].replace("'", '')] = pair.split(splitter)[1].replace("'", '')
                        req_list.append(reconstructed_dict)
            else:
                req_list = text
            reconstructed_col.append(req_list)
        df[col] = reconstructed_col
        return df

    def extract_rowwise_aspect_polarity(self, df, on, key, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries. 
        Do same for polarities and create new column for the same.
        """
        try:
            df.iloc[0][on][0][key]
        except:
            df = self.reconstruct_strings(df, on)

        df['len'] = df[on].apply(lambda x: len(x))
        if min_val is not None:
            df.loc[df['len'] == 0, 'len'] = min_val
        df = df.loc[df.index.repeat(df['len'])]
        df['record_idx'] = df.groupby(df.index).cumcount()
        df['aspect'] = df[[on, 'record_idx']].apply(lambda x : (x[0][x[1]][key], x[0][x[1]]['polarity']) if len(x[0]) != 0 else ('',''), axis=1)
        df['polarity'] = df['aspect'].apply(lambda x: x[-1])
        df['aspect'] = df['aspect'].apply(lambda x: x[0])
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df
    
    def extract_rowwise_aspect_opinions(self, df, aspect_col, opinion_col, key, min_val = None):
        """
        Create duplicate records based on number of aspect term labels in the dataset.
        Extract each aspect term for each row for reviews with muliple aspect term entries. 
        Do same for polarities and create new column for the same.
        """
        df['len'] = df[aspect_col].apply(lambda x: len(x))
        if min_val is not None:
            df.loc[df['len'] == 0, 'len'] = min_val
        df = df.loc[df.index.repeat(df['len'])]
        df['record_idx'] = df.groupby(df.index).cumcount()
        df['aspect'] = df[[aspect_col, 'record_idx']].apply(lambda x : x[0][x[1]][key] if len(x[0]) != 0 else '', axis=1)
        df['opinion_term'] = df[[opinion_col, 'record_idx']].apply(lambda x : x[0][x[1]][key] if len(x[0]) != 0 else '', axis=1)
        df['aspect'] = df['aspect'].apply(lambda x: ' '.join(x))
        df['opinion_term'] = df['opinion_term'].apply(lambda x: ' '.join(x))
        df = df.drop(['len', 'record_idx'], axis=1).reset_index(drop = True)
        return df

    def create_data_in_ate_format(self, df, key, text_col, aspect_col, bos_instruction = '', 
                    eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, aspect_col)
        df['labels'] = df[aspect_col].apply(lambda x: ', '.join([i[key] for i in x]))
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df

    def create_data_in_atsc_format(self, df, on='aspectTerms', key='term', 
                               text_col='raw_text', aspect_col='term', 
                               bos_instruction='', delim_instruction='', 
                               eos_instruction=''):
        # If on is None, return the original DataFrame
        if on is None:
            return df
        
        # Function to safely parse and extract aspect terms
        def extract_aspect_info(aspect_terms_str):
            try:
                # Parse the string representation of the list of dictionaries
                aspect_terms = ast.literal_eval(aspect_terms_str)
                
                # If it's an empty list or 'noaspectterm', return empty strings
                if (not aspect_terms or 
                    (len(aspect_terms) == 1 and 
                    aspect_terms[0].get('term') == 'noaspectterm')):
                    return '', 'none'
                
                # Extract terms and polarities
                terms = [item.get(key, '') for item in aspect_terms]
                polarities = [item.get('polarity', 'none') for item in aspect_terms]
                
                # Join multiple terms and take the first polarity
                return ', '.join(terms), polarities[0]
            
            except (ValueError, SyntaxError):
                # If parsing fails, return the original string and 'none' polarity
                return str(aspect_terms_str), 'none'
        
        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Extract aspects and polarities
        df[aspect_col], df['polarity'] = zip(*df[on].apply(extract_aspect_info))
        
        # Create the formatted text
        df['text'] = (bos_instruction + df[text_col] + 
                    delim_instruction + df[aspect_col] + 
                    eos_instruction)
        
        # Rename polarity column to labels
        df = df.rename(columns={'polarity': 'labels'})
        
        return df

    def create_data_in_aspe_format(self, df, key, label_key, text_col, aspect_col, bos_instruction = '', 
                                         eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        try:
            df.iloc[0][aspect_col][0][key]
        except:
            df = self.reconstruct_strings(df, aspect_col)
        df['labels'] = df[aspect_col].apply(lambda x: ', '.join([f"{i[key]}:{i[label_key]}" for i in x]))
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df
    
    def create_data_in_aooe_format(self, df, aspect_col, opinion_col, key, text_col, 
                               bos_instruction = '', delim_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        if df is None:
            return
        df = self.extract_rowwise_aspect_opinions(df, aspect_col=aspect_col, opinion_col=opinion_col, key=key, min_val=1)
        df['text'] = df[[text_col, 'aspect']].apply(lambda x: bos_instruction + x[0] + delim_instruction + x[1] + eos_instruction, axis=1)
        df = df.rename(columns = {'opinion_term': 'labels'})
        return df
    
    def create_data_in_aope_format(self, df, key, text_col, aspect_col, opinion_col,
                                         bos_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        df['labels'] = df[[aspect_col, opinion_col]].apply(lambda x: ', '.join([f"{' '.join(i[key])}:{' '.join(j[key])}" for i, j in zip(x[0], x[1])]), axis=1)
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df
    
    def create_data_in_aoste_format(self, df, key, label_key, text_col, aspect_col, opinion_col,
                                         bos_instruction = '', eos_instruction = ''):
        """
        Prepare the data in the input format required.
        """
        label_map = {'POS':'positive', 'NEG':'negative', 'NEU':'neutral'}
        df['labels'] = df[[aspect_col, opinion_col]].apply(lambda x: ', '.join([f"{' '.join(i[key])}:{' '.join(j[key])}:{label_map[i[label_key]]}" for i, j in zip(x[0], x[1])]), axis=1)
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)
        return df
    
    def create_data_in_joint_task_format(self, df, term_key, polarity_key, text_col, aspect_col, 
                                    bos_instruction='', eos_instruction=''):
        """
        Prepare the data in the input format required for joint aspect term extraction and sentiment classification.
        """
        if df is None:
            return
            
        try:
            df.iloc[0][aspect_col][0][term_key]
        except:
            df = self.reconstruct_strings(df, aspect_col)
            
        # Format labels as "term:polarity" pairs
        df['labels'] = df[aspect_col].apply(
            lambda x: ', '.join([f"{i[term_key]}:{i[polarity_key]}" for i in x])
        )
        
        # Format input text with instructions
        df['text'] = df[text_col].apply(lambda x: bos_instruction + x + eos_instruction)

        return df

    @staticmethod
    def load_jsonl_data(file_path):
        """
        Load JSONL file and return as pandas DataFrame.

        Args:
            file_path: Path to JSONL file

        Returns:
            pd.DataFrame with columns from JSONL records
        """
        records = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    def create_data_in_dimasr_format(self, df, text_col='Text', quadruplet_col='Quadruplet',
                                      aspect_col='Aspect', bos_instruction='',
                                      delim_instruction='', eos_instruction='',
                                      is_train=True):
        """
        Prepare data for DimASR task (Dimensional Aspect Sentiment Regression).

        For training: Expands Quadruplet/Triplet entries to one row per aspect with VA labels.
        For inference: Expands Aspect list to one row per aspect (no labels).

        Args:
            df: Input DataFrame (from JSONL)
            text_col: Column name for review text (default: 'Text')
            quadruplet_col: Column name for Quadruplet/Triplet data (default: 'Quadruplet')
            aspect_col: Column name for aspect list (default: 'Aspect')
            bos_instruction: Beginning of sequence instruction
            delim_instruction: Delimiter instruction (e.g., " The aspect is ")
            eos_instruction: End of sequence instruction
            is_train: Whether this is training data (has VA labels)

        Returns:
            DataFrame with 'text', 'labels', 'ID', 'aspect', 'original_text' columns
        """
        if df is None:
            return None

        expanded_rows = []

        for idx, row in df.iterrows():
            text = row[text_col]
            record_id = row['ID']

            # Check for training data formats (Quadruplet or Triplet)
            if is_train and (quadruplet_col in df.columns or 'Triplet' in df.columns):
                # Determine which column to use
                data_col = quadruplet_col if quadruplet_col in df.columns else 'Triplet'
                data_items = row.get(data_col, [])

                if isinstance(data_items, str):
                    data_items = json.loads(data_items)

                if data_items is None or len(data_items) == 0:
                    continue

                for item in data_items:
                    aspect = item.get('Aspect', 'NULL')
                    va = item.get('VA', '5.00#5.00')

                    # Format input text with instructions
                    formatted_text = bos_instruction + text + delim_instruction + str(aspect) + eos_instruction

                    expanded_rows.append({
                        'ID': record_id,
                        'text': formatted_text,
                        'labels': va,  # e.g., "7.12#6.88"
                        'aspect': aspect,
                        'original_text': text
                    })
            else:
                # Inference data: extract from Aspect list
                aspects = row.get(aspect_col, [])

                if isinstance(aspects, str):
                    try:
                        aspects = json.loads(aspects)
                    except json.JSONDecodeError:
                        aspects = [aspects]

                if aspects is None:
                    aspects = []

                for aspect in aspects:
                    formatted_text = bos_instruction + text + delim_instruction + str(aspect) + eos_instruction

                    expanded_rows.append({
                        'ID': record_id,
                        'text': formatted_text,
                        'labels': '',  # No labels for inference
                        'aspect': aspect,
                        'original_text': text
                    })

        return pd.DataFrame(expanded_rows)

    def _generate_reasoning_template(self, va_string, aspect):
        """
        Generate synthetic reasoning text based on VA values for CoT training.

        Args:
            va_string: VA string in format "X.XX#X.XX"
            aspect: The aspect term

        Returns:
            Reasoning template string
        """
        try:
            parts = va_string.strip().split('#')
            valence = float(parts[0]) if len(parts) > 0 else 5.0
            arousal = float(parts[1]) if len(parts) > 1 else 5.0
        except (ValueError, IndexError):
            valence, arousal = 5.0, 5.0

        # Valence reasoning
        if valence >= 7.5:
            v_sentiment = "strong positive sentiment"
            v_desc = "strongly positive"
        elif valence >= 6.5:
            v_sentiment = "positive sentiment"
            v_desc = "positive"
        elif valence >= 5.5:
            v_sentiment = "mildly positive sentiment"
            v_desc = "slightly positive"
        elif valence >= 4.5:
            v_sentiment = "neutral sentiment"
            v_desc = "neutral"
        elif valence >= 3.5:
            v_sentiment = "mildly negative sentiment"
            v_desc = "slightly negative"
        elif valence >= 2.5:
            v_sentiment = "negative sentiment"
            v_desc = "negative"
        else:
            v_sentiment = "strong negative sentiment"
            v_desc = "strongly negative"

        # Arousal reasoning
        if arousal >= 7.5:
            a_intensity = "high emotional intensity"
            a_desc = "high"
        elif arousal >= 6.5:
            a_intensity = "elevated emotional engagement"
            a_desc = "moderate-high"
        elif arousal >= 5.5:
            a_intensity = "moderate emotional engagement"
            a_desc = "moderate"
        elif arousal >= 4.5:
            a_intensity = "neutral emotional state"
            a_desc = "neutral"
        elif arousal >= 3.5:
            a_intensity = "calm emotional state"
            a_desc = "low-moderate"
        else:
            a_intensity = "very calm, subdued emotional state"
            a_desc = "low"

        reasoning = f"The text expresses {v_sentiment} about the {aspect}. The emotional tone shows {a_intensity}.\n"
        reasoning += f"Valence: {v_desc} (~{valence:.1f}). Arousal: {a_desc} (~{arousal:.1f}).\n"
        reasoning += f"output: {va_string}"

        return reasoning

    def create_data_in_dimasr_cot_format(self, df, text_col='Text', quadruplet_col='Quadruplet',
                                          aspect_col='Aspect', bos_instruction='',
                                          delim_instruction='', eos_instruction='',
                                          is_train=True):
        """
        Prepare data for DimASR task with Chain of Thought format.

        For training: Generates input with CoT reasoning template as labels.
        For inference: Same as regular format but with reasoning prompt.

        Args:
            df: Input DataFrame (from JSONL)
            text_col: Column name for review text (default: 'Text')
            quadruplet_col: Column name for Quadruplet data (default: 'Quadruplet')
            aspect_col: Column name for aspect list (default: 'Aspect')
            bos_instruction: Beginning of sequence instruction (CoT version)
            delim_instruction: Delimiter instruction (e.g., " The aspect is ")
            eos_instruction: End of sequence instruction (e.g., ".\\nreasoning:")
            is_train: Whether this is training data (has VA labels)

        Returns:
            DataFrame with 'text', 'labels', 'ID', 'aspect', 'original_text' columns
        """
        if df is None:
            return None

        expanded_rows = []

        for idx, row in df.iterrows():
            text = row[text_col]
            record_id = row['ID']

            # Check for training data formats (Quadruplet or Triplet)
            if is_train and (quadruplet_col in df.columns or 'Triplet' in df.columns):
                data_col = quadruplet_col if quadruplet_col in df.columns else 'Triplet'
                data_items = row.get(data_col, [])

                if isinstance(data_items, str):
                    data_items = json.loads(data_items)

                if data_items is None or len(data_items) == 0:
                    continue

                for item in data_items:
                    aspect = item.get('Aspect', 'NULL')
                    va = item.get('VA', '5.00#5.00')

                    # Format input text with CoT instructions
                    formatted_text = bos_instruction + text + delim_instruction + str(aspect) + eos_instruction

                    # Generate CoT reasoning as label
                    cot_label = self._generate_reasoning_template(va, aspect)

                    expanded_rows.append({
                        'ID': record_id,
                        'text': formatted_text,
                        'labels': cot_label,  # Full CoT format with reasoning
                        'va': va,  # Store raw VA for metrics
                        'aspect': aspect,
                        'original_text': text
                    })
            else:
                # Inference data: extract from Aspect list
                aspects = row.get(aspect_col, [])

                if isinstance(aspects, str):
                    try:
                        aspects = json.loads(aspects)
                    except json.JSONDecodeError:
                        aspects = [aspects]

                if aspects is None:
                    aspects = []

                for aspect in aspects:
                    formatted_text = bos_instruction + text + delim_instruction + str(aspect) + eos_instruction

                    expanded_rows.append({
                        'ID': record_id,
                        'text': formatted_text,
                        'labels': '',  # No labels for inference
                        'va': '',
                        'aspect': aspect,
                        'original_text': text
                    })

        return pd.DataFrame(expanded_rows)

    def set_data_for_training_semeval(self, tokenize_function):
        """
        Create the training and test dataset as huggingface datasets format.
        """
        # Define train and test sets
        dataset_dict_id, dataset_dict_ood = {}, {}

        if self.train_df_id is not None:
            dataset_dict_id['train'] = Dataset.from_pandas(self.train_df_id)
        if self.test_df_id is not None:
            dataset_dict_id['test'] = Dataset.from_pandas(self.test_df_id)
        if self.val_df_id is not None:
            dataset_dict_id['validation'] = Dataset.from_pandas(self.val_df_id)
        if len(dataset_dict_id) > 1:
            indomain_dataset = DatasetDict(dataset_dict_id)
            indomain_tokenized_datasets = indomain_dataset.map(tokenize_function, batched=True)
        else:
            indomain_dataset = {}
            indomain_tokenized_datasets = {}

        if self.train_df_ood is not None:
            dataset_dict_ood['train'] = Dataset.from_pandas(self.train_df_ood)
        if self.test_df_ood is not None:
            dataset_dict_ood['test'] = Dataset.from_pandas(self.test_df_ood)
        if self.val_df_ood is not None:
            dataset_dict_ood['validation'] = Dataset.from_pandas(self.val_df_ood)
        if len(dataset_dict_id) > 1:
            other_domain_dataset = DatasetDict(dataset_dict_ood)
            other_domain_tokenized_dataset = other_domain_dataset.map(tokenize_function, batched=True)
        else:
            other_domain_dataset = {}
            other_domain_tokenized_dataset = {}

        return indomain_dataset, indomain_tokenized_datasets, other_domain_dataset, other_domain_tokenized_dataset
        