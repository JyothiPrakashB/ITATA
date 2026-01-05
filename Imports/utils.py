import numpy as np
import math
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
)


class RewardWeightedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Custom Seq2Seq Trainer that incorporates reward-weighted loss for RL fine-tuning.

    The reward is computed based on RMSE_VA: reward = exp(-alpha * rmse_va)
    Lower RMSE = higher reward. The loss is weighted inversely by reward to focus
    gradient updates on harder samples.
    """

    def __init__(self, *args, reward_alpha=0.5, rl_weight=0.1, **kwargs):
        """
        Args:
            reward_alpha: Scaling factor for reward computation (default: 0.5)
            rl_weight: Weight for the reward-weighted loss component (default: 0.1)
        """
        super().__init__(*args, **kwargs)
        self.reward_alpha = reward_alpha
        self.rl_weight = rl_weight

    def _compute_per_sample_rmse_va(self, predictions, labels):
        """
        Compute per-sample RMSE_VA from predicted and ground truth VA values.

        Args:
            predictions: Decoded prediction strings
            labels: Ground truth label strings

        Returns:
            Tensor of per-sample RMSE_VA values
        """
        rmse_values = []

        for pred, label in zip(predictions, labels):
            try:
                # Extract VA from predictions (handles both CoT and standard format)
                pred_v, pred_a = self._extract_va_from_string(pred)
                label_v, label_a = self._extract_va_from_string(label)

                # Compute RMSE_VA for this sample
                rmse = math.sqrt((pred_v - label_v)**2 + (pred_a - label_a)**2)
                rmse_values.append(rmse)
            except:
                # Default to high RMSE on parse failure
                rmse_values.append(4.0)

        return torch.tensor(rmse_values, device=self.args.device)

    def _extract_va_from_string(self, text):
        """
        Extract valence and arousal values from a string.
        Handles both "V.VV#A.AA" format and "output: V.VV#A.AA" CoT format.
        """
        # Try to find VA pattern in the text
        va_pattern = r'(\d+\.?\d*)\s*#\s*(\d+\.?\d*)'
        match = re.search(va_pattern, text)

        if match:
            valence = float(match.group(1))
            arousal = float(match.group(2))
            # Clamp to valid range
            valence = max(1.0, min(9.0, valence))
            arousal = max(1.0, min(9.0, arousal))
            return valence, arousal

        return 5.0, 5.0  # Default neutral

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with reward-weighted component.

        During training, we compute:
        1. Standard supervised loss (cross-entropy)
        2. Generate predictions for reward computation
        3. Compute reward-weighted adjustment

        Total loss = supervised_loss + rl_weight * weighted_loss
        """
        # Get standard supervised loss
        outputs = model(**inputs)
        supervised_loss = outputs.loss

        # For efficiency, only compute reward weighting periodically during training
        # or when we have access to labels for reward computation
        if self.model.training and self.rl_weight > 0:
            try:
                # Generate predictions for the current batch
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        max_length=256 if hasattr(self.args, 'max_length') else 128,
                        num_beams=1,  # Greedy for speed
                        do_sample=False
                    )

                # Decode predictions and labels
                predictions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                labels = inputs.get('labels')
                if labels is not None:
                    # Replace -100 with pad token for decoding
                    labels_for_decode = labels.clone()
                    labels_for_decode[labels_for_decode == -100] = self.tokenizer.pad_token_id
                    label_texts = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

                    # Compute rewards
                    rmse_values = self._compute_per_sample_rmse_va(predictions, label_texts)
                    rewards = torch.exp(-self.reward_alpha * rmse_values)

                    # Compute per-sample loss weights (focus on harder samples)
                    weights = 1.0 - rewards  # Higher weight for lower reward
                    weights = weights.to(supervised_loss.device)

                    # The reward-weighted component is the supervised loss scaled by difficulty
                    # This encourages the model to focus on harder samples
                    weighted_component = supervised_loss * weights.mean()

                    total_loss = supervised_loss + self.rl_weight * weighted_component
                else:
                    total_loss = supervised_loss

            except Exception as e:
                # Fallback to standard loss on any error
                total_loss = supervised_loss
        else:
            total_loss = supervised_loss

        return (total_loss, outputs) if return_outputs else total_loss


class T5Generator:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=64, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def tokenize_function_inputs_cot(self, sample):
        """
        Tokenize function for CoT format with longer output length.
        """
        model_inputs = self.tokenizer(sample['text'], max_length=512, truncation=True)
        labels = self.tokenizer(sample["labels"], max_length=256, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self, tokenized_datasets, use_reward_weighted=False, rl_weight=0.1,
              reward_alpha=0.5, **kwargs):
        """
        Train the generative model.

        Args:
            tokenized_datasets: HuggingFace DatasetDict with train/validation splits
            use_reward_weighted: Whether to use RewardWeightedSeq2SeqTrainer (default: False)
            rl_weight: Weight for reward-weighted loss component (default: 0.1)
            reward_alpha: Alpha for reward computation exp(-alpha * rmse) (default: 0.5)
            **kwargs: Arguments passed to Seq2SeqTrainingArguments
        """
        # Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
        )

        # Select trainer type
        if use_reward_weighted:
            print(f'Using RewardWeightedSeq2SeqTrainer (rl_weight={rl_weight}, alpha={reward_alpha})')
            trainer = RewardWeightedSeq2SeqTrainer(
                self.model,
                args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                reward_alpha=reward_alpha,
                rl_weight=rl_weight,
            )
        else:
            trainer = Seq2SeqTrainer(
                self.model,
                args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
            )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, max_length = 128, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch, max_length = max_length)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred, is_triplet_extraction=False):
        total_pred = 0
        total_gt = 0
        tp = 0
        if not is_triplet_extraction:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    for pred_val in pred_list:
                        if pred_val in gt_val or gt_val in pred_val:
                            tp+=1
                            break

        else:
            for gt, pred in zip(y_true, y_pred):
                gt_list = gt.split(', ')
                pred_list = pred.split(', ')
                total_pred+=len(pred_list)
                total_gt+=len(gt_list)
                for gt_val in gt_list:
                    gt_asp = gt_val.split(':')[0]

                    try:
                        gt_op = gt_val.split(':')[1]
                    except:
                        continue

                    try:
                        gt_sent = gt_val.split(':')[2]
                    except:
                        continue

                    for pred_val in pred_list:
                        pr_asp = pred_val.split(':')[0]

                        try:
                            pr_op = pred_val.split(':')[1]
                        except:
                            continue

                        try:
                            pr_sent = gt_val.split(':')[2]
                        except:
                            continue

                        if pr_asp in gt_asp and pr_op in gt_op and gt_sent == pr_sent:
                            tp+=1

        p = tp/total_pred
        r = tp/total_gt
        return p, r, 2*p*r/(p+r), None

    def get_metrics_regression(self, y_true, y_pred):
        """
        Calculate regression metrics for DimASR task using the official RMSE_VA formula.

        RMSE_VA = sqrt( sum((V_pred - V_gold)^2 + (A_pred - A_gold)^2) / N )

        Args:
            y_true: List of ground truth VA strings (e.g., ["7.12#6.88", ...])
            y_pred: List of predicted VA strings

        Returns:
            dict with RMSE_VA (official metric), PCC_V, PCC_A, and other metrics
        """
        true_valence, true_arousal = [], []
        pred_valence, pred_arousal = [], []

        for gt, pred in zip(y_true, y_pred):
            # Parse ground truth
            try:
                gt_parts = gt.strip().split('#')
                gt_v = float(gt_parts[0]) if len(gt_parts) > 0 else 5.0
                gt_a = float(gt_parts[1]) if len(gt_parts) > 1 else 5.0
            except (ValueError, IndexError):
                gt_v, gt_a = 5.0, 5.0  # Default neutral

            # Parse prediction with flexible handling
            try:
                # Handle various formats the model might output
                pred_clean = pred.strip().replace(',', '.')
                pred_parts = pred_clean.split('#')
                pred_v = float(pred_parts[0]) if len(pred_parts) > 0 else 5.0
                pred_a = float(pred_parts[1]) if len(pred_parts) > 1 else 5.0
            except (ValueError, IndexError):
                pred_v, pred_a = 5.0, 5.0  # Default neutral on parse failure

            # Clamp to valid range [1.0, 9.0]
            pred_v = max(1.0, min(9.0, pred_v))
            pred_a = max(1.0, min(9.0, pred_a))

            true_valence.append(gt_v)
            true_arousal.append(gt_a)
            pred_valence.append(pred_v)
            pred_arousal.append(pred_a)

        # Convert to numpy arrays
        true_valence = np.array(true_valence)
        true_arousal = np.array(true_arousal)
        pred_valence = np.array(pred_valence)
        pred_arousal = np.array(pred_arousal)

        n = len(true_valence)

        # Official RMSE_VA: sqrt( sum((V_p - V_g)^2 + (A_p - A_g)^2) / N )
        total_sq_error = np.sum((pred_valence - true_valence)**2 + (pred_arousal - true_arousal)**2)
        rmse_va = math.sqrt(total_sq_error / n)

        # Pearson Correlation Coefficient
        try:
            pcc_v, _ = pearsonr(true_valence, pred_valence)
        except:
            pcc_v = 0.0

        try:
            pcc_a, _ = pearsonr(true_arousal, pred_arousal)
        except:
            pcc_a = 0.0

        # Individual RMSE for valence and arousal
        rmse_v = math.sqrt(np.mean((true_valence - pred_valence)**2))
        rmse_a = math.sqrt(np.mean((true_arousal - pred_arousal)**2))

        return {
            'RMSE_VA': rmse_va,      # Official evaluation metric
            'PCC_V': pcc_v,           # Pearson correlation for valence
            'PCC_A': pcc_a,           # Pearson correlation for arousal
            'PCC_avg': (pcc_v + pcc_a) / 2,
            'RMSE_V': rmse_v,
            'RMSE_A': rmse_a
        }

    def parse_va_prediction(self, prediction):
        """
        Parse a VA prediction string and return formatted values.

        Args:
            prediction: Raw model output string

        Returns:
            Formatted VA string "X.XX#X.XX" clamped to [1.00, 9.00]
        """
        try:
            pred_clean = prediction.strip().replace(',', '.')
            parts = pred_clean.split('#')
            valence = float(parts[0]) if len(parts) > 0 else 5.0
            arousal = float(parts[1]) if len(parts) > 1 else 5.0
        except (ValueError, IndexError):
            valence, arousal = 5.0, 5.0

        # Clamp to valid range [1.00, 9.00]
        valence = max(1.0, min(9.0, valence))
        arousal = max(1.0, min(9.0, arousal))

        return f"{valence:.2f}#{arousal:.2f}"

    def parse_cot_va_prediction(self, prediction, return_reasoning=False):
        """
        Parse a CoT VA prediction string and extract the VA values.

        The model outputs format like:
        "The text expresses... Valence: positive (~7.5). Arousal: high (~7.0).
        output: 7.50#7.00"

        Args:
            prediction: Raw model output string with CoT reasoning
            return_reasoning: If True, return tuple (va_string, reasoning)

        Returns:
            Formatted VA string "X.XX#X.XX" or tuple (va_string, reasoning)
        """
        # Try to find "output: X.XX#X.XX" pattern first
        output_pattern = r'output:\s*(\d+\.?\d*)\s*#\s*(\d+\.?\d*)'
        match = re.search(output_pattern, prediction, re.IGNORECASE)

        if match:
            valence = float(match.group(1))
            arousal = float(match.group(2))
        else:
            # Fallback: try to find any VA pattern
            va_pattern = r'(\d+\.?\d*)\s*#\s*(\d+\.?\d*)'
            match = re.search(va_pattern, prediction)
            if match:
                valence = float(match.group(1))
                arousal = float(match.group(2))
            else:
                valence, arousal = 5.0, 5.0

        # Clamp to valid range [1.00, 9.00]
        valence = max(1.0, min(9.0, valence))
        arousal = max(1.0, min(9.0, arousal))

        va_string = f"{valence:.2f}#{arousal:.2f}"

        if return_reasoning:
            # Extract reasoning (everything before "output:")
            reasoning_match = re.search(r'^(.*?)(?:output:|$)', prediction, re.DOTALL | re.IGNORECASE)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
            return va_string, reasoning

        return va_string

    def _extract_va_values(self, text):
        """
        Helper method to extract raw valence and arousal float values from text.

        Args:
            text: String containing VA values

        Returns:
            Tuple of (valence, arousal) floats
        """
        # Try "output: X.XX#X.XX" pattern first
        output_pattern = r'output:\s*(\d+\.?\d*)\s*#\s*(\d+\.?\d*)'
        match = re.search(output_pattern, text, re.IGNORECASE)

        if not match:
            # Fallback to any VA pattern
            va_pattern = r'(\d+\.?\d*)\s*#\s*(\d+\.?\d*)'
            match = re.search(va_pattern, text)

        if match:
            valence = float(match.group(1))
            arousal = float(match.group(2))
            valence = max(1.0, min(9.0, valence))
            arousal = max(1.0, min(9.0, arousal))
            return valence, arousal

        return 5.0, 5.0

    def get_metrics_regression_cot(self, y_true, y_pred):
        """
        Calculate regression metrics for DimASR CoT task.
        Same as get_metrics_regression but handles CoT output format.

        Args:
            y_true: List of ground truth VA strings (may be CoT format)
            y_pred: List of predicted VA strings (may be CoT format)

        Returns:
            dict with RMSE_VA, PCC_V, PCC_A, and other metrics
        """
        true_valence, true_arousal = [], []
        pred_valence, pred_arousal = [], []

        for gt, pred in zip(y_true, y_pred):
            gt_v, gt_a = self._extract_va_values(gt)
            pred_v, pred_a = self._extract_va_values(pred)

            true_valence.append(gt_v)
            true_arousal.append(gt_a)
            pred_valence.append(pred_v)
            pred_arousal.append(pred_a)

        # Convert to numpy arrays
        true_valence = np.array(true_valence)
        true_arousal = np.array(true_arousal)
        pred_valence = np.array(pred_valence)
        pred_arousal = np.array(pred_arousal)

        n = len(true_valence)

        # Official RMSE_VA: sqrt( sum((V_p - V_g)^2 + (A_p - A_g)^2) / N )
        total_sq_error = np.sum((pred_valence - true_valence)**2 + (pred_arousal - true_arousal)**2)
        rmse_va = math.sqrt(total_sq_error / n)

        # Pearson Correlation Coefficient
        try:
            pcc_v, _ = pearsonr(true_valence, pred_valence)
        except:
            pcc_v = 0.0

        try:
            pcc_a, _ = pearsonr(true_arousal, pred_arousal)
        except:
            pcc_a = 0.0

        # Individual RMSE for valence and arousal
        rmse_v = math.sqrt(np.mean((true_valence - pred_valence)**2))
        rmse_a = math.sqrt(np.mean((true_arousal - pred_arousal)**2))

        return {
            'RMSE_VA': rmse_va,
            'PCC_V': pcc_v,
            'PCC_A': pcc_a,
            'PCC_avg': (pcc_v + pcc_a) / 2,
            'RMSE_V': rmse_v,
            'RMSE_A': rmse_a
        }


class T5Classifier:
    def __init__(self, model_checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, force_download = True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, force_download = True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
        self.device = 'cuda' if torch.has_cuda else ('mps' if torch.has_mps else 'cpu')

    def tokenize_function_inputs(self, sample):
        """
        Udf to tokenize the input dataset.
        """
        sample['input_ids'] = self.tokenizer(sample["text"], max_length = 512, truncation = True).input_ids
        sample['labels'] = self.tokenizer(sample["labels"], max_length = 64, truncation = True).input_ids
        return sample
        
    def train(self, tokenized_datasets, **kwargs):
        """
        Train the generative model.
        """

        # Set training arguments
        args = Seq2SeqTrainingArguments(
            **kwargs
            )

        # Define trainer object
        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"] if tokenized_datasets.get("validation") is not None else None,
            tokenizer=self.tokenizer, 
            data_collator = self.data_collator 
        )
        print("Trainer device:", trainer.args.device)

        # Finetune the model
        torch.cuda.empty_cache()
        print('\nModel training started ....')
        trainer.train()

        # Save best model
        trainer.save_model()
        return trainer

    def get_labels(self, tokenized_dataset, batch_size = 4, sample_set = 'train'):
        """
        Get the predictions from the trained model.
        """
        def collate_fn(batch):
            input_ids = [torch.tensor(example['input_ids']) for example in batch]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            return input_ids
        
        dataloader = DataLoader(tokenized_dataset[sample_set], batch_size=batch_size, collate_fn=collate_fn)
        predicted_output = []
        self.model.to(self.device)
        print('Model loaded to: ', self.device)

        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            output_ids = self.model.generate(batch)
            output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts:
                predicted_output.append(output_text)
        return predicted_output
    
    def get_metrics(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='macro'), \
            f1_score(y_true, y_pred, average='macro'), accuracy_score(y_true, y_pred)