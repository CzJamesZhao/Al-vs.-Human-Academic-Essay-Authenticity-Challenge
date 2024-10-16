import json
import pandas as pd
import nlpaug.augmenter.word as naw
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import logging

import os
from scorer.task import evaluate, _read_gold_labels_file, _read_tsv_input_file, correct_labels

# 1. 设置日志记录格式
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# 2. 数据加载函数
def load_data(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 3. 加载训练和验证数据
train_df = load_data('data/academic_essay_english_train.jsonl')
val_df = load_data('data/academic_essay_english_dev.jsonl')

# 4. 标签映射
label_mapping = {'ai': 1, 'human': 0}
train_labels = [label_mapping[label] for label in train_df['label'].tolist()]
val_labels = [label_mapping[label] for label in val_df['label'].tolist()]

# 5. 数据增强
def augment_texts(texts, augmentation_ratio=0.2):
    augmenter = naw.SynonymAug(aug_p=augmentation_ratio)
    augmented_texts = []
    for text in texts:
        augmented_text = augmenter.augment(text)
        augmented_texts.extend(augmented_text)
    return augmented_texts

train_texts = train_df['essay'].tolist()
train_texts_augmented = augment_texts(train_texts)
train_texts.extend(train_texts_augmented)
train_labels.extend(train_labels)  # Ensure that the length matches augmented texts

# 6. 加载 XLM-RoBERTa 模型和分词器
tokenizer = XLMRobertaTokenizer.from_pretrained('./xlm-roberta-base')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_df['essay'].tolist(), truncation=True, padding=True, max_length=128)

# 7. 创建 Dataset 类
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# 8. 训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
)

# 9. 模型初始化
model = XLMRobertaForSequenceClassification.from_pretrained('./xlm-roberta-base', num_labels=2)

# 10. 创建 Trainer 实例并训练
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)
trainer.train()

# 11. 保存验证集的预测结果
predictions = trainer.predict(val_dataset)
pred_labels = predictions.predictions.argmax(-1)
output_tsv_path = "predictions.tsv"

# 将验证集预测结果写入 TSV 文件
with open(output_tsv_path, 'w', encoding='utf-8') as f_out:
    f_out.write("id\tprediction\n")
    for idx, label in enumerate(pred_labels):
        f_out.write(f"{val_df['id'].iloc[idx]}\t{label}\n")

# 12. 调用 scorer/task.py 进行评估
gold_file_path = 'data/academic_essay_english_dev.jsonl'
gold_labels = _read_gold_labels_file(gold_file_path)
pred_labels = _read_tsv_input_file(output_tsv_path)

# 检查标签并进行评估
if correct_labels(pred_labels, gold_labels):
    acc, precision, recall, f1 = evaluate(pred_labels, gold_labels)
    logging.info(f"Evaluation Metrics: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
else:
    logging.error("Prediction labels do not match gold labels.")

# 13. 加载测试集数据并进行分词
test_df = load_data('data/academic_essay_english_dev_test_no_label.jsonl')
test_texts = test_df['essay'].tolist()
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# 14. 创建测试集 Dataset
test_dataset = TextDataset(test_encodings)

# 15. 使用训练好的模型对测试集进行预测
test_predictions = trainer.predict(test_dataset)
test_pred_labels = test_predictions.predictions.argmax(-1)

# 16. 保存测试集的预测结果
test_output_tsv_path = "test_predictions.tsv"
with open(test_output_tsv_path, 'w', encoding='utf-8') as f_out:
    f_out.write("id\tprediction\n")
    for idx, label in enumerate(test_pred_labels):
        f_out.write(f"{test_df['id'].iloc[idx]}\t{label}\n")

logging.info(f"Test predictions saved to {test_output_tsv_path}")
