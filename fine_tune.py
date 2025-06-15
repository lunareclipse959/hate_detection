import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    DataCollatorForSeq2Seq

# --- 0. 配置参数 ---
# 确保 MODEL_NAME 指向你服务器上本地模型的正确路径
MODEL_NAME = "./"
OUTPUT_DIR = "./fine_tuned_hate_speech_quadruple_model"  # 模型保存目录，建议使用新名称以区分任务
LOGGING_DIR = f"{OUTPUT_DIR}/logs"

# 数据处理相关参数
MAX_INPUT_LENGTH = 512  # 输入文本的最大长度，超出截断
# MAX_TARGET_LENGTH 必须足够长以包含完整的四元组字符串及所有分隔符。
# 强烈建议运行 "analyze_target_length.py" 脚本来根据你的实际数据精确确定此值。
# 经验值通常在150-300之间，但请务必验证。
MAX_TARGET_LENGTH = 256
SPECIAL_TOKENS = ['[SEP]', '[END]']  # 自定义特殊分隔符，与任务输出格式严格对应

# 训练参数
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
NUM_TRAIN_EPOCHS = 5
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
SAVE_STEPS = 300
SAVE_TOTAL_LIMIT = 2
FP16_ENABLED = True

# --- 1. 初始化分词器和模型 ---
print(f"正在加载模型和分词器：{MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 添加自定义特殊tokens并调整模型embedding大小
# 这是关键步骤，确保模型能识别和生成 [SEP] 和 [END]
print(f"正在添加自定义特殊Tokens：{SPECIAL_TOKENS}")
# add_special_tokens 方法返回实际添加的tokens数量
num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
if num_added_toks > 0:
    # 只有当实际添加了新的tokens时才调整模型嵌入层，避免不必要的操作
    model.resize_token_embeddings(len(tokenizer))
    print(f"已添加 {num_added_toks} 个新Tokens，并调整了模型嵌入层以适应新的词汇表大小。")
else:
    print(f"特殊Tokens {SPECIAL_TOKENS} 已存在于分词器中，无需重复添加。")


# --- 2. 数据加载函数 ---
def load_and_format_data(file_path):
    """
    加载JSON文件，并将其转换为Hugging Face Dataset所需的格式。
    此任务的训练数据格式要求非常严格，'output' 字段必须是模型要生成的精确字符串格式。

    预期JSON文件格式示例：
    [
        {
            "id": "123",
            "content": "你可真是头蠢驴，这都做不好。",
            "output": "你 | 蠢驴 | non-hate | non-hate [END]"
        },
        {
            "id": "456",
            "content": "老黑我是真的讨厌，媚黑的还倒贴。",
            "output": "老黑 | 讨厌 | Racism | hate [SEP] 媚黑的 | 倒贴 | Racism | hate [END]"
        },
        // 更多数据...
    ]
    请注意：
    - 四元组内部元素之间使用 " | " 分隔（注意空格）。
    - 每个四元组末尾使用 " [END]" 结束符（注意空格）。
    - 多个四元组之间使用 " [SEP] " 分隔（注意空格）。
    - 'id' 和 'content' 字段用于输入，'output' 字段作为模型的目标标签。
    """
    if not os.path.exists(file_path):
        print(f"警告：数据文件 {file_path} 不存在。")
        return None

    print(f"正在读取数据文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 将原始数据转换为 "input" 和 "label" 字段，以符合Trainer的通用约定。
    # "label" 字段将直接包含完整的、严格格式化的仇恨四元组字符串。
    # 模型将学习生成这个准确的字符串。
    formatted_data = [{"input": x["content"], "label": x["output"]} for x in raw_data]
    print(f"文件 {file_path} 已加载 {len(formatted_data)} 条数据。")
    return Dataset.from_list(formatted_data)


# --- 3. 数据预处理函数 ---
def preprocess_function(examples):
    """
    对输入文本和目标标签（即仇恨四元组字符串）进行分词编码。
    这里不进行padding，而是将padding交给DataCollatorForSeq2Seq处理，以提高效率。
    """
    # 对输入评论进行编码
    inputs = tokenizer(
        examples["input"],
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    # 对目标四元组字符串进行编码。
    # examples["label"] 此时已经是严格格式化的四元组字符串（如 "Target | Arg | Group | Hateful [END] [SEP] ..."）。
    targets = tokenizer(
        examples["label"],
        truncation=True,
        max_length=MAX_TARGET_LENGTH
    )

    # 将目标序列的input_ids作为模型的labels。
    # 在T5模型中，decoder的输入通常是label本身（移位操作），损失计算是基于这些label进行的。
    inputs["labels"] = targets["input_ids"]
    return inputs


# --- 4. 加载和预处理数据集 ---
print("正在加载数据集...")
train_dataset = load_and_format_data('./train.json')
if train_dataset is None:
    raise FileNotFoundError("训练集 train.json 未找到或为空！请确保文件存在且符合预期格式。")

eval_dataset = load_and_format_data('./test.json')  # 尝试加载验证集
if eval_dataset is None:
    print("未找到测试集 test.json，训练过程中将不进行评估。建议提供测试集以监控训练效果。")

print("正在对数据集进行分词预处理 (这可能需要一些时间)...")
tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,  # 批处理模式，加速处理
    # 移除原始列，只保留模型需要的 input_ids, attention_mask, labels
    # Trainer 会自动处理这些字段
    remove_columns=train_dataset.column_names
)
print(f"训练集预处理完成。样本数: {len(tokenized_train)}")

tokenized_eval = None
if eval_dataset:
    tokenized_eval = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    print(f"验证集预处理完成。样本数: {len(tokenized_eval)}")

# --- 5. 数据Collator设置 ---
# DataCollatorForSeq2Seq 负责在批处理时动态填充序列，并为标签序列设置-100（忽略损失计算）。
# 动态填充意味着填充到当前批次中最长的序列长度，而不是预设的MAX_LENGTH，这样更高效。
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,  # 传入model，colator会使用model.config.pad_token_id进行标签填充（对于T5通常是0）
    padding=True,  # 启用动态填充，填充到批次中的最大长度
    # max_length=MAX_INPUT_LENGTH, # DataCollator也可以设置此参数，但通常与preprocess_function中的truncation配合使用即可
    label_pad_token_id=tokenizer.pad_token_id  # T5模型通常用pad_token_id填充labels
)

# --- 6. 训练参数设置 ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_dir=LOGGING_DIR,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,  # 最多保存N个检查点
    fp16=FP16_ENABLED,  # 启用混合精度训练 (推荐用于支持的GPU，如RTX 4090)
    predict_with_generate=True,  # 评估时使用生成模式（Decode）
    generation_max_length=MAX_TARGET_LENGTH,  # 评估时生成输出的最大长度
    evaluation_strategy="epoch" if tokenized_eval else "no",  # 如果有验证集，则每个epoch评估
    load_best_model_at_end=True if tokenized_eval else False,  # 如果有评估，训练结束时加载最佳模型
    metric_for_best_model="eval_loss" if tokenized_eval else None,  # 用于选择最佳模型的指标，默认为验证损失
    report_to="none",  # 不使用WandB、TensorBoard等报告工具。可改为"tensorboard"进行可视化。
    # 更多高级参数如 warmup_steps, weight_decay, gradient_checkpointing 等可根据需要添加
)

# --- 7. 初始化Trainer并开始训练 ---
print("正在初始化训练器...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,  # 如果存在，则传入验证集
    tokenizer=tokenizer,  # 传入tokenizer，DataCollator和Trainer会使用
    data_collator=data_collator,
    # compute_metrics=None, # 如果你需要自定义评估指标（例如，在解析四元组后计算F1），可以在这里提供函数
)

print("\n--- 训练开始！---")
trainer.train()

# --- 8. 保存最终模型 ---
print(f"\n--- 训练完成！正在保存最终模型到：{OUTPUT_DIR} ---")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)  # 同时保存分词器，确保推理时使用正确的词汇表
print("模型和分词器已成功保存。")