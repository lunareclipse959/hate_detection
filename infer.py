import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# --- 0. 配置参数 (与训练代码保持一致) ---
OUTPUT_DIR = "./fine_tuned_hate_speech_quadruple_model"  # 模型保存目录，必须与训练时保存的目录一致
MAX_INPUT_LENGTH = 512  # 输入文本的最大长度
MAX_TARGET_LENGTH = 256  # 目标输出（四元组字符串）的最大长度，应与训练时保持一致
SPECIAL_TOKENS = ['[SEP]', '[END]']  # 自定义特殊分隔符，必须与训练时使用的保持一致

# 定义四元组元素的键名，便于后续结构化
QUADRUPLE_KEYS = ["Target", "Argument", "Targeted Group", "Hateful"]

# 测试数据文件路径 (必须存在)
TEST_DATA_FILE = './test3.json'
RESULT_FILE_PATH = 'result.txt'  # 推理结果将实时写入此文件

# --- 1. 检查模型文件是否存在 ---
# 检查模型和分词器的关键文件
model_bin_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
model_safetensors_path = os.path.join(OUTPUT_DIR, "model.safetensors")
config_path = os.path.join(OUTPUT_DIR, "config.json")
tokenizer_config_path = os.path.join(OUTPUT_DIR, "tokenizer_config.json")
tokenizer_file_path = os.path.join(OUTPUT_DIR, "tokenizer.json")  # Fast tokenizer file

if not (os.path.exists(config_path) and os.path.exists(tokenizer_config_path) and os.path.exists(tokenizer_file_path)):
    print(f"错误：在 '{OUTPUT_DIR}' 目录中找不到完整的模型或分词器配置文件。")
    print("请确保训练已成功完成，并且模型已保存到此目录。")
    exit()

if not (os.path.exists(model_bin_path) or os.path.exists(model_safetensors_path)):
    print(f"错误：在 '{OUTPUT_DIR}' 目录中找不到模型权重文件 (pytorch_model.bin 或 model.safetensors)。")
    print("请确保训练已成功完成，并且模型已保存到此目录。")
    exit()

# --- 2. 加载模型和分词器 ---
print(f"正在加载训练好的模型和分词器：{OUTPUT_DIR}")
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR)

# 确保分词器中添加了自定义特殊tokens
# 尽管save_pretrained通常会保存这些信息，但为了鲁棒性，再次添加是个好习惯，
# 确保在加载的模型中这些特殊token的ID与训练时保持一致。
num_added_toks = tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
if num_added_toks > 0:
    print(f"已在加载的分词器中重新添加 {num_added_toks} 个自定义Tokens。")
else:
    print(f"特殊Tokens {SPECIAL_TOKENS} 已存在于分词器中。")

# --- 3. 设置设备 (GPU或CPU) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # 将模型设置为评估模式

print(f"模型已加载到：{device}")


# --- 4. 定义解析函数 ---
def parse_quadruples(generated_text: str, special_sep: str, special_end: str, keys: list):
    """
    解析模型生成的四元组字符串。

    Args:
        generated_text (str): 模型生成的原始字符串，包含四元组、[SEP] 和 [END]。
        special_sep (str): 四元组之间的分隔符，如 '[SEP]'。
        special_end (str): 单个四元组的结束符，如 '[END]'。
        keys (list): 四元组元素的键名列表，如 ["Target", "Argument", "Targeted Group", "Hateful"]。

    Returns:
        list[dict]: 包含解析出的仇恨四元组的字典列表。
    """
    parsed_quads = []

    # 移除字符串开头和结尾的空白，以及可能的pad token
    text_to_parse = generated_text.strip()
    if text_to_parse.startswith(tokenizer.pad_token):
        text_to_parse = text_to_parse[len(tokenizer.pad_token):].strip()

    # 将多个四元组字符串以 " [SEP] " 分割
    # 注意：这里假设模型会生成 " [SEP] " 带有空格
    quadruple_strings = text_to_parse.split(f" {special_sep} ")

    for q_str in quadruple_strings:
        # 移除每个四元组末尾的 [END] 并清理空白
        q_str_cleaned = q_str.replace(special_end, '').strip()

        # 如果清理后字符串为空，跳过
        if not q_str_cleaned:
            continue

        # 将四元组的元素以 " | " 分割
        # 注意：这里假设模型会生成 " | " 带有空格
        elements = q_str_cleaned.split(" | ")

        if len(elements) == len(keys):
            # 将元素映射到定义的键
            quadruple = {keys[i]: elements[i].strip() for i in range(len(keys))}
            parsed_quads.append(quadruple)
        else:
            # 打印警告，但仍然尝试处理后续的四元组
            print(f"警告: 发现一个格式不正确的四元组（元素数量不匹配）：'{q_str_cleaned}'")

    return parsed_quads


# --- 5. 定义推理函数 ---
def generate_and_parse_quadruple(text: str, model, tokenizer, device, max_length=MAX_TARGET_LENGTH):
    """
    对输入的文本生成仇恨言论四元组字符串，并进行解析。
    返回原始生成字符串和解析后的结构化列表。
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True
    ).to(device)

    # 使用模型生成输出序列
    generated_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_length,
        num_beams=5,  # 束搜索大小，可根据性能和质量需求调整
        do_sample=False,  # 设置为True可以增加多样性，False则结果更确定
        early_stopping=True,  # 生成EOS token后提前停止
    )

    # 解码生成的token ID为字符串
    # skip_special_tokens=False 以便我们可以看到 [SEP] 和 [END]
    raw_generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    # 解析四元组
    parsed_quads = parse_quadruples(raw_generated_text, SPECIAL_TOKENS[0], SPECIAL_TOKENS[1], QUADRUPLE_KEYS)

    return raw_generated_text, parsed_quads


# --- 6. 加载测试数据 (必须加载) ---
def load_test_data(file_path):
    """
    加载测试数据，假定文件格式与训练集相似，只需要 'content' 字段用于输入。
    如果文件不存在或解析失败，则抛出错误。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误：测试数据文件 '{file_path}' 不存在。请确保文件路径正确。")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 提取 'content' 字段作为输入，并确保其不为空
        contents = [item.get("content", "") for item in data if item.get("content") is not None]
        if not contents:
            raise ValueError(f"错误：文件 '{file_path}' 中没有找到有效的 'content' 字段数据。请检查文件内容。")
        return contents
    except json.JSONDecodeError as e:
        raise ValueError(f"错误：无法解析文件 '{file_path}'。请检查其JSON格式是否正确。详细信息：{e}")


print(f"正在加载测试数据：{TEST_DATA_FILE}")
try:
    test_inputs = load_test_data(TEST_DATA_FILE)
    print(f"已从 {TEST_DATA_FILE} 加载 {len(test_inputs)} 条测试数据。")
except (FileNotFoundError, ValueError) as e:
    print(e)
    exit()  # 加载失败则直接退出

# --- 7. 开始推理并将结果实时写入文件 ---
print(f"\n--- 开始推理测试，结果将实时写入 '{RESULT_FILE_PATH}' ---")

# 使用 'w' 模式，每次运行会清空文件并重写。
# 如果希望在文件末尾追加，请将 'w' 改为 'a'。
with open(RESULT_FILE_PATH, 'w', encoding='utf-8') as f_result:
    for i, text in enumerate(test_inputs):
        print(f"\n--- 处理示例 {i + 1}/{len(test_inputs)} ---")
        print(f"输入文本: {text}")

        raw_output, parsed_quadruples = generate_and_parse_quadruple(text, model, tokenizer, device)

        # 将结果写入文件
        f_result.write(f"\n--- 示例 {i + 1} ---\n")
        f_result.write(f"输入文本: {text}\n")
        f_result.write(f"原始模型输出: {raw_output}\n")
        f_result.write("解析后四元组:\n")

        if parsed_quadruples:
            for j, quad in enumerate(parsed_quadruples):
                f_result.write(f"  四元组 {j + 1}:\n")
                for key, value in quad.items():
                    f_result.write(f"    {key}: {value}\n")
        else:
            f_result.write("  未解析到有效的四元组。\n")

        # 实时刷新文件缓冲区，确保内容立即写入磁盘
        f_result.flush()
        os.fsync(f_result.fileno())  # 确保数据真正写入物理磁盘

        # 同样在控制台打印，方便实时查看进度
        print(f"原始模型输出: {raw_output}")
        print("解析后四元组:")
        if parsed_quadruples:
            for j, quad in enumerate(parsed_quadruples):
                print(f"  四元组 {j + 1}:")
                for key, value in quad.items():
                    print(f"    {key}: {value}")
        else:
            print("  未解析到有效的四元组。")

print(f"\n--- 推理测试完成！结果已写入 '{RESULT_FILE_PATH}' ---")