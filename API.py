import json
import requests
import time
from tqdm import tqdm

# ✅ 设置 DeepSeek API key
DEEPSEEK_API_KEY = "xxxxx"

# ✅ DeepSeek Chat 接口 URL
DEEPSEEK_URL = "xxxxxx"

# ✅ 模型调用函数 - 不打印，只返回
def deepseek_extract_4tuple(text):
    prompt = f"""
    你是一名中文文本审核专家，现在接到了一个任务，需要对来自社交平台（如：微博、知乎、贴吧等）的用户评论进行严格审查，识别其中是否包含针对某类群体的歧视性、仇恨性言论。请你遵循以下规范完成任务：
【任务内容】
具体你需要提取出一个四元组：
1. 评论对象（Target）： 帖子的评述对象，如一个人或一个群体。当实例无具体目标时设为NULL，例如:"保持安全!"，但是无具体目标的评论比较罕见。
2. 论点（Argument）： 包含对评论目标关键论点的信息片段，对目标的评价、指责、贬低或陈述，可能是贬义、偏见、攻击性话语等。
3. 目标群体（Targeted Group）： 指包含仇恨信息的评论对象-论点对涉及的目标群体。标注的目标群体包括“地域歧视”、“种族歧视”、“性别歧视”、“LGBTQ群体歧视”、“其他”共5类。分别标记为"Region"、"Racism"、"Sexism"、"LGBTQ"、"others"。（如果不是仇恨内容则为non-hate）
4. 是否仇恨（Hateful）： 评论对象-论点对是否构成了对某些群体的仇恨言论。请标记为 "hate" 或 "non-hate"。
【特殊规则】
对于非仇恨文本和不包含特定群体的一般攻击性言论，同样需要对目标群体和观点进行抽取，并设为Non-hate。由于样本中可能有多个评论对象，因此可以包含多个四元组。 
每个四元组中各个元素之间用" | "分割，并利用 [END] 结尾；如果一条样本中包含多个四元组，不同四元组之间利用 [SEP] 分割。
论点必须是评论的子集，在原句中。你只能输出一行，不要包含其他任何解释的内容。对于同一个评论对象，可能同时包含不同类别的仇恨类型，此时应用英文逗号','分隔。
请严格按照顺序和格式提交，不要省略空格。
【示例】
输入： 你可真是头蠢驴，这都做不好。
输出： 你 | 蠢驴 | non-hate | non-hate [END]
输入： 老黑我是真的讨厌，媚黑的还倒贴。
输出： 老黑 | 讨厌 | Racism | hate [SEP] 媚黑的 | 倒贴 | Racism | hate [END]
【以下是你要处理的评论】
"{text}"
请你按照上述规范，从该评论中抽取仇恨言论的四元组并输出。
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    payload = {
        "model": "deepseek-chat",  # 根据你的账号权限适配，如 deepseek-coder 等
        "messages": [
            {"role": "system", "content": "你是一个中文文本审核系统，擅长识别仇恨言论。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "top_p": 1.0
    }

    try:
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result_text = response.json()['choices'][0]['message']['content']
            return result_text.strip().replace("\n", " ")
        else:
            return "NULL | NULL | non-hate | non-hate [END]"
    except Exception:
        return "NULL | NULL | non-hate | non-hate [END]"

# ✅ 加载测试集
with open("test1.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# ✅ 打开输出文件（初始化就清空内容）
with open("result.txt", "w", encoding="utf-8") as f:
    pass  # 清空文件

# ✅ 按样本逐条写入结果
with open("result.txt", "a", encoding="utf-8") as out_f:
    for item in tqdm(test_data):
        content = item["content"]
        result = deepseek_extract_4tuple(content)

        # 实时写入文件
        out_f.write(result + "\n")
        out_f.flush()

        # 防止接口频率限制
        time.sleep(1)