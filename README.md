# Chinese_hate_detection
中文细粒度仇恨言论识别与四元组抽取
本项目旨在对来自中文社交平台（如微博、知乎、贴吧等）的用户评论进行严格审查，识别其中是否包含针对某类群体的歧视性、仇恨性言论，并以结构化四元组的形式进行抽取。

任务目标
核心任务是从评论文本中提取一个包含以下四个元素的四元组：

评论对象 (Target)
论点 (Argument)
目标群体 (Targeted Group)
是否仇恨 (Hateful)
项目支持单条评论中包含多个四元组，并严格遵循特定的输出格式和分隔符。

技术方案
本项目提供了两种实现方式：

基于大型语言模型API调用 (LLM API Calling)

核心思想： 利用强大的预训练大型语言模型（LLM）的通用理解和生成能力。
实现方式： 通过精心设计的提示工程 (Prompt Engineering)，将任务要求、规则和示例注入到输入提示词中，引导LLM直接生成符合格式的四元组输出。
所用模型： 讯飞星火大模型 ,Deepseek。

基于预训练模型微调 (Model Fine-tuning)

核心思想： 在特定任务的标注数据集上，对一个预训练模型进行额外的训练，使其深度适应任务的数据分布和复杂输出逻辑。
实现方式： 将任务视为一个文本到文本 (Text-to-Text) 的生成问题，利用编码器-解码器架构的模型进行端到端训练。
所用模型： IDEA-CCNL/Randeng-T5-784M-Chinese。

文件说明
train.json: 训练数据集，用于模型微调。
test.json: 测试数据集，用于评估模型性能及API调用示例。
fine_tune.py : 实现基于 Randeng-T5-784M-Chinese 模型的微调代码。
infer.py: 实现微调模型的测试代码。
API.py : 实现基于大模型API调用进行仇恨言论抽取的代码。
result.txt: API 调用方式的输出结果文件，实时更新。

运行环境
确保您已安装必要的Python库：
pip install torch transformers accelerate datasets sentencepiece jsonlines openai websocket-client

性能对比 
初步实验结果表明，两种方法均能以一定的分数完成任务。其中，基于微调的 IDEA-CCNL/Randeng-T5-784M-Chinese 模型在遵循复杂格式约束和抽取精度方面表现略优于直接调用大型语言模型API的方法。

