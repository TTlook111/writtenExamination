import os
import json
import pandas as pd
from openai import OpenAI
import httpx # 显式导入 httpx 用于网络配置
from tqdm import tqdm
import time

# 配置 API Key 和 Base URL
# 注意：实际使用时建议将 API Key 存储在环境变量中，而不是直接硬编码
API_KEY = "sk-0701a5d1f9ae44ca9d31b9041c098e36" 
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-turbo" # 使用 Qwen-Turbo 模型

def get_client():
    """初始化 OpenAI 客户端"""
    # 使用 trust_env=False 强制忽略系统代理设置
    http_client = httpx.Client(trust_env=False)
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        http_client=http_client
    )

def load_local_data():
    """
    加载本地 Parquet 数据集文件。
    """
    parquet_path = os.path.join("data", "0000.parquet")
    print(f"正在加载本地数据集文件: {parquet_path}...")
    
    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            print(f"成功加载数据，共 {len(df)} 条。")
            # 仅采样 10 条用于演示，避免消耗过多 Token
            sample_df = df.sample(n=10, random_state=42)
            print("为节省 Token，本次运行仅随机抽取 10 条数据进行演示。")
            return sample_df
        except Exception as e:
            raise ValueError(f"无法读取 Parquet 文件: {e}")
    else:
        raise FileNotFoundError(f"未找到数据文件: {parquet_path}")

def generate_prompt(text):
    """
    设计 Prompt。
    针对通用中文文本情感分析。
    """
    system_prompt = (
        "你是一位资深的情感分析专家。"
        "你的任务是分析用户输入的文本，并将其情感倾向分类为“positive”（正面）或“negative”（负面）。\n"
        "\n"
        "输出格式要求：\n"
        "1. 你必须输出一个合法的 JSON 对象。\n"
        "2. JSON 对象必须包含以下键：\n"
        "   - \"label\": 情感标签，必须是 \"positive\" 或 \"negative\" 之一。\n"
        "   - \"confidence\": 一个 0.0 到 1.0 之间的浮点数，表示你的置信度。\n"
        "   - \"reasoning\": 用中文简要说明你选择该标签的理由（针对文本的具体内容）。\n"
        "\n"
        "请不要包含任何 markdown 格式或额外的文本。只输出原始 JSON 字符串。"
    )
    
    user_prompt = f"文本内容: \"{text}\"\n\n请分析该文本的情感倾向。"
    
    return system_prompt, user_prompt

def label_text(client, text, retries=3):
    """
    调用 LLM 进行标注，带有重试逻辑。
    """
    system_content, user_content = generate_prompt(text)
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': user_content}
                ],
                temperature=0.0,  # 低温度以保证确定性
            )
            
            content = completion.choices[0].message.content.strip()
            
            # 移除可能存在的 markdown 代码块标记
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content)
            return result
            
        except json.JSONDecodeError:
            # print(f"JSON 解析错误，尝试次数 {attempt+1}。内容: {content}")
            pass
        except Exception as e:
            print(f"API 调用错误，尝试次数 {attempt+1}: {e}")
            time.sleep(1) # 退避重试
            
    # 如果所有重试都失败
    return {"label": "error", "confidence": 0.0, "reasoning": "无法解析模型输出"}

def main():
    print("--- 开始自动标注任务 ---")
    
    # 1. 加载数据
    try:
        df = load_local_data()
    except Exception as e:
        print(f"严重错误: {e}")
        return

    print(f"已加载 {len(df)} 条样本。")
    
    # 2. 初始化客户端
    client = get_client()
    
    # 3. 批量推理
    results = []
    print(f"正在使用模型 {MODEL_NAME} 进行标注...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['text']
        true_label = row.get('label', None) 
        
        prediction = label_text(client, text)
        
        result_entry = {
            "text": text,
            "true_label": true_label,
            "model_label": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "reasoning": prediction.get("reasoning")
        }
        results.append(result_entry)
        
    # 4. 保存结果
    results_df = pd.DataFrame(results)
    output_path = "labeling_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n标注完成。结果已保存至 {output_path}")
    
    # 5. 简单分析
    print("\n--- 结果分析 ---")
    # 如果有真实标签，计算准确率
    if 'true_label' in results_df.columns:
        correct = 0
        total = 0
        for _, row in results_df.iterrows():
            if row['true_label'] is not None and row['model_label'] and row['model_label'] != "error":
                # 归一化比较：转换为字符串并处理可能的整数/字符串差异
                # 数据集中的 label 是 0 或 1
                t_label = str(row['true_label']).strip()
                m_label = str(row['model_label']).lower().strip()
                
                # 映射模型输出的 positive/negative 到 1/0
                if m_label == "positive":
                    m_label_int = "1"
                elif m_label == "negative":
                    m_label_int = "0"
                else:
                    m_label_int = "-1" # 未知
                
                if t_label == m_label_int:
                    correct += 1
                total += 1
        
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"准确率: {accuracy:.2f}% ({correct}/{total})")
        
    # 展示几个示例
    print("\n--- 输出示例 ---")
    print(results_df[['text', 'model_label', 'reasoning']].head(3).to_string())

if __name__ == "__main__":
    main()
