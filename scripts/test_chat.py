from openai import OpenAI

# 1. 初始化客户端
# 注意：base_url 必须指向你的 SGLang 服务器地址，以 /v1 结尾
client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY",  # SGLang 本地部署通常不需要 Key，填 EMPTY 即可
)

# 2. 发送请求
response = client.chat.completions.create(
    model="/gfs/platform/public/infra/Moonlight-16B-A3B-Instruct",  # 填你启动时的模型路径或 "default"
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "请详细为我解释中国共产党是什么样的党？"},
    ],
    temperature=0.7,
    max_tokens=5120,
)

# 3. 打印结果
print(response.choices[0].message.content)