from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()

async def fake_video_streamer():
    """
    模拟一个生成数据的生成器。
    它会一点一点地产生数据，而不是一次性返回。
    """
    for i in range(10):
        # 模拟数据处理耗时
        await asyncio.sleep(1) 
        yield f"Data chunk {i}\n"

@app.get("/")
async def main():
    return {"message": "Hello World"}

@app.get("/stream")
async def stream():
    
    return StreamingResponse(fake_video_streamer(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # 启动服务
    uvicorn.run(app, host="127.0.0.1", port=8000)