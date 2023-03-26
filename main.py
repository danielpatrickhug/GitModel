import openai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src import Pipeline

app = FastAPI()

# Add the CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRepoInput(BaseModel):
    config: str
    repo: str
    repo_name: str
    openai_secret: str


@app.post("/api/topic-modeling")
async def analyze_repo(input_data: AnalyzeRepoInput):
    openai_secret = input_data.openai_secret
    openai.api_key = openai_secret

    pipeline = Pipeline.from_yaml(input_data.config)
    try:
        gnn_head_outputs, topic_model_outputs = pipeline.run(input_data.repo, input_data.repo_name)
        if topic_model_outputs:
            topic_model_output = topic_model_outputs[0]
            return JSONResponse(content=topic_model_output["tree"])
        else:
            raise HTTPException(status_code=400, detail="Error processing topic model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
