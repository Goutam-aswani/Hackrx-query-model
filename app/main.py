from fastapi import FastAPI
from app.api.router import router as hackrx_router

app = FastAPI(
    title="Intelligent Query–Retrieval System",
    description="API for the HackRx competition, built with FastAPI and LangChain.",
    version="1.0.0"
)

# Include the router from our api module
# We can add a prefix for versioning our API
app.include_router(hackrx_router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the HackRx Intelligent Query-Retrieval System API."}