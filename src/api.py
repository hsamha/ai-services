from fastapi import APIRouter, Depends

from src.features.llm.router import router as llm_router
from src.features.search.router import router as search_router
from src.headers import request_headers

api_router = APIRouter(prefix="/api/v1", dependencies=[Depends(request_headers)])
api_router.include_router(search_router)
api_router.include_router(llm_router)
