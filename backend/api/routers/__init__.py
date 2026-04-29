from fastapi import APIRouter

from backend.api.routers import classify, company, mappings, obligations, score, search


api_router = APIRouter()
api_router.include_router(company.router, tags=["company"])
api_router.include_router(classify.router, tags=["classify"])
api_router.include_router(obligations.router, tags=["obligations"])
api_router.include_router(mappings.router, tags=["mappings"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(score.router, tags=["score"])
