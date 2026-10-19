"""
Lược đồ dữ liệu cho hiện vật và hình ảnh sử dụng Pydantic.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field

class Artifact(BaseModel):
    artifact_id: str
    name_vi: Optional[str] = None
    name_en: Optional[str] = None
    period: Optional[str] = None
    material: Optional[str] = None
    dimensions: Optional[str] = None
    origin_site: Optional[str] = None
    current_location: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    license: Optional[str] = None
    retrieved_at: datetime
    notes: Optional[str] = None
    knowledge_ref: Optional[str] = None
    spatial_context: Optional[dict] = None
    verification_status: Optional[str] = "unverified"
    inferred_metadata: Optional[dict] = None

class ImageInfo(BaseModel):
    image_id: str
    artifact_id: str
    file_path: str
    image_url: str
    author: Optional[str] = None
    license: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mime: Optional[str] = None
    sha256: Optional[str] = None
    phash: Optional[str] = None
    view_type: Optional[str] = None
    retrieved_at: datetime

