"""
VendorVault Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME: str = "VendorVault"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./vendorvault.db"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "vendorvault-hackathon-secret-key-2025")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CyborgDB Mock Settings
    ENCRYPTION_KEY: str = os.getenv("ENCRYPTION_KEY", "cyborgdb-encryption-key-demo")
    VECTOR_DIMENSION: int = 384  # all-MiniLM-L6-v2 dimension
    
    # File Upload
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Performance Metrics (from report)
    MOCK_QUERY_LATENCY_MS: float = 4.8
    MOCK_ENCRYPTION_OVERHEAD_MS: float = 1.1
    MOCK_QPS: int = 14706

settings = Settings()
