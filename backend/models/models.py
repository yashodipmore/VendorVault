"""
Database Models for VendorVault
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, LargeBinary, JSON
from sqlalchemy.sql import func
from app.database import Base

class Contract(Base):
    __tablename__ = "contracts"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    vendor_name = Column(String(255), index=True)
    contract_type = Column(String(100))  # Service, Product, Lease, etc.
    contract_value = Column(Float, default=0.0)
    currency = Column(String(10), default="USD")
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    status = Column(String(50), default="Active")  # Active, Expired, Pending
    
    # Extracted content
    raw_text = Column(Text)
    summary = Column(Text)
    key_terms = Column(JSON)  # JSON list of key terms
    
    # Encrypted vector embedding (simulating CyborgDB)
    encrypted_embedding = Column(LargeBinary)
    embedding_encrypted = Column(Boolean, default=True)
    
    # Anomaly detection
    anomaly_score = Column(Float, default=0.0)
    is_anomaly = Column(Boolean, default=False)
    
    # Clause classification
    clauses = Column(JSON)  # JSON with classified clauses
    
    # Metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    tenant_id = Column(String(100), default="default")
    department = Column(String(100))
    
    # Processing metrics
    processing_time_ms = Column(Float)
    encryption_time_ms = Column(Float)


class VectorIndex(Base):
    """Simulated CyborgDB Vector Index"""
    __tablename__ = "vector_index"
    
    id = Column(Integer, primary_key=True, index=True)
    contract_id = Column(Integer, index=True)
    chunk_id = Column(Integer)
    chunk_text = Column(Text)
    encrypted_vector = Column(LargeBinary)
    encryption_key_id = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())


class Vendor(Base):
    __tablename__ = "vendors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    category = Column(String(100))
    total_contracts = Column(Integer, default=0)
    total_value = Column(Float, default=0.0)
    avg_contract_duration = Column(Float)  # in months
    risk_score = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())


class SearchLog(Base):
    """Audit trail for searches"""
    __tablename__ = "search_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text)
    encrypted_query = Column(Boolean, default=True)
    results_count = Column(Integer)
    latency_ms = Column(Float)
    user_id = Column(String(100))
    tenant_id = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())


class SystemMetrics(Base):
    """Performance metrics storage"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100))
    metric_value = Column(Float)
    metric_unit = Column(String(50))
    recorded_at = Column(DateTime, server_default=func.now())
