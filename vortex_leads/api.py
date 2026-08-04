from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
import json
from vortex_leads.qualifier import LeadQualifier
from vortex_leads.store import LeadStore

# Initialize components
qualifier = LeadQualifier()
store = LeadStore(os.path.expanduser("~/vortex_leads.db"))

# Create FastAPI app
app = FastAPI(title="VORTEX Lead System API", description="API para gestão de leads VORTEX")

# Pydantic models for request/response
class Lead(BaseModel):
    id: Optional[int] = None
    name: str
    email: str
    phone: Optional[str] = None
    lead_type: str
    area_interests: Optional[List[str]] = None
    message: Optional[str] = None
    level: int
    price: int
    status: str = "novo"
    created_at: Optional[datetime] = None

class LeadCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    lead_type: str
    area_interests: Optional[List[str]] = None
    message: Optional[str] = None
    level: int
    price: int
    status: str = "novo"

class LeadUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    lead_type: Optional[str] = None
    area_interests: Optional[List[str]] = None
    message: Optional[str] = None
    level: Optional[int] = None
    price: Optional[int] = None
    status: Optional[str] = None

class FormData(BaseModel):
    nome: str
    email: str
    telefone: Optional[str] = None
    tipoConsulta: str
    areaInteresse: Optional[str] = None
    mensagem: Optional[str] = None

# API Routes
@app.post("/leads", response_model=Lead)
async def create_lead(lead: LeadCreate):
    lead_id = store.create(lead.dict())
    return {**lead.dict(), "id": lead_id}

@app.get("/leads", response_model=List[Lead])
async def list_leads():
    return store.list()

@app.get("/leads/{lead_id}", response_model=Lead)
async def get_lead(lead_id: int):
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead

@app.put("/leads/{lead_id}", response_model=Lead)
async def update_lead(lead_id: int, lead_update: LeadUpdate):
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    update_data = lead_update.dict(exclude_unset=True)
    store.update(lead_id, update_data)
    return {**lead, **update_data, "id": lead_id}

@app.delete("/leads/{lead_id}")
async def delete_lead(lead_id: int):
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    store.delete(lead_id)
    return {"message": "Lead deleted successfully", "id": lead_id}

@app.post("/webhook/netlify-form")
async def netlify_form_webhook(form_data: FormData):
    try:
        # Extract lead from form data
        lead = qualifier.extract_lead_from_form(form_data.dict())
        
        # Create lead in database
        lead_id = store.create(lead)
        
        # Trigger automated processes (future integration)
        # This could trigger email outreach, Slack notifications, etc.
        
        return {
            "success": True,
            "message": "Lead received and processed successfully",
            "lead_id": lead_id,
            "level": lead["level"],
            "price": lead["price"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)