import os
import time
from typing import Literal, Optional, Tuple, Set

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ConfigDict

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    Boolean,
    func,
    text as sql_text,
)
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://neondb_owner:npg_YyFPc4BEdu3N@ep-withered-sea-ahgolqjm.c-3.us-east-1.aws.neon.tech/neondb?sslmode=require"
)

TRIAGE_ARTIFACTS_PATH = os.getenv(
    "TRIAGE_ARTIFACTS_PATH",
    "./model/triage_artifacts.joblib",
)

SEVERE_SYMPTOMS_CSV = os.getenv(
    "SEVERE_SYMPTOMS_CSV",
    "./dataset/severe_symptoms.csv",
)

LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() in ("1", "true", "yes")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
ALLOW_LLM_DOWNGRADE = os.getenv("ALLOW_LLM_DOWNGRADE", "false").lower() in ("1", "true", "yes")
LLM_USE_OUTPUT_FIXING = os.getenv("LLM_USE_OUTPUT_FIXING", "true").lower() in ("1", "true", "yes")

LLM_OVERRIDE_CONF_THRESHOLD = float(os.getenv("LLM_OVERRIDE_CONF_THRESHOLD", "80"))

MODEL_READY: bool = False
MODEL = None
MLB = None

SEVERE_SYMPTOM_SET: Set[str] = set()
SEVERE_LIST_READY: bool = False

LLM_READY: bool = False
LLM = None

class Base(DeclarativeBase):
    pass


class TriageRecord(Base):
    __tablename__ = "triage_records"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    symptoms = Column(Text, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(16), nullable=False)
    severity = Column(String(16), nullable=False)
    duration_days = Column(Integer, nullable=False)

    
    prediction_class = Column(Integer, nullable=False)          
    recommendation = Column(String(64), nullable=False)         
    confidence_percent = Column(Float, nullable=False)          
    decision_source = Column(String(64), nullable=False)

    
    model_prediction_class = Column(Integer, nullable=True)
    model_recommendation = Column(String(64), nullable=True)
    model_confidence_percent = Column(Float, nullable=True)

    llm_used = Column(Boolean, nullable=False, default=False)
    llm_overrode = Column(Boolean, nullable=False, default=False)
    llm_user_explanation = Column(Text, nullable=True)  
    llm_corrected_output_explanation = Column(Text, nullable=True) 
    llm_internal_explanation = Column(Text, nullable=True)
    llm_model_name = Column(String(64), nullable=True)
    llm_latency_ms = Column(Float, nullable=True)


class FeedbackRecord(Base):
    __tablename__ = "feedback_records"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    rating = Column(Integer, nullable=True)
    feedback_text = Column(Text, nullable=True)


_engine = None
_SessionLocal = None


def init_db():
    """Initialize DB engine/session only if DATABASE_URL provided."""
    global _engine, _SessionLocal
    if not DATABASE_URL:
        return
    _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=_engine)

    ensure_llm_columns(_engine)


def ensure_llm_columns(engine):
    """Try to ALTER TABLE and add LLM columns if missing (best effort)."""
    try:
        with engine.begin() as conn:
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS model_prediction_class INTEGER"))
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS model_recommendation VARCHAR(64)"))
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS model_confidence_percent DOUBLE PRECISION"))

            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS llm_used BOOLEAN DEFAULT FALSE"))
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS llm_overrode BOOLEAN DEFAULT FALSE"))
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS llm_user_explanation TEXT"))
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS llm_corrected_output_explanation TEXT"))
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS llm_internal_explanation TEXT"))
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS llm_model_name VARCHAR(64)"))
            conn.execute(sql_text("ALTER TABLE triage_records ADD COLUMN IF NOT EXISTS llm_latency_ms DOUBLE PRECISION"))
    except Exception as e:
        print(f"[DB] ensure_llm_columns skipped/failed: {e}")


def get_db():
    if _SessionLocal is None:
        yield None
        return
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()

def split_and_clean_symptoms(x) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    parts = str(x).lower().split(",")
    parts = [p.strip() for p in parts if p.strip()]
    return list(set(parts))


def encode_age_value(age: int) -> int:
    age = int(age)
    if age <= 5:
        return 0
    elif 6 <= age <= 15:
        return 1
    elif 16 <= age <= 45:
        return 2
    elif 46 <= age <= 60:
        return 3
    else:
        return 4


def encode_duration_days(duration_days: int) -> int:
    d = int(duration_days)
    return 0 if d < 3 else 1


def mapping_sev_value(severity: str) -> int:
    sev_map = {"Mild": 0, "Moderate": 1}
    if severity not in sev_map:
        raise ValueError("Severity must be Mild/Moderate for model encoding.")
    return sev_map[severity]


def mapping_gen_value(gender: str) -> int:
    gen_map = {"Female": 0, "Male": 1}
    if gender not in gen_map:
        raise ValueError("Gender must be Male/Female.")
    return gen_map[gender]

def load_severe_symptom_set(csv_path: str) -> Set[str]:
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    if "Severe_Symptoms" not in df.columns:
        return set()

    severe_set: Set[str] = set()
    for cell in df["Severe_Symptoms"].dropna().astype(str).tolist():
        severe_set.update(split_and_clean_symptoms(cell))
    severe_set.discard("")
    return severe_set

def build_model_dataframe(
    symptoms_str: str,
    age: int,
    gender: str,
    severity: str,
    duration_days: int,
) -> pd.DataFrame:
    if MODEL is None or MLB is None:
        raise RuntimeError("Model artifacts are not loaded.")

    symptoms_list = split_and_clean_symptoms(symptoms_str)

    enc = MLB.transform([symptoms_list])
    enc_dense = enc.toarray() if hasattr(enc, "toarray") else enc
    symptom_df = pd.DataFrame(enc_dense, columns=MLB.classes_)

    row = {
        "age": encode_age_value(age),
        "gender": mapping_gen_value(gender),
        "duration": encode_duration_days(duration_days),
        "severity": mapping_sev_value(severity),  # Mild/Moderate only
    }
    meta_df = pd.DataFrame([row])

    return pd.concat([symptom_df, meta_df], axis=1)


def model_predict_with_confidence(X: pd.DataFrame) -> Tuple[int, float]:
    """
    Returns:
      pred_class: 1=Doctor Consultation, 0=Drug
      confidence_percent: probability of pred_class * 100
    """
    pred = int(MODEL.predict(X)[0])
    proba = MODEL.predict_proba(X)[0]
    classes = list(getattr(MODEL, "classes_", []))

    idx_1 = classes.index(1) if 1 in classes else min(1, len(proba) - 1)
    idx_0 = classes.index(0) if 0 in classes else 0

    doctor_conf = float(proba[idx_1]) * 100.0
    drug_conf = float(proba[idx_0]) * 100.0

    conf = doctor_conf if pred == 1 else drug_conf
    return pred, conf

class LLMReviewResult(BaseModel):
    approve_model_decision: bool = Field(..., description="True if model decision is reasonable/safe.")
    final_decision: Literal["Doctor Consultation", "Drug"] = Field(..., description="Final decision after review.")
    brief_reason: str = Field(..., description="Short justification (1-3 lines). No drugs.")


class LLMChangeExplanation(BaseModel):
    llm_corrected_output_explanation: str = Field(..., description="DB-only: why decision changed. No drugs.")


class LLMUserExplanation(BaseModel):
    user_explanation: str = Field(..., description="Short explanation shown to user. No drugs. Not a diagnosis.")


class LLMRuleExplanation(BaseModel):
    user_explanation: str = Field(..., description="Shown to user. No drugs. Not a diagnosis.")
    internal_explanation: str = Field(..., description="DB-only explanation. No drugs.")


MODEL_REVIEW_PROMPT = PromptTemplate(
    input_variables=[
        "age", "gender", "severity", "duration_days", "symptoms_list",
        "model_recommendation", "model_confidence_percent", "format_instructions"
    ],
    template=(
        "You are a clinical safety reviewer for a symptom-based triage system.\n"
        "You must be conservative: if anything seems risky, uncertain, or potentially serious, choose Doctor Consultation.\n"
        "Do NOT recommend any drugs or dosages. Do NOT claim a definitive diagnosis.\n\n"
        "Patient input:\n"
        "- Age: {age}\n"
        "- Gender: {gender}\n"
        "- Severity: {severity}\n"
        "- Duration (days): {duration_days}\n"
        "- Symptoms (clean list): {symptoms_list}\n\n"
        "ML model output:\n"
        "- Recommendation: {model_recommendation}\n"
        "- Confidence (model): {model_confidence_percent:.2f}%\n\n"
        "Task:\n"
        "1) Decide if the model recommendation is reasonable and safe given the input.\n"
        "2) If not safe/reasonable, override with the safer choice.\n\n"
        "{format_instructions}\n"
    ),
)

MODEL_CHANGE_EXPLAIN_PROMPT = PromptTemplate(
    input_variables=[
        "age", "gender", "severity", "duration_days", "symptoms_list",
        "model_recommendation", "final_decision", "format_instructions"
    ],
    template=(
        "You are writing a DB-only explanation for why the system changed the ML model decision.\n"
        "Rules:\n"
        "- No drug recommendations.\n"
        "- Mention the key symptom(s)/risk factors and safety rationale.\n"
        "- Keep it concise (2-5 sentences).\n\n"
        "Patient input:\n"
        "- Age: {age}, Gender: {gender}, Severity: {severity}, Duration: {duration_days} days\n"
        "- Symptoms: {symptoms_list}\n\n"
        "ML model recommendation: {model_recommendation}\n"
        "Final decision after LLM review: {final_decision}\n\n"
        "{format_instructions}\n"
    ),
)

RULE_SEVERITY_EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["age", "gender", "severity", "duration_days", "symptoms_list", "format_instructions"],
    template=(
        "You are generating an explanation for a triage decision made by a RULE.\n"
        "Decision is Doctor Consultation because Severity is 'Severe'.\n"
        "Rules:\n"
        "- No drug recommendations.\n"
        "- No definitive diagnosis.\n"
        "- Provide a short user explanation + a separate internal explanation.\n\n"
        "Patient:\n"
        "- Age: {age}, Gender: {gender}\n"
        "- Severity: {severity}\n"
        "- Duration: {duration_days} days\n"
        "- Symptoms: {symptoms_list}\n\n"
        "{format_instructions}\n"
    ),
)

RULE_SEVERE_SYMPTOM_EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["age", "gender", "severity", "duration_days", "symptoms_list", "matched_severe_symptoms", "format_instructions"],
    template=(
        "You are generating an explanation for a triage decision made by a RULE.\n"
        "Decision is Doctor Consultation because one or more severe/red-flag symptoms were detected.\n"
        "Rules:\n"
        "- No drug recommendations.\n"
        "- No definitive diagnosis.\n"
        "- Mention which symptom(s) triggered the rule.\n"
        "- Provide a short user explanation + a separate internal explanation.\n\n"
        "Patient:\n"
        "- Age: {age}, Gender: {gender}\n"
        "- Severity: {severity}\n"
        "- Duration: {duration_days} days\n"
        "- Symptoms: {symptoms_list}\n"
        "- Matched severe symptom(s): {matched_severe_symptoms}\n\n"
        "{format_instructions}\n"
    ),
)

USER_EXPLAIN_PROMPT = PromptTemplate(
    input_variables=["age", "gender", "severity", "duration_days", "symptoms_list", "final_decision", "format_instructions"],
    template=(
        "You are generating a short explanation for the USER for a symptom-based triage system.\n"
        "Rules:\n"
        "- No drug recommendations.\n"
        "- No definitive diagnosis.\n"
        "- Use careful language: 'may', 'could', 'possible'.\n"
        "- Keep it short (3-6 lines).\n"
        "- If decision is Doctor Consultation, explain why doctor visit is safer.\n"
        "- If decision is Drug, explain why it seems minor and suggest monitoring and seeking care if worse.\n\n"
        "Patient:\n"
        "- Age: {age}, Gender: {gender}\n"
        "- Severity: {severity}\n"
        "- Duration: {duration_days} days\n"
        "- Symptoms: {symptoms_list}\n\n"
        "Final decision: {final_decision}\n\n"
        "{format_instructions}\n"
    ),
)


def _llm_invoke_structured(prompt: PromptTemplate, parser: PydanticOutputParser, variables: dict):
    """Runs LLM and parses output into a Pydantic object (best effort)."""
    if not LLM_READY or LLM is None:
        raise RuntimeError("LLM is not ready.")

    variables = dict(variables)
    variables["format_instructions"] = parser.get_format_instructions()

    prompt_value = prompt.format_prompt(**variables)
    prompt_text = prompt_value.to_string()

    msg = LLM.invoke(prompt_text)
    content = getattr(msg, "content", str(msg))

    if LLM_USE_OUTPUT_FIXING:
        fixing = OutputFixingParser.from_llm(parser=parser, llm=LLM)
        return fixing.parse_with_prompt(content, prompt_value)

    return parser.parse(content)


ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="Medical Triage System API (v2 + LLM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageInput(BaseModel):
    symptoms: str = Field(..., description="Comma-separated symptoms, e.g. 'fever, cough'")
    age: int = Field(..., ge=0, le=120, description="Age in years (numeric)")
    gender: Literal["Male", "Female"]
    severity: Literal["Mild", "Moderate", "Severe"]
    duration: int = Field(..., ge=0, le=365, description="Duration in days (numeric)")

    @field_validator("symptoms")
    @classmethod
    def symptoms_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("symptoms cannot be empty")
        return v


class TriageOutput(BaseModel):
    prediction_id: int
    recommendation: Literal["Doctor Consultation", "Drug"]
    prediction_class: int  # 1 Doctor, 0 Drug
    confidence_percent: float = Field(..., ge=0.0, le=100.0)

    decision_source: Literal["severity_rule", "severe_symptom_rule", "model", "llm_override"]
    llm_used: bool
    llm_overrode: bool
    user_explanation: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": 12,
                "recommendation": "Doctor Consultation",
                "prediction_class": 1,
                "confidence_percent": 78.20,
                "decision_source": "llm_override",
                "llm_used": True,
                "llm_overrode": True,
                "user_explanation": "Your symptoms may indicate a condition that could require clinical evaluation...",
            }
        }
    )


class FeedbackInput(BaseModel):
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_text: Optional[str] = Field(None)

    def model_post_init(self, __context):
        if self.rating is None and (self.feedback_text is None or not self.feedback_text.strip()):
            raise ValueError("At least one of rating or feedback_text must be provided")


class FeedbackOutput(BaseModel):
    id: int
    message: str


@app.on_event("startup")
def startup():
    global MODEL_READY, MODEL, MLB, SEVERE_SYMPTOM_SET, SEVERE_LIST_READY
    global LLM_READY, LLM

    # DB
    init_db()

    # Artifacts
    try:
        artifacts = joblib.load(TRIAGE_ARTIFACTS_PATH)
        MODEL = artifacts.get("model")
        MLB = artifacts.get("symptom_mlb")
        MODEL_READY = (MODEL is not None) and (MLB is not None)
    except Exception as e:
        MODEL_READY = False
        MODEL = None
        MLB = None
        print(f"[startup] Failed to load artifacts: {e}")

    
    try:
        SEVERE_SYMPTOM_SET = load_severe_symptom_set(SEVERE_SYMPTOMS_CSV)
        SEVERE_LIST_READY = len(SEVERE_SYMPTOM_SET) > 0
    except Exception as e:
        SEVERE_SYMPTOM_SET = set()
        SEVERE_LIST_READY = False
        print(f"[startup] Failed to load severe symptoms CSV: {e}")

    
    try:
        if LLM_ENABLED and os.getenv("CEREBRAS_API_KEY"):
            LLM = ChatOpenAI(
                model=os.getenv("LLM_MODEL", LLM_MODEL),
                base_url=os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"),
                api_key=os.environ["CEREBRAS_API_KEY"],
                temperature=float(os.getenv("LLM_TEMPERATURE", str(LLM_TEMPERATURE))),
                use_responses_api=False, 
            )
            LLM_READY = True
        else:
            LLM = None
            LLM_READY = False
    except Exception as e:
        LLM = None
        LLM_READY = False
        print(f"[startup] Failed to init LLM: {e}")

    print(
        f"[startup] MODEL_READY={MODEL_READY}, "
        f"SEVERE_LIST_READY={SEVERE_LIST_READY}, severe_count={len(SEVERE_SYMPTOM_SET)}, "
        f"LLM_READY={LLM_READY}, LLM_MODEL={LLM_MODEL}, "
        f"OVERRIDE_IF_CONF_LT={LLM_OVERRIDE_CONF_THRESHOLD}"
    )


@app.get("/")
def root():
    return {
        "message": "Medical Triage System API (v2 + LLM)",
        "endpoints": {"health": "/health", "predict": "/predict", "feedback": "/feedback"},
    }


@app.get("/health")
def health_check():
    status = "healthy" if MODEL_READY else "degraded"
    return {
        "status": status,
        "model_loaded": MODEL_READY,
        "mlb_loaded": MODEL_READY,
        "severe_list_loaded": SEVERE_LIST_READY,
        "severe_symptom_count": len(SEVERE_SYMPTOM_SET),
        "llm_ready": LLM_READY,
        "llm_model": LLM_MODEL if LLM_READY else None,
        "allow_llm_downgrade": ALLOW_LLM_DOWNGRADE,
        "llm_override_conf_threshold": LLM_OVERRIDE_CONF_THRESHOLD,
    }


@app.post("/predict", response_model=TriageOutput)
def predict(data: TriageInput, db: Session = Depends(get_db)) -> TriageOutput:
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded. Check /health.")

    symptoms_list = split_and_clean_symptoms(data.symptoms)
    matched = set(symptoms_list).intersection(SEVERE_SYMPTOM_SET)
    matched_severe_symptoms_str = ", ".join(sorted(matched)) if matched else ""


    base_source = "model"
    base_class = None
    base_rec = None
    base_conf = None

    if data.severity == "Severe":
        base_source = "severity_rule"
        base_class = 1
        base_rec = "Doctor Consultation"
        base_conf = 100.0
    elif matched:
        base_source = "severe_symptom_rule"
        base_class = 1
        base_rec = "Doctor Consultation"
        base_conf = 100.0
    else:
        try:
            X = build_model_dataframe(
                symptoms_str=data.symptoms,
                age=data.age,
                gender=data.gender,
                severity=data.severity,   
                duration_days=data.duration,
            )
            base_class, base_conf = model_predict_with_confidence(X)
            base_rec = "Doctor Consultation" if base_class == 1 else "Drug"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    final_source = base_source
    final_class = int(base_class)
    final_rec = str(base_rec)
    final_conf = float(round(base_conf, 2))

    llm_used = False
    llm_overrode = False
    llm_user_explanation = ""
    llm_corrected_output_explanation = None
    llm_internal_explanation = None
    llm_latency_ms = None

    if LLM_READY:
        llm_used = True
        t0 = time.time()

        try:
           
            if base_source == "severity_rule":
                parser = PydanticOutputParser(pydantic_object=LLMRuleExplanation)
                out: LLMRuleExplanation = _llm_invoke_structured(
                    RULE_SEVERITY_EXPLAIN_PROMPT,
                    parser,
                    {
                        "age": data.age,
                        "gender": data.gender,
                        "severity": data.severity,
                        "duration_days": data.duration,
                        "symptoms_list": symptoms_list,
                    },
                )
                llm_user_explanation = out.user_explanation
                llm_internal_explanation = out.internal_explanation

            elif base_source == "severe_symptom_rule":
                parser = PydanticOutputParser(pydantic_object=LLMRuleExplanation)
                out: LLMRuleExplanation = _llm_invoke_structured(
                    RULE_SEVERE_SYMPTOM_EXPLAIN_PROMPT,
                    parser,
                    {
                        "age": data.age,
                        "gender": data.gender,
                        "severity": data.severity,
                        "duration_days": data.duration,
                        "symptoms_list": symptoms_list,
                        "matched_severe_symptoms": matched_severe_symptoms_str or "N/A",
                    },
                )
                llm_user_explanation = out.user_explanation
                llm_internal_explanation = out.internal_explanation


            else:
                can_override = float(base_conf) < float(LLM_OVERRIDE_CONF_THRESHOLD)

                if can_override:
                    review_parser = PydanticOutputParser(pydantic_object=LLMReviewResult)
                    review: LLMReviewResult = _llm_invoke_structured(
                        MODEL_REVIEW_PROMPT,
                        review_parser,
                        {
                            "age": data.age,
                            "gender": data.gender,
                            "severity": data.severity,
                            "duration_days": data.duration,
                            "symptoms_list": symptoms_list,
                            "model_recommendation": base_rec,
                            "model_confidence_percent": float(base_conf),
                        },
                    )

                    proposed_final = review.final_decision
                    proposed_final_class = 1 if proposed_final == "Doctor Consultation" else 0

                    if (not ALLOW_LLM_DOWNGRADE) and (base_class == 1) and (proposed_final_class == 0):
                        proposed_final = base_rec
                        proposed_final_class = base_class

                    if proposed_final != base_rec:
                        llm_overrode = True
                        final_source = "llm_override"
                        final_rec = proposed_final
                        final_class = proposed_final_class

                        
                        final_conf = float(round(base_conf, 2))

                        
                        change_parser = PydanticOutputParser(pydantic_object=LLMChangeExplanation)
                        change_out: LLMChangeExplanation = _llm_invoke_structured(
                            MODEL_CHANGE_EXPLAIN_PROMPT,
                            change_parser,
                            {
                                "age": data.age,
                                "gender": data.gender,
                                "severity": data.severity,
                                "duration_days": data.duration,
                                "symptoms_list": symptoms_list,
                                "model_recommendation": base_rec,
                                "final_decision": final_rec,
                            },
                        )
                        llm_corrected_output_explanation = change_out.llm_corrected_output_explanation

            
                user_parser = PydanticOutputParser(pydantic_object=LLMUserExplanation)
                user_out: LLMUserExplanation = _llm_invoke_structured(
                    USER_EXPLAIN_PROMPT,
                    user_parser,
                    {
                        "age": data.age,
                        "gender": data.gender,
                        "severity": data.severity,
                        "duration_days": data.duration,
                        "symptoms_list": symptoms_list,
                        "final_decision": final_rec,
                    },
                )
                llm_user_explanation = user_out.user_explanation

        except Exception as e:
            print(f"[LLM] failed: {e}")
            if not llm_user_explanation:
                if final_rec == "Doctor Consultation":
                    llm_user_explanation = (
                        "Based on your input, it may be safer to consult a doctor. "
                        "If symptoms worsen or you feel very unwell, seek urgent care. (Not a diagnosis.)"
                    )
                else:
                    llm_user_explanation = (
                        "Your symptoms may be mild, but monitor closely. "
                        "If symptoms worsen, last longer, or new severe symptoms appear, consult a doctor. (Not a diagnosis.)"
                    )
        finally:
            llm_latency_ms = (time.time() - t0) * 1000.0

    else:
        
        if final_rec == "Doctor Consultation":
            llm_user_explanation = (
                "Based on your input, doctor consultation is recommended for safer evaluation. "
                "If symptoms worsen, seek urgent care. (Not a diagnosis.)"
            )
        else:
            llm_user_explanation = (
                "Your symptoms may be minor. Monitor your condition and consult a doctor if symptoms worsen or persist. (Not a diagnosis.)"
            )


    record_id = -1
    if db is not None:
        try:
            record = TriageRecord(
                symptoms=data.symptoms,
                age=int(data.age),
                gender=str(data.gender),
                severity=str(data.severity),
                duration_days=int(data.duration),

                # final
                prediction_class=int(final_class),
                recommendation=str(final_rec),
                confidence_percent=float(final_conf),
                decision_source=str(final_source),

                # model/base
                model_prediction_class=int(base_class) if base_class is not None else None,
                model_recommendation=str(base_rec) if base_rec is not None else None,
                model_confidence_percent=float(round(base_conf, 2)) if base_conf is not None else None,

                # llm
                llm_used=bool(llm_used),
                llm_overrode=bool(llm_overrode),
                llm_user_explanation=llm_user_explanation,
                llm_corrected_output_explanation=llm_corrected_output_explanation,
                llm_internal_explanation=llm_internal_explanation,
                llm_model_name=LLM_MODEL if llm_used else None,
                llm_latency_ms=float(round(llm_latency_ms, 2)) if llm_latency_ms is not None else None,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            record_id = record.id
        except Exception as e:
            db.rollback()
            print(f"[DB] Failed to insert triage record: {e}")
            record_id = -1

    return TriageOutput(
        prediction_id=record_id,
        recommendation=final_rec,
        prediction_class=final_class,
        confidence_percent=final_conf,
        decision_source=final_source,
        llm_used=llm_used,
        llm_overrode=llm_overrode,
        user_explanation=llm_user_explanation,
    )


@app.post("/feedback", response_model=FeedbackOutput)
def submit_feedback(data: FeedbackInput, db: Session = Depends(get_db)) -> FeedbackOutput:
    if db is None:
        raise HTTPException(status_code=503, detail="Database is not configured (DATABASE_URL missing).")

    try:
        feedback = FeedbackRecord(
            rating=data.rating,
            feedback_text=data.feedback_text.strip() if data.feedback_text else None,
        )
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        return FeedbackOutput(id=feedback.id, message="Feedback submitted successfully")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_llm:app", host="0.0.0.0", port=8000, reload=True)
