# config.py

ORIGINS = [
    "http://localhost:5173", 
]

CORS_CONFIG = {
    "allow_origins": ORIGINS,
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
